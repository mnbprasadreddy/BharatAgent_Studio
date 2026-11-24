"""Agent stage: retrieves chunks from the sources drawn into the agent (and only those), runs the tools drawn
into the agent when the router asks for them, then asks the chosen model (through generate(), with retry and
local fallback) to answer from the chunks and tool results with [n] citations, streaming the answer as token
events while it is written. Each cited sentence is checked against what it cites (numbers and meaning) and
dropped if unsupported. Every question's model calls (tool plan + answer) are added up into one usage row
with tokens and estimated cost. If nothing supported is left, the agent says it does not know."""

import dataclasses
import time
from collections.abc import Iterator

from ... import usage
from ...config import settings
from ...models import generate, pricing
from ...models.generate import LANGUAGE_NOTES, AllProvidersFailed, is_local
from ...rag import citations, claims, conversation, embeddings, prompt, store
from ...rag.retrieve import BROAD_MAX, retrieve
from ...rag.store import Chunk
from ...tools import planner, runner
from ..context import RunContext
from ..graph import SOURCE_TYPES, TOOL_TYPES, Graph, Node


def _log(ctx: RunContext, **row) -> None:
    """Write the usage row; a logging failure is reported on the answer but never loses the answer."""
    if ctx.user_id is None:
        return
    try:
        usage.log(user_id=ctx.user_id, sources=[n.id for n in ctx.sources + ctx.tools], language=ctx.language,
                  **ctx.owner, **row)
    except Exception as exc:
        ctx.warnings.append(f"Usage was not logged: {exc}")


def _account(ctx: RunContext) -> None:
    """Add up every model call of this question (tool plan + answer): tokens, cost, time; one usage row."""
    gens = ctx.generations
    final = gens[-1]
    tokens_in, tokens_out = sum(g.input_tokens for g in gens), sum(g.output_tokens for g in gens)
    costs = [pricing.cost(g.model_id, g.input_tokens, g.output_tokens, is_local(g.model)) for g in gens]
    priced = all(c["priced"] for c in costs)
    usd = round(sum(c["usd"] for c in costs), 8) if priced else None
    local_ms = sum(g.response_ms for g in gens if is_local(g.model))
    ctx.generation = {**final.to_dict(), "input_tokens": tokens_in, "output_tokens": tokens_out,
                      "tokens_estimated": any(g.tokens_estimated for g in gens),
                      "response_ms": sum(g.response_ms for g in gens),
                      "fallback": next((g.fallback for g in gens if g.fallback), None),
                      "steps": [{"step": s, "model": g.label, "input_tokens": g.input_tokens,
                                 "output_tokens": g.output_tokens, "ms": g.response_ms}
                                for s, g in zip(ctx.generation_steps, gens)]}
    ctx.cost = {"usd": usd, "inr": round(usd * settings.usd_to_inr, 6) if usd is not None else None,
                "priced": priced, "api_fee": not all(is_local(g.model) for g in gens), "local_compute_ms": local_ms}
    ctx.comparison = pricing.comparison([(k, m, is_local(k)) for k, m in generate.model_ids().items()],
                                        tokens_in, tokens_out)
    note = LANGUAGE_NOTES.get(final.model)
    ctx.model_note = note if note and ctx.language in ("te", "hi") else None
    _log(ctx, model=final.label, model_requested=ctx.model_key, input_tokens=tokens_in, output_tokens=tokens_out,
         tokens_estimated=ctx.generation["tokens_estimated"], cost_usd=usd, cost_inr=ctx.cost["inr"],
         response_ms=ctx.generation["response_ms"], local_compute_ms=local_ms,
         fallback_from=ctx.generation["fallback"]["from"] if ctx.generation["fallback"] else None)


def _generate(ctx: RunContext, node: Node, system: str, user: str, step: str, stream_text: bool) -> Iterator[dict]:
    """One model call through generate(). With stream_text the pieces are forwarded as token events (the
    answer); otherwise they are only collected (the tool plan). "ms" is when the api received each piece,
    so a client can tell streaming from buffering. Returns the text."""
    start = time.perf_counter()
    text = ""
    try:
        for event in generate.stream(ctx.model_key, system, user, ctx.local_only):
            if "text" in event:
                text += event["text"]
                if stream_text:
                    yield {"type": "token", "stage": "agent", "node_id": node.id, "text": event["text"],
                           "ms": round((time.perf_counter() - start) * 1000)}
            elif "reset" in event:
                text = ""  # the model failed after writing; its partial text must not be shown
                if stream_text:
                    yield {"type": "token_reset", "stage": "agent", "node_id": node.id, **event["reset"]}
            elif "fallback" in event:
                yield {"type": "fallback", "stage": "agent", "node_id": node.id, **event["fallback"]}
            else:
                ctx.generations.append(event["done"])
                ctx.generation_steps.append(step)
    except AllProvidersFailed as exc:
        _log(ctx, model=f"{ctx.model_key}:{generate.model_ids()[ctx.model_key]}", model_requested=ctx.model_key,
             input_tokens=sum(g.input_tokens for g in ctx.generations),
             output_tokens=sum(g.output_tokens for g in ctx.generations), tokens_estimated=False, cost_usd=0,
             cost_inr=0, response_ms=round((time.perf_counter() - start) * 1000), local_compute_ms=0,
             status="error", error_code=exc.code, fallback_from=ctx.model_key if len(exc.attempts) > 1 else None)
        raise
    return ctx.generations[-1].text


def _use_tools(ctx: RunContext, node: Node, relevant: list[Chunk]) -> Iterator[dict]:
    """Plan and run tool calls. Yields tool_call / tool_result events (the canvas lights the tool node),
    and returns the results as numbered sources the answer can cite."""
    offered = runner.describe(ctx.tools)
    for unavailable in [t for t in offered if t.get("error")]:
        ctx.warnings.append(f"Tool {unavailable['node_id']} is unavailable: {unavailable['error']}")
    if not any(t.get("tool") for t in offered):
        return []
    prompt_text = planner.plan_prompt(_with_history(ctx), offered, [c.content for c in relevant])
    # Models sometimes reply without a readable plan (a reasoning model may return no text at all); ask once more.
    for attempt in range(planner.PLAN_ATTEMPTS):
        plan_text = yield from _generate(ctx, node, planner.PLANNER_SYSTEM, prompt_text, "tool_plan", stream_text=False)
        calls, problem = planner.parse(plan_text, offered)
        if not planner.unreadable(problem):
            break
    if problem:
        ctx.warnings.append(problem)
    connected = {n.id: n for n in ctx.tools}
    as_sources = []
    for call in calls:
        yield {"type": "tool_call", "stage": "agent", "node_id": call.node_id, "tool": call.tool, "args": call.args}
        result = runner.run(call, connected)
        ctx.tool_calls.append(result)
        yield {"type": "tool_result", "stage": "agent", **result}
        if result["ok"]:
            label = "calculator" if call.tool == "calculator" else f"{connected[call.node_id].config.get('server')}.{call.tool}"
            # A tool result becomes a numbered source the model must cite; negative ids have no stored vector,
            # so the claim check verifies its numbers only.
            as_sources.append(Chunk(-(len(as_sources) + 1), call.node_id, f"{label} (tool)", 0, None,
                                    f"Tool {label} was called with {call.args} and returned: {result['output']}", 1.0))
    return as_sources


def _searchable(ctx: RunContext) -> dict[str, str]:
    """Stored source id -> node id, for the connected sources this run may search. With a source map (every
    Phase 8 run) only the mapped, authorized sources; without one (unit tests) the node ids themselves."""
    if ctx.source_map is None:
        return {n.id: n.id for n in ctx.sources}
    return {ctx.source_map[n.id]: n.id for n in ctx.sources if n.id in ctx.source_map}


def _to_nodes(chunks: list[Chunk], searchable: dict[str, str]) -> list[Chunk]:
    """Report chunks under their node id, so citations and canvas highlights name the node."""
    return [dataclasses.replace(c, source_id=searchable.get(c.source_id, c.source_id)) for c in chunks]


def _rewrite(ctx: RunContext) -> str | None:
    """Turn a follow-up into a standalone search query with one short call to the same model as the answer
    (same local-only rule, so a local-only run never reaches a cloud model). Counted as its own step in tokens
    and cost. Returns None if the model is unavailable or gives nothing usable; the search then falls back to
    the previous question plus this one."""
    text = ""
    events = generate.stream(ctx.model_key, conversation.REWRITE_SYSTEM,
                             conversation.rewrite_prompt(ctx.question, ctx.history), ctx.local_only)
    try:
        for event in events:
            if "text" in event:
                text += event["text"]
            elif "reset" in event:
                text = ""
            elif "fallback" in event:
                # Not worth switching models for a search query (loading another local model can take a
                # minute): stop here and search with the earlier question instead. The answer step keeps
                # its own retry and fallback.
                events.close()
                ctx.warnings.append(f"The follow-up was not rewritten ({event['fallback']['code']}); searched with "
                                    "the earlier question instead.")
                return None
            elif "done" in event:
                ctx.generations.append(event["done"])
                ctx.generation_steps.append("query_rewrite")
    except AllProvidersFailed as exc:
        ctx.warnings.append(f"The follow-up could not be rewritten ({exc.code}); searched with the earlier question instead.")
        return None
    return conversation.clean_rewrite(text, ctx.question)


def _previous_evidence(ctx: RunContext, searchable: dict[str, str], query: str) -> list[Chunk]:
    """The chunks the previous answer cited, read again now: only from sources this run may search (deleted,
    unauthorized or re-pointed sources drop out), with their current text. Tool results are not reused."""
    node_to_source = {node: src for src, node in searchable.items()}
    keys, files = [], {}
    for cit in ctx.previous_citations:
        src = node_to_source.get(cit.get("source_id"))
        if src is not None and isinstance(cit.get("chunk"), int):
            keys.append((src, cit["chunk"]))
            files[(src, cit["chunk"])] = cit.get("file")
    if not keys:
        return []
    found = store.fetch(embeddings.embed_one(query), ctx.scope, keys)
    return [c for c in found if files.get((c.source_id, c.chunk_index)) == c.file_name]


def _retrieve(ctx: RunContext, searchable: dict[str, str]) -> tuple[list[Chunk], list[Chunk]]:
    """What to search for, and with what:
    - a standalone question is searched as it is (no extra model call, and no earlier topic mixed in);
    - a follow-up is rewritten into a standalone query, and both it and the question are searched;
    - a request to transform the previous answer (explain, simplify, example, summarise, translate, continue)
      also reuses the chunks that answer cited, re-checked against what this run may search;
    - a broad question gets several sections (bounded), not just the single closest chunk."""
    source_ids = list(searchable)
    follow = conversation.is_follow_up(ctx.question, ctx.history)
    queries, rewritten = [ctx.question], None
    if follow:
        rewritten = _rewrite(ctx)
        if rewritten:
            queries.insert(0, rewritten)
        else:
            earlier = [text for role, text, *_ in ctx.history if role == "user"]
            queries.insert(0, f"{earlier[-1][:300]} {ctx.question}")
    elif conversation.needs_subject(ctx.question) and ctx.source_map is not None:
        # A vague first question ("What are its main features?") has no earlier message to say what "its" is;
        # the project does: search it together with the names of the sources this run may search (no model call).
        # It only helps a matching chunk clear the threshold; the threshold, citations and claim check are unchanged.
        try:
            titles = store.source_titles(ctx.scope, source_ids)
        except Exception as exc:  # never let this extra search break the answer
            titles = []
            ctx.warnings.append(f"Project source names were not used for the search: {exc}")
        if titles:
            queries.insert(0, conversation.subject_query(ctx.question, titles))
    queries = list(dict.fromkeys(queries))
    broad = any(conversation.is_broad(q) for q in queries)
    limit = BROAD_MAX if broad else settings.top_k
    best: dict[int, Chunk] = {}
    pool: dict[int, Chunk] = {}
    for q in queries:
        relevant, candidates = retrieve(q, source_ids, ctx.scope, broad)
        for c in relevant:
            if c.id not in best or c.similarity > best[c.id].similarity:
                best[c.id] = c
        for c in candidates:
            if c.id not in pool or c.similarity > pool[c.id].similarity:
                pool[c.id] = c
    reused: list[Chunk] = []
    if follow and conversation.transforms(ctx.question) and ctx.previous_citations:
        reused = [c for c in _previous_evidence(ctx, searchable, queries[0]) if c.id not in best]
    ranked = sorted(best.values(), key=lambda c: c.similarity, reverse=True)
    relevant = (reused + ranked)[:max(limit, len(reused))]
    candidates = sorted(pool.values(), key=lambda c: c.similarity, reverse=True)[:limit]
    ctx.retrieval_info = {"follow_up": follow, "queries": queries, "rewrite": rewritten, "broad": broad,
                          "project_subject": not follow and len(queries) > 1,
                          "reused": [f"{c.file_name} chunk {c.chunk_index}" for c in reused]}
    ctx.detailed = broad or bool(conversation.transforms(ctx.question) - {"translate"})
    return relevant, candidates


def _with_history(ctx: RunContext) -> str:
    """The question, preceded by a bounded excerpt of the conversation so far (never more than a few messages,
    each cut to 600 characters). Retrieval and citations still come only from the current, authorized sources."""
    if not ctx.history:
        return ctx.question
    lines = [f"{'User' if role == 'user' else 'Assistant'}: {text[:600]}" for role, text in ctx.history]
    return "Earlier in this conversation:\n" + "\n".join(lines) + f"\n\nCurrent question: {ctx.question}"


def run(ctx: RunContext, node: Node, graph: Graph) -> Iterator[dict]:
    """A generator: yields token and tool events, and returns the stage detail when finished."""
    connected = graph.connected_to(node.id)
    ctx.sources = [n for n in connected if n.type in SOURCE_TYPES]
    ctx.tools = [n for n in connected if n.type in TOOL_TYPES]
    # The chat's model dropdown wins over the agent node's setting. Local-only can come from the request,
    # the agent node, or the server (LOCAL_ONLY in .env); any of them is enough.
    ctx.model_key = ctx.requested_model or node.config.get("model", "gemini")
    ctx.local_only = ctx.local_only or node.config.get("local_only") is True or settings.local_only
    detail = {
        "model_requested": ctx.model_key,
        "local_only": ctx.local_only,
        "sources": [{"id": n.id, "type": n.type} for n in ctx.sources],
        "tools": [{"id": n.id, "type": n.type} for n in ctx.tools],
    }

    def finish(**extra) -> dict:
        if ctx.generations:
            _account(ctx)
        return {**detail, **extra, "tool_calls": ctx.tool_calls, "generation": ctx.generation, "cost": ctx.cost}

    # Search only the connected sources; an unconnected source is never passed to retrieve().
    relevant: list[Chunk] = []
    if ctx.sources and ctx.route != "direct":
        searchable = _searchable(ctx)
        relevant, candidates = _retrieve(ctx, searchable)
        relevant, candidates = _to_nodes(relevant, searchable), _to_nodes(candidates, searchable)
        detail["retrieval"] = {
            **ctx.retrieval_info,
            "scope": ctx.scope,
            "searched": list(searchable.values()),
            "min_similarity": settings.min_similarity,
            "candidates": [{"source_id": c.source_id, "file": c.file_name, "chunk": c.chunk_index,
                            "similarity": round(c.similarity, 3)} for c in candidates],
            "relevant": len(relevant),
        }

    # Tools run only when the router chose them and only tools drawn into the agent are offered.
    tool_sources: list[Chunk] = []
    if ctx.route == "tool" and ctx.tools:
        tool_sources = yield from _use_tools(ctx, node, relevant)
    evidence = relevant + tool_sources

    # Direct answer: nothing to retrieve or compute; no citations, marked as not from your sources.
    if ctx.route == "direct" or (not ctx.sources and not tool_sources):
        yield from _generate(ctx, node, prompt.direct_system(ctx.language), _with_history(ctx), "answer", stream_text=True)
        ctx.answer = ctx.generations[-1].text
        ctx.grounded, ctx.supported, ctx.citations = False, False, []
        return finish(mode="direct", model_called=True)

    ctx.grounded = True
    if not evidence:
        ctx.answer, ctx.supported, ctx.citations = prompt.DONT_KNOW[ctx.language], False, []
        return finish(mode="rag", model_called=bool(ctx.generations),
                      reason="No chunk from the connected sources is relevant enough to answer.")

    yield from _generate(ctx, node, prompt.rag_system(ctx.language),
                         prompt.rag_prompt(_with_history(ctx), evidence, detailed=ctx.detailed, language=ctx.language), "answer", stream_text=True)
    ctx.answer = ctx.generations[-1].text
    checked = citations.check(ctx.answer, evidence)
    ctx.citations = checked["citations"]
    if citations.is_dont_know(ctx.answer) or not ctx.citations:
        # Either the model said it does not know, or it answered without citing a real chunk.
        ctx.answer, ctx.supported, ctx.citations = prompt.DONT_KNOW[ctx.language], False, []
        reason = ("The model found no answer in the sources." if citations.is_dont_know(checked["answer"])
                  else "The model's answer cited no real source chunk, so it was not shown.")
        return finish(mode="rag", model_called=True, reason=reason, invalid_markers=checked["invalid_markers"])
    # Each cited sentence must be backed by what it cites (numbers and meaning); others are removed.
    try:
        verified = claims.verify(checked["answer"], evidence)
    except Exception as exc:  # e.g. Ollama down: keep the cited answer, but say it was not checked
        ctx.warnings.append(f"Claims were not checked against the sources: {exc}")
        verified = {"answer": checked["answer"], "claims": [], "removed": 0, "checked": False}
    ctx.claims = verified["claims"]
    if not verified["answer"]:
        ctx.answer, ctx.supported, ctx.citations = prompt.DONT_KNOW[ctx.language], False, []
        return finish(mode="rag", model_called=True, claims=ctx.claims,
                      reason="No sentence of the model's answer was backed by the source it cited.")
    rechecked = citations.check(verified["answer"], evidence)  # keep only citations still used
    ctx.answer, ctx.citations, ctx.supported = rechecked["answer"], rechecked["citations"], bool(rechecked["citations"])
    return finish(mode="rag", model_called=True, invalid_markers=checked["invalid_markers"], claims=ctx.claims,
                  claims_removed=verified["removed"])
