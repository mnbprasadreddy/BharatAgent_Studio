"""The agent: retrieval only from connected sources, streamed tokens, checked citations, "I don't know"
when nothing supports an answer, fallback to the local model, and tokens and cost on every answer.
The provider clients themselves are tested in test_models.py."""

import pytest

from app.models import gemini
from app.pipeline.engine import run_pipeline
from app.pipeline.graph import Graph
from app.pipeline.stages import agent
from app.rag.prompt import DONT_KNOW
from app.rag.store import Chunk

REFUND_CHUNK = Chunk(1, "s1", "refund-policy.pdf", 0, 1,
                     "If a lab test is cancelled before the sample is collected, 80 percent is refunded.", 0.71)


def run(graph: dict, question: str, language: str | None = None) -> list[dict]:
    return list(run_pipeline(Graph.model_validate(graph), question, language))


@pytest.fixture
def spy_retrieve(monkeypatch):
    """Records which source ids were searched and returns REFUND_CHUNK for any search."""
    calls = []

    def fake(question, source_ids, scope="inline", broad=False):
        calls.append(list(source_ids))
        return ([REFUND_CHUNK], [REFUND_CHUNK]) if source_ids else ([], [])

    monkeypatch.setattr(agent, "retrieve", fake)
    return calls


def answer_with(monkeypatch, text: str):
    def fake(system, prompt):
        for word in text.split(" "):
            yield {"text": word + " "}
        yield {"usage": {"input_tokens": 50, "output_tokens": 9}, "finish_reason": "STOP"}
    monkeypatch.setattr(gemini, "stream", fake)


def only_s1(graph: dict) -> dict:
    """The handoff graph with s2 and both tools removed: only s1 is connected."""
    keep = {"n1", "n2", "n3", "n4", "n5", "s1"}
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] in keep]
    graph["edges"] = [e for e in graph["edges"] if e["from"] in keep]
    return graph


def test_cited_answer(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, "80 percent is refunded if cancelled before collection [1].")
    events = run(only_s1(full_graph), "lab test refund?")
    result = events[-1]
    assert result["supported"] is True and result["grounded"] is True
    assert result["citations"][0]["file"] == "refund-policy.pdf"
    assert result["citations"][0]["page"] == 1 and result["citations"][0]["chunk"] == 0
    assert result["usage"] == {"input_tokens": 50, "output_tokens": 9, "estimated": False}


def test_tokens_stream_between_agent_running_and_done(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, "80 percent is refunded [1].")
    events = run(only_s1(full_graph), "lab test refund?")
    kinds = [(e["type"], e.get("stage"), e.get("status")) for e in events]
    start = kinds.index(("stage", "agent", "running"))
    end = kinds.index(("stage", "agent", "done"))
    tokens = [e for e in events[start:end] if e["type"] == "token"]
    assert len(tokens) >= 4 and all(t["node_id"] == "n4" for t in tokens)
    assert all(isinstance(t["ms"], int) for t in tokens)
    assert [t["ms"] for t in tokens] == sorted(t["ms"] for t in tokens)


def test_only_connected_sources_are_searched(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, "Answer [1].")
    run(only_s1(full_graph), "refund?")
    assert spy_retrieve == [["s1"]]


def test_unconnected_source_never_searched(monkeypatch, spy_retrieve, full_graph):
    # s2 is in the graph but its line to the agent is removed: the graph is invalid and nothing runs.
    from app.pipeline.validate import validate_graph
    full_graph["edges"] = [e for e in full_graph["edges"] if e["from"] != "s2"]
    assert validate_graph(Graph.model_validate(full_graph))  # the API refuses to run this graph


def test_nothing_relevant_means_dont_know_without_model_call(monkeypatch, full_graph):
    monkeypatch.setattr(agent, "retrieve", lambda q, ids, scope="inline", broad=False: ([], [REFUND_CHUNK]))
    called = []
    monkeypatch.setattr(gemini, "stream", lambda s, p: called.append(1) or iter(()))
    events = run(only_s1(full_graph), "What time does the hostel gate close?")
    result = events[-1]
    assert result["answer"] == DONT_KNOW[result["language"]]
    assert result["supported"] is False and result["citations"] == []
    assert called == []
    detail = next(e for e in events if e.get("stage") == "agent" and e.get("status") == "done")["detail"]
    assert detail["model_called"] is False and detail["retrieval"]["relevant"] == 0


def test_answer_without_valid_citation_is_replaced(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, "The refund is 100 percent [4].")  # [4] does not exist
    result = run(only_s1(full_graph), "lab test refund?", "en")[-1]
    assert result["answer"] == DONT_KNOW["en"]
    assert result["supported"] is False and result["citations"] == []


def test_answer_with_no_citation_is_replaced(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, "The refund is 80 percent.")
    assert run(only_s1(full_graph), "lab test refund?", "en")[-1]["answer"] == DONT_KNOW["en"]


def test_model_says_dont_know(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, DONT_KNOW["te"])
    result = run(only_s1(full_graph), "lab test refund?", "te")[-1]
    assert result["answer"] == DONT_KNOW["te"] and result["supported"] is False


def test_invalid_markers_removed_from_shown_answer(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, "80 percent is refunded [1][6].")
    result = run(only_s1(full_graph), "lab test refund?", "en")[-1]
    assert "[6]" not in result["answer"] and "[1]" in result["answer"]


def test_direct_route_has_no_citations(monkeypatch, spy_retrieve, default_graph):
    result = run(default_graph, "Namaste!")[-1]
    assert result["grounded"] is False and result["citations"] == [] and spy_retrieve == []


def test_failed_gemini_falls_back_to_local(monkeypatch, default_graph):
    def broken(system, prompt, **kwargs):
        raise gemini.GeminiError("GEMINI_API_KEY is not set", code="no_key")
        yield  # pragma: no cover
    monkeypatch.setattr(gemini, "stream", broken)
    events = run(default_graph, "hello there")
    fallback = next(e for e in events if e["type"] == "fallback")
    assert fallback["from"] == "gemini" and fallback["to"] == "local" and fallback["code"] == "no_key"
    result = events[-1]
    assert result["type"] == "result" and result["model"].startswith("local:")
    assert result["fallback"]["from"] == "gemini" and result["model_requested"] == "gemini"


def test_all_models_failing_gives_one_clear_error(monkeypatch, default_graph):
    from app.models import ollama

    def broken(code):
        def stream(system, prompt, **kwargs):
            raise gemini.GeminiError(f"{code} happened", code=code)
            yield  # pragma: no cover
        return stream
    monkeypatch.setattr(gemini, "stream", broken("quota_daily"))
    monkeypatch.setattr(ollama, "stream", broken("network_error"))
    events = run(default_graph, "hello there")
    error = events[-1]
    assert error["type"] == "error" and error["code"] == "all_providers_failed" and error["stage"] == "agent"
    assert [(a["model"], a["code"]) for a in error["attempts"]] == [
        ("gemini", "quota_daily"), ("local", "network_error"), ("local_small", "network_error")]
    assert "Every model failed" in error["message"]
    assert not any(e["type"] == "result" for e in events)


def test_model_choice_from_request_beats_agent_setting(default_graph):
    events = list(run_pipeline(Graph.model_validate(default_graph), "hello there", None, "groq"))
    assert events[-1]["model"].startswith("groq:") and events[-1]["model_requested"] == "groq"


def test_every_answer_carries_tokens_and_cost(default_graph):
    result = run(default_graph, "hello there")[-1]
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2, "estimated": False}
    assert result["cost"]["priced"] is True and result["cost"]["usd"] > 0 and result["cost"]["api_fee"] is True
    assert {row["model"] for row in result["comparison"]} == {"gemini", "groq", "local", "local_small"}


def test_no_model_call_means_no_cost(monkeypatch, full_graph):
    monkeypatch.setattr(agent, "retrieve", lambda q, ids, scope="inline", broad=False: ([], []))
    result = run(only_s1(full_graph), "What time does the hostel gate close?")[-1]
    assert result["model"] is None and result["cost"] is None and result["usage"] is None


def test_answer_in_the_wrong_language_is_flagged(monkeypatch, spy_retrieve, full_graph):
    # The small local model answered Telugu questions in English in the evaluation: that must show.
    answer_with(monkeypatch, "The hostel gate closes at 9:30 PM every day [1].")
    result = run(only_s1(full_graph), "హాస్టల్ గేట్ ఎన్ని గంటలకు మూసివేస్తారు?")[-1]
    assert result["language"] == "te" and result["language_match"] is False
    assert any("not written in Telugu" in w for w in result["warnings"])


def test_answer_in_the_right_language_is_not_flagged(monkeypatch, spy_retrieve, full_graph):
    answer_with(monkeypatch, "80 percent is refunded [1].")
    result = run(only_s1(full_graph), "lab test refund?", "en")[-1]
    assert result["language_match"] is True and result["warnings"] == []


HANDOFF = Chunk(7, "s1", "handoff.pdf", 0, 1, "BharatAgent Studio is a local, ChatGPT-style assistant where you draw "
                "an AI agent pipeline and connect your own documents.", 0.72)
FIRST_ANSWER = ("This project is about building BharatAgent Studio for a hackathon, a local ChatGPT-style app where the "
                "user draws an AI agent pipeline on a canvas [1].")
REWRITTEN = "Detailed explanation of the BharatAgent Studio project"


def _searches(monkeypatch, hits_for):
    """A retrieve() that records each search query and answers from hits_for(query)."""
    searched = []

    def fake(question, source_ids, scope="inline", broad=False):
        searched.append(question)
        hits = hits_for(question)
        return hits, hits
    monkeypatch.setattr(agent, "retrieve", fake)
    return searched


def _models(monkeypatch, rewrite_text=REWRITTEN, answer="BharatAgent Studio is a local assistant where you draw an AI agent pipeline [1]."):
    """Gemini that writes the rewrite for the rewrite prompt and the answer otherwise; records each system prompt."""
    from app.rag import conversation
    calls = []

    def stream(system, prompt, **kw):
        calls.append("rewrite" if system == conversation.REWRITE_SYSTEM else "answer")
        yield {"text": rewrite_text if system == conversation.REWRITE_SYSTEM else answer}
        yield {"usage": {"input_tokens": 40, "output_tokens": 12}, "finish_reason": "STOP"}
    monkeypatch.setattr(gemini, "stream", stream)
    return calls


def test_reported_follow_up_is_rewritten_and_answered_not_declined(monkeypatch, full_graph):
    """The reported defect: after "What is this project about?", the follow-up "tell clerly about this allobrate
    the explination" was searched as typed (or glued to the previous question), matched nothing well enough and
    was answered "I don't know" without asking the model. Now it is rewritten into a standalone query first."""
    searched = _searches(monkeypatch, lambda q: [HANDOFF] if "BharatAgent Studio" in q else [])
    calls = _models(monkeypatch)
    history = [("user", "What is this project about?"), ("assistant", FIRST_ANSWER)]
    events = list(run_pipeline(Graph.model_validate(only_s1(full_graph)), "tell clerly about this allobrate the explination",
                               history=history))
    result = events[-1]
    agent_done = next(e for e in events if e.get("stage") == "agent" and e.get("status") == "done")["detail"]
    assert searched[0] == REWRITTEN and "tell clerly about this allobrate the explination" in searched
    assert agent_done["retrieval"]["rewrite"] == REWRITTEN and agent_done["retrieval"]["follow_up"] is True
    assert calls == ["rewrite", "answer"]
    assert result["supported"] is True and result["citations"] and "BharatAgent Studio" in result["answer"]
    steps = [s["step"] for s in agent_done["generation"]["steps"]]
    assert steps == ["query_rewrite", "answer"] and result["usage"]["input_tokens"] == 80  # both calls counted


def test_standalone_question_costs_no_extra_model_call(monkeypatch, full_graph):
    """A new, self-contained question in a conversation is searched as it is: no rewrite call, no earlier topic."""
    searched = _searches(monkeypatch, lambda q: [HANDOFF])
    calls = _models(monkeypatch)
    history = [("user", "Above what order value is delivery free?"), ("assistant", "Rs. 499 [1].")]
    run_pipeline_events = list(run_pipeline(Graph.model_validate(only_s1(full_graph)),
                                            "What is the Spoon-Knife repository meant for?", history=history))
    assert searched == ["What is the Spoon-Knife repository meant for?"] and calls == ["answer"]
    assert run_pipeline_events[-1]["type"] == "result"


def test_unusable_rewrite_falls_back_to_the_previous_question(monkeypatch, full_graph):
    searched = _searches(monkeypatch, lambda q: [HANDOFF] if "project" in q else [])
    _models(monkeypatch, rewrite_text="Sorry, I cannot help with that.")
    history = [("user", "What is this project about?"), ("assistant", FIRST_ANSWER)]
    events = list(run_pipeline(Graph.model_validate(only_s1(full_graph)), "Explain it in detail.", history=history))
    assert searched[0] == "What is this project about? Explain it in detail." and events[-1]["supported"] is True


def test_a_real_abstention_still_says_i_do_not_know(monkeypatch, full_graph):
    """Rewriting must not turn an unanswerable follow-up into an answer: nothing relevant, nothing said."""
    _searches(monkeypatch, lambda q: [])
    calls = _models(monkeypatch, rewrite_text="Monthly parking fee at the BharatAgent office")
    history = [("user", "What is this project about?"), ("assistant", FIRST_ANSWER)]
    result = list(run_pipeline(Graph.model_validate(only_s1(full_graph)), "And what is its parking fee?",
                               history=history))[-1]
    assert result["supported"] is False and result["answer"] in DONT_KNOW.values() and calls == ["rewrite"]


def test_transforming_an_answer_reuses_only_evidence_this_run_may_still_search(monkeypatch):
    """Reuse of the previous answer's chunks goes through the current source map: a cited source that is no
    longer searchable (deleted, unauthorized, not connected) is never fetched."""
    from app.pipeline.context import RunContext
    from app.rag import store
    asked = []

    def fetch(vector, scope, keys):
        asked.append(keys)
        return [Chunk(9, "src7", "handoff.pdf", 3, 1, "Feature two: every answer shows its cost.", 0.4)]
    monkeypatch.setattr(store, "fetch", fetch)
    monkeypatch.setattr(agent.embeddings, "embed_one", lambda text: [0.0])
    _searches(monkeypatch, lambda q: [])
    _models(monkeypatch, rewrite_text="Second feature of BharatAgent Studio")
    ctx = RunContext(question="Explain the second feature in simple words.", model_key="gemini")
    ctx.history = [("user", "What are its main features?"), ("assistant", "Cited answers [1]. Cost on every answer [2].")]
    ctx.previous_citations = [{"source_id": "doc", "file": "handoff.pdf", "chunk": 3},
                              {"source_id": "deleted_doc", "file": "old.pdf", "chunk": 1},
                              {"source_id": "t1", "file": "calculator (tool)", "chunk": 0}]
    relevant, _ = agent._retrieve(ctx, {"src7": "doc"})
    assert asked == [[("src7", 3)]]                     # only the source this run may search
    assert [c.chunk_index for c in relevant] == [3] and ctx.retrieval_info["reused"] == ["handoff.pdf chunk 3"]


def test_a_language_asked_for_in_the_question_is_used():
    from app.pipeline.context import RunContext
    from app.pipeline.graph import Node
    from app.pipeline.stages import input as input_stage
    agent_node = Node(id="n4", type="agent", config={"language": "en"})
    for question, lang in [("Explain the same thing in Telugu.", "te"), ("Explain it in Hindi.", "hi"),
                           ("telugu lo cheppu", "te"), ("What is this project about?", "en")]:
        ctx = RunContext(question=question)
        input_stage.run(ctx, agent_node, agent_node)
        assert ctx.language == lang, question
    ctx = RunContext(question="Explain it in Hindi.", requested_language="te")  # the chat selector still wins
    input_stage.run(ctx, agent_node, agent_node)
    assert ctx.language == "te"


def test_follow_up_detection_handles_typos_and_topic_changes():
    from app.rag import conversation
    history = [("user", "What is this project about?"), ("assistant", "It is BharatAgent Studio [1].")]
    for q in ["tell clerly about this allobrate the explination", "Explain it in detail.", "What are its main features?",
              "Give me an example.", "Summarize everything in five points.", "Continue the explanation.", "And the fee?"]:
        assert conversation.is_follow_up(q, history), q
    for q in ["What is the Spoon-Knife repository meant for?", "What is the monthly parking fee at the company office?",
              "Above what order value is delivery free on the e-commerce platform?"]:
        assert not conversation.is_follow_up(q, history), q
    assert not conversation.is_follow_up("Explain it in detail.", [])  # no conversation yet
    assert conversation.clean_rewrite('Query: "BharatAgent Studio overview"', "x") == "BharatAgent Studio overview"
    assert conversation.clean_rewrite("I don't know.", "x") is None


def test_rewrite_never_switches_models_it_falls_back_to_the_earlier_question(monkeypatch, full_graph):
    """If the chosen model cannot rewrite (rate limit, quota), the rewrite stops instead of loading another
    model just for a search query; the search uses the earlier question with this one."""
    from app.models import generate
    from app.rag import conversation
    searched = _searches(monkeypatch, lambda q: [HANDOFF] if "project" in q else [])
    real = generate.stream

    def stream(model, system, prompt, local_only=False):
        if system == conversation.REWRITE_SYSTEM:
            yield {"fallback": {"from": model, "to": "local", "code": "quota_minute", "reason": "429"}}
            raise AssertionError("the rewrite must stop at the first fallback")
        yield from real(model, system, prompt, local_only)
    monkeypatch.setattr(agent.generate, "stream", stream)
    _models(monkeypatch)
    history = [("user", "What is this project about?"), ("assistant", FIRST_ANSWER)]
    events = list(run_pipeline(Graph.model_validate(only_s1(full_graph)), "Explain it in detail.", history=history))
    assert searched[0] == "What is this project about? Explain it in detail." and events[-1]["supported"] is True
    assert any("not rewritten" in w for w in events[-1]["warnings"])
    assert conversation.clean_rewrite("BharatAgent Studio how it works\x00", "x") == "BharatAgent Studio how it works"


def test_detailed_answers_are_shorter_in_telugu_and_hindi():
    """A detailed Telugu answer of eight sentences hit the local model's output cap (1,550 tokens) and lost its
    citations; detailed answers in Telugu and Hindi ask for at most four sentences."""
    from app.rag import prompt
    assert "up to 8 full sentences" in prompt.rag_prompt("q", [HANDOFF], detailed=True, language="en")
    for lang in ("te", "hi"):
        assert "up to 4 full sentences" in prompt.rag_prompt("q", [HANDOFF], detailed=True, language=lang)
    assert "one or two full sentences" in prompt.rag_prompt("q", [HANDOFF], detailed=False, language="te")


def test_vague_first_question_is_searched_with_the_project_source_names(monkeypatch, full_graph):
    """Reported after the follow-up fix: "What are its main features?" as the FIRST message of a new chat matched
    nothing (0.425 < 0.50) because there was no earlier message to say what "its" is. The project's source names
    are now searched with it; no model call is added, and a run without sources the person may search adds nothing."""
    from app.rag import store
    searched = _searches(monkeypatch, lambda q: [HANDOFF] if "BharatAgent Studio Build Handoff" in q else [])
    asked = []
    monkeypatch.setattr(store, "source_titles", lambda scope, ids: asked.append(ids) or ["BharatAgent Studio Build Handoff _1_.pdf"])
    calls = _models(monkeypatch)
    events = list(run_pipeline(Graph.model_validate(only_s1(full_graph)), "What are its main features?",
                               source_map={"s1": "src208"}))
    retrieval = next(e for e in events if e.get("stage") == "agent" and e.get("status") == "done")["detail"]["retrieval"]
    assert searched[0] == "What are its main features? BharatAgent Studio Build Handoff 1"
    assert asked == [["src208"]] and retrieval["project_subject"] is True and calls == ["answer"]
    assert events[-1]["supported"] is True and events[-1]["citations"]


def test_specific_first_questions_and_unanswerable_ones_are_unchanged(monkeypatch, full_graph):
    from app.rag import conversation, store
    searched = _searches(monkeypatch, lambda q: [])
    monkeypatch.setattr(store, "source_titles", lambda scope, ids: ["BharatAgent Studio Build Handoff.pdf"])
    _models(monkeypatch)
    result = list(run_pipeline(Graph.model_validate(only_s1(full_graph)),
                               "What is the monthly parking fee at the company office?", source_map={"s1": "src208"}))[-1]
    assert searched == ["What is the monthly parking fee at the company office?"]   # not vague: searched as it is
    assert result["supported"] is False and result["answer"] in DONT_KNOW.values()
    searched.clear()
    result = list(run_pipeline(Graph.model_validate(only_s1(full_graph)), "What are its main features?",
                               source_map={"s1": "src208"}))[-1]
    assert len(searched) == 2 and result["supported"] is False       # the names added, still nothing relevant
    assert conversation.subject_query("q", ["a_b-c.pdf", "A B C.docx"]) == "q a b c"
    assert not conversation.needs_subject("How many days of paid leave do employees get each year?")
    assert conversation.needs_subject("Explain the architecture.") and conversation.needs_subject("What is this project about?")
