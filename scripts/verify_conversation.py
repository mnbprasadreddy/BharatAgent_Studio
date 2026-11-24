"""Conversational RAG check with real models: one multi-turn conversation over a project's knowledge, asked
through project chat or a shared agent, recording for every turn the search queries the agent used, the
retrieved chunks and scores, the answer, its citations, the answer language, the model of every step and the
time taken.

The project holds the given main PDF (for example the BharatAgent Studio Build Handoff) plus
samples/ecommerce-architecture.pdf, so turn 13 can change topic to a different source. A fresh test account is
registered for every run (random password, never printed).

  python scripts/verify_conversation.py --pdf "<path to the Build Handoff PDF>" --model local --mode project
  python scripts/verify_conversation.py --pdf ... --model groq --mode agent --pause 20 --out docs/results/conv.json
  python scripts/verify_conversation.py --pdf ... --model local --turns 1,3,10   # targeted re-check, in order

Each turn is judged with simple, stated expectations (see TURNS): an answer that must be supported and cited,
in the right language, mentioning the expected terms; or an honest abstention. A turn where any step (the
rewrite or the answer) was done by another model than the one asked for is a FALLBACK, never a pass for the
requested model. Exit code 1 if any turn fails.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _session import add_file, sign_up  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TE, HI = re.compile(r"[ఀ-౿]"), re.compile(r"[ऀ-ॿ]")

# question, expectation: ("answer", language, regex the answer must match) or ("abstain",)
TURNS = [
    ("What is this project about?", ("answer", "en", r"(?i)bharatagent|agent|pipeline|assistant")),
    ("Explain it in detail.", ("answer", "en", r"(?i)pipeline|agent|source|model|document")),
    ("tell clerly about this allobrate the explination", ("answer", "en", r"(?i)pipeline|agent|source|model|document")),
    ("What are its main features?", ("answer", "en", r"(?i)cit|model|cost|telugu|hindi|pipeline|source")),
    ("How does it work?", ("answer", "en", r"(?i)pipeline|guardrail|router|agent|retriev|chunk|source")),
    ("Explain its architecture.", ("answer", "en", r"(?i)docker|fastapi|postgres|react|ollama|container|api|web")),
    ("Explain the second feature in simple words.", ("answer", "en", r".")),
    ("Give me an example.", ("answer", "en", r".")),
    ("Summarize everything in five points.", ("answer", "en", r".")),
    ("Explain the same thing in Telugu.", ("answer", "te", r".")),
    ("Explain it in Hindi.", ("answer", "hi", r".")),
    ("Continue the explanation.", ("answer", "en", r".")),
    ("Above what order value is delivery free on the e-commerce platform?", ("answer", "en", r"499")),
    ("What is the monthly parking fee at the company office?", ("abstain",)),
]


def language_of(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "?"
    te, hi = len(TE.findall(text)), len(HI.findall(text))
    if max(te, hi) / len(letters) >= 0.4:
        return "te" if te >= hi else "hi"
    return "en"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="the main document (e.g. the Build Handoff PDF)")
    ap.add_argument("--model", default="local", choices=["local", "local_small", "groq", "gemini"])
    ap.add_argument("--mode", default="project", choices=["project", "agent"])
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", help="JSON file for the full record")
    ap.add_argument("--only-main", action="store_true",
                    help="a project with only the main PDF (no second document, so 'it' cannot mean another one)")
    ap.add_argument("--turns", help="run only these turns, in order, e.g. 1,3,9,10 (a targeted re-check)")
    ap.add_argument("--pause", type=float, default=0, help="seconds between turns (Groq's free tier limits tokens per minute)")
    args = ap.parse_args()

    acct = sign_up(args.api, f"conv-{args.model}-{args.mode}")
    c = acct.client
    main_id = add_file(acct, Path(args.pdf))
    other_id = None if args.only_main else add_file(acct, ROOT / "samples" / "ecommerce-architecture.pdf")
    graph_id = None
    if args.mode == "agent":
        main_nodes = ["input", "guardrails", "router", "agent", "output"]
        nodes = [{"id": f"n{i + 1}", "type": t, "config": {"model": args.model, "language": "en"} if t == "agent" else {}}
                 for i, t in enumerate(main_nodes)]
        docs = {"doc_main": main_id, **({"doc_shop": other_id} if other_id else {})}
        nodes += [{"id": n, "type": "source_doc", "config": {"source_ref": sid}} for n, sid in docs.items()]
        edges = [{"from": f"n{i}", "to": f"n{i + 1}"} for i in range(1, 5)] + [{"from": n, "to": "n4"} for n in docs]
        r = c.post("/graphs", json={"project_id": acct.project_id, "name": "Conversation check agent",
                                    "graph": {"nodes": nodes, "edges": edges}})
        r.raise_for_status()
        graph_id = r.json()["id"]
    conv = c.post("/conversations", json={"project_id": acct.project_id, "graph_id": graph_id,
                                          "title": "conversation check"}).json()["id"]

    rows, failed = [], 0
    chosen = {int(x) for x in args.turns.split(",")} if args.turns else set(range(1, len(TURNS) + 1))
    for i, (question, expect) in enumerate(TURNS, start=1):
        if i not in chosen:
            continue
        started = time.time()
        r = c.post("/pipeline/run?stream=false", json={"question": question, "conversation_id": conv, "model": args.model})
        took = round(time.time() - started, 1)
        events = r.json()["events"] if r.status_code == 200 else []
        last = events[-1] if events else {"type": "error", "message": r.text[:200]}
        agent = next((e.get("detail") or {} for e in events if e.get("stage") == "agent" and e.get("status") == "done"), {})
        ret = agent.get("retrieval") or {}
        answer = last.get("answer") or last.get("message") or ""
        cites = [f"{x.get('file')} p{x.get('page')} c{x.get('chunk')}" for x in (last.get("citations") or [])]
        lang = language_of(answer)
        steps = (agent.get("generation") or {}).get("steps", [])
        # An answer (or rewrite) by another model than the one asked for is a fallback, not this model's result.
        fell_back = any(not str(st.get("model", "")).startswith(f"{args.model}:") for st in steps)
        if expect[0] == "abstain":
            ok = last.get("type") == "result" and not last.get("supported") and not fell_back
        else:
            ok = (last.get("type") == "result" and bool(last.get("supported")) and bool(cites) and not fell_back
                  and lang == expect[1] and re.search(expect[2], answer) is not None)
        failed += not ok
        row = {"turn": i, "question": question, "ok": ok, "expected": expect, "seconds": took,
               "search_queries": ret.get("queries") or [question], "rewrite": ret.get("rewrite"),
               "reused_evidence": ret.get("reused"), "candidates": ret.get("candidates"), "relevant": ret.get("relevant"),
               "answer": answer, "citations": cites, "language": lang, "answered_by": last.get("model"),
               "steps": [(s.get("step"), s.get("model"), s.get("ms")) for s in steps],
               "reason": agent.get("reason"), "warnings": last.get("warnings"), "fallback": fell_back}
        rows.append(row)
        flag = "PASS" if ok else ("FALLBACK" if fell_back else "FAIL")
        print(f"\n[{flag}] {i:2}. {question}  ({took}s, {last.get('model')}, lang {lang}, steps {row['steps']})", flush=True)
        print(f"     search: {row['search_queries']}", flush=True)
        top = [(x.get("file", "")[:18], x.get("chunk"), x.get("similarity")) for x in (ret.get("candidates") or [])[:4]]
        print(f"     top chunks: {top}  relevant={ret.get('relevant')}", flush=True)
        print(f"     answer: {answer[:260]}", flush=True)
        print(f"     cites: {cites}", flush=True)
        if not ok and agent.get("reason"):
            print(f"     reason: {agent.get('reason')}", flush=True)
        if args.pause and i < max(chosen):
            time.sleep(args.pause)
    print(f"\n{len(rows) - failed} of {len(rows)} turns passed ({args.model}, {args.mode} chat)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"model": args.model, "mode": args.mode, "run": time.strftime("%Y-%m-%d %H:%M:%S"),
                                              "turns": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
