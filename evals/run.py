"""Answer-quality eval: run an agent over a QA dataset, grade every answer, save results.

    python -m evals.run evals/qa_docs.jsonl --collection docs --provider local
    python -m evals.run evals/qa_work.jsonl --collection work --source ~/Documents/MissionPlanner
    python -m evals.report evals/results/*.json            # compare runs

Per case: the agent answers from a fresh conversation; the judge grades facts, contradictions
and refusals; citations are checked deterministically against the expected source files.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from agents.researcher import researcher_agent
from core.console import Console
from core.llm import make_llm
from evals.exact import score
from evals.judge import Judge
from rag.store import Store

RESULTS_DIR = Path(__file__).parent / "results"
_REF = re.compile(r"^\[\d+\]\s+(\S+)", re.M)


class CountingConsole(Console):
    """Silent console that counts tool calls, including those of sub-agents."""

    def __init__(self, counter: dict | None = None):
        super().__init__(verbose=False, quiet=True)
        self.counter = counter if counter is not None else {"tools": 0}

    def tool_call(self, name: str, args, size: int) -> None:
        self.counter["tools"] += 1

    def child(self, name: str) -> Console:
        return CountingConsole(self.counter)


def cited_refs(answer: str) -> list[str]:
    """Refs from the "Sources:" footer the agent appends ("<source>/<path>:<a>-<b>")."""
    _, _, footer = answer.partition("\nSources:\n")
    return _REF.findall(footer)


def safe_grade(judge: Judge, case: dict, answer: str) -> dict:
    """A judge failure marks the case ungraded instead of killing the whole run."""
    try:
        return judge.grade(case, answer.partition("\nSources:\n")[0])
    except Exception as e:
        return {"facts": [], "contradictions": [], "refused": False, "verdict": "ungraded",
                "comment": f"judge error: {type(e).__name__}: {str(e)[:200]}"}


def fact_recall(case: dict, grade: dict) -> float | None:
    facts = grade["facts"]
    return sum(f["present"] for f in facts) / len(facts) if case["facts"] and facts else None


def rejudge(path: Path, kind: str, model: str | None = None, only_ungraded: bool = False) -> Path:
    """Grade saved answers again with another judge; the agent is not rerun."""
    run = json.loads(path.read_text())
    cases = {c["id"]: c for c in map(json.loads, open(run["dataset"], encoding="utf-8"))}
    judge = Judge(kind, model)
    for r in run["cases"]:
        if only_ungraded and r["grade"]["verdict"] != "ungraded":
            continue
        r["grade"] = safe_grade(judge, cases[r["id"]], r["answer"])
        r["fact_recall"] = fact_recall(cases[r["id"]], r["grade"])
        print(f"{r['id']:9} {r['grade']['verdict']:9} {r['grade']['comment'][:80]}")
    run["judge"], run["judge_usage"] = judge.name, judge.usage
    tag = judge.model.split(":")[0] if judge.kind == "local" else judge.kind
    out = path if only_ungraded else path.with_name(path.stem.split("@")[0] + f"@{tag}.json")
    out.write_text(json.dumps(run, ensure_ascii=False, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="QA .jsonl, or a results .json with --rejudge")
    ap.add_argument("--rejudge", action="store_true", help="Re-grade saved answers, don't rerun the agent")
    ap.add_argument("--only-ungraded", action="store_true", help="With --rejudge: retry failed grades in place")
    ap.add_argument("--collection")
    ap.add_argument("--provider", choices=["local", "anthropic"], default="local")
    ap.add_argument("--judge", choices=["local", "claude"], default="local")
    ap.add_argument("--judge-model", help="Local judge model (default: the agent's LOCAL_MODEL)")
    ap.add_argument("--source", action="append", help="Restrict retrieval to these indexed folders")
    ap.add_argument("--only", help="Comma-separated case ids")
    args = ap.parse_args()

    from evals.report import summarize, table
    if args.rejudge:
        out = rejudge(Path(args.dataset), args.judge, args.judge_model, args.only_ungraded)
        print(f"\nSaved {out}\n" + table([summarize(json.loads(out.read_text()))]))
        return
    if not args.collection:
        ap.error("--collection is required")

    cases = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    sources = [str(Path(s).expanduser().resolve()) for s in args.source or []] or None

    llm = make_llm(args.provider)
    # index only, no working folder: the same setup as earlier "rag" runs, so results compare
    agent = researcher_agent(None, args.collection, Store(args.collection), sources=sources)
    judge = Judge(args.judge, args.judge_model)
    print(f"{len(cases)} cases · agent researcher/{args.provider}:{llm.model} · judge {judge.name}")

    rows = []
    for c in cases:
        before, counter, t0 = llm.usage.copy(), {"tools": 0}, time.time()
        try:
            answer, error = agent.run(llm, [{"role": "user", "content": c["question"]}],
                                      CountingConsole(counter)), None
        except Exception as e:  # keep going: a crash is a failed case, not a failed run
            answer, error = "", f"{type(e).__name__}: {e}"
        seconds = time.time() - t0
        refs = cited_refs(answer)
        grade = safe_grade(judge, c, answer)
        rows.append({
            "id": c["id"], "answerable": c["answerable"], "question": c["question"],
            "answer": answer, "error": error, "seconds": round(seconds, 1),
            "tool_calls": counter["tools"], "usage": (llm.usage - before).snapshot(),
            "cited": refs,
            "citation_hit": any(r.startswith(tuple(c["sources"])) for r in refs) if c["sources"] else None,
            "fact_recall": fact_recall(c, grade),
            "exact": score(c["facts"], answer.partition("\nSources:\n")[0]),
            "grade": grade,
        })
        r = rows[-1]
        recall = f"{r['fact_recall']:.2f}" if r["fact_recall"] is not None else "  - "
        ex = r["exact"]
        terms = f"{ex['found']}/{ex['total']}" if ex["total"] else " - "
        print(f"{c['id']:9} {grade['verdict']:9} facts {recall}  terms {terms:5}  cite {str(r['citation_hit']):5}  "
              f"{seconds:5.0f}s  {grade['comment'][:70]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M')}-{Path(args.dataset).stem}-{args.provider}"
    out = RESULTS_DIR / f"{stem}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset, "collection": args.collection, "provider": args.provider,
        "model": llm.model, "judge": judge.name, "judge_usage": judge.usage,
        "sources": sources, "created": time.strftime("%Y-%m-%d %H:%M"), "cases": rows,
    }, ensure_ascii=False, indent=1))
    print(f"\nSaved {out}")
    print(table([summarize(json.loads(out.read_text()))]))


if __name__ == "__main__":
    main()
