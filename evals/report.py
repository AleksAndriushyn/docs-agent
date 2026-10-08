"""Compare eval runs side by side:  python -m evals.report evals/results/*.json [--md out.md]"""
from __future__ import annotations

import argparse
import json
from statistics import mean, median

from evals.exact import score


def exact_stats(run: dict, cases: list[dict]) -> tuple[float | None, float | None]:
    """Hard-term recall, and how often the judge credited a fact whose terms are missing.
    Computed from saved answers, so older result files get it too."""
    refs = {c["id"]: c for c in map(json.loads, open(run["dataset"], encoding="utf-8"))}
    found = total = lenient = checkable = 0
    for c in cases:
        ex = c.get("exact") or score(refs[c["id"]]["facts"], c["answer"].partition("\nSources:\n")[0])
        found, total = found + ex["found"], total + ex["total"]
        judged = {f["fact"]: f["present"] for f in c["grade"]["facts"]}
        for f in ex["facts"]:
            if f["terms"]:
                checkable += 1
                lenient += bool(judged.get(f["fact"]) and f["missing"])
    return (found / total if total else None), (lenient / checkable if checkable else None)


def summarize(run: dict) -> dict:
    every = run["cases"]  # timing and spend count all cases; quality only graded ones
    cases = [c for c in every if c["grade"]["verdict"] != "ungraded"]
    ans = [c for c in cases if c["answerable"]]
    unans = [c for c in cases if not c["answerable"]]
    cited = [c for c in ans if c["citation_hit"] is not None]
    cost = sum(c["usage"]["cost_usd"] or 0 for c in every)
    term_recall, lenient = exact_stats(run, ans)
    return {
        "term_recall": term_recall,
        "judge_lenient": lenient,
        "run": f"{run['dataset'].split('/')[-1]} · {run['provider']}:{run['model']} · judge {run['judge'].split(':')[1]}",
        "judge": run["judge"],
        "n": len(cases),
        "ungraded": len(every) - len(cases),
        "correct": sum(c["grade"]["verdict"] == "correct" for c in ans) / len(ans) if ans else None,
        "fact_recall": mean(c["fact_recall"] for c in ans if c["fact_recall"] is not None) if ans else None,
        "contradictions": sum(bool(c["grade"]["contradictions"]) for c in ans) / len(ans) if ans else None,
        "refusal_ok": sum(c["grade"]["verdict"] == "correct" for c in unans) / len(unans) if unans else None,
        "citation_hit": sum(bool(c["citation_hit"]) for c in cited) / len(cited) if cited else None,
        "median_s": median(c["seconds"] for c in every),
        "tokens": sum(c["usage"]["input"] + c["usage"]["output"] for c in every),
        "cost": cost if any(c["usage"]["cost_usd"] is not None for c in every) else None,
        "errors": sum(bool(c["error"]) for c in every),
    }


COLUMNS = [  # key, header, format
    ("run", "run", "{}"), ("correct", "correct", "{:.0%}"), ("fact_recall", "fact recall", "{:.0%}"),
    ("term_recall", "terms found", "{:.0%}"), ("judge_lenient", "judge lenient", "{:.0%}"),
    ("contradictions", "contradicts", "{:.0%}"), ("refusal_ok", "refuses OK", "{:.0%}"),
    ("citation_hit", "cites source", "{:.0%}"), ("median_s", "median s", "{:.0f}"),
    ("tokens", "tokens", "{:,}"), ("cost", "cost $", "{:.3f}"), ("errors", "errors", "{}"),
    ("ungraded", "ungraded", "{}"),
]


def table(summaries: list[dict]) -> str:
    lines = ["| " + " | ".join(h for _, h, _ in COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    for s in summaries:
        cells = [("—" if s[k] is None else fmt.format(s[k])) for k, _, fmt in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    judges = sorted({s["judge"] for s in summaries})
    return "\n".join(lines) + f"\n\njudge: {', '.join(judges)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--md", help="Also write the table to this markdown file")
    args = ap.parse_args()
    out = table([summarize(json.load(open(f, encoding="utf-8"))) for f in args.results])
    print(out)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(out + "\n")


if __name__ == "__main__":
    main()
