"""Retrieval benchmark: hybrid search vs hybrid + cross-encoder rerank on the same candidates.

    python -m evals.retrieval evals/retrieval_work.jsonl --collection work

Each line: {"q": question, "expect": [prefixes of relevant "<source>/<path>"]}.
A hit counts when a top-k result starts with any of them (several files can answer one question).
"""
from __future__ import annotations

import argparse
import json
import time

from core.config import RERANK_CANDIDATES
from rag.rerank import Reranker
from rag.search import Retriever

KS = (1, 3, 5)


def rank_of(hits, expect: list[str], depth: int = max(KS)) -> int | None:
    for i, h in enumerate(hits[:depth], 1):
        if h.display_ref().startswith(tuple(expect)):
            return i
    return None


def summary(ranks: list[int | None]) -> str:
    n = len(ranks)
    hits = "  ".join(f"hit@{k} {sum(1 for r in ranks if r and r <= k) / n:.2f}" for k in KS)
    mrr = sum(1 / r for r in ranks if r) / n
    return f"{hits}  MRR@{max(KS)} {mrr:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--collection", default="work")
    ap.add_argument("--candidates", type=int, default=RERANK_CANDIDATES)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    retriever = Retriever(args.collection, rerank=False)
    reranker = Reranker()
    base, reranked, recall, rerank_s = [], [], [], 0.0

    for c in cases:
        candidates = retriever.search(c["q"], k=args.candidates)
        t0 = time.time()
        rr = reranker.rerank(c["q"], list(candidates), k=max(KS))
        rerank_s += time.time() - t0
        b, r = rank_of(candidates, c["expect"]), rank_of(rr, c["expect"])
        recall.append(rank_of(candidates, c["expect"], depth=len(candidates)) is not None)
        base.append(b)
        reranked.append(r)
        mark = "+" if (r or 99) < (b or 99) else "-" if (r or 99) > (b or 99) else " "
        print(f"{mark} {str(b or '—'):>2} -> {str(r or '—'):>2}  {c['q']}")

    print(f"\n{len(cases)} queries, {args.candidates} candidates "
          f"(relevant among them: {sum(recall)}/{len(cases)}), reranker {reranker.name}")
    print(f"hybrid:   {summary(base)}")
    print(f"reranked: {summary(reranked)}")
    print(f"rerank latency: {rerank_s / len(cases) * 1000:.0f} ms/query")


if __name__ == "__main__":
    main()
