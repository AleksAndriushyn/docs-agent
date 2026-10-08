"""Retriever debugging without an LLM:  python -m rag.search "throttle failsafe" -k 5"""
from __future__ import annotations

import argparse

from core.config import RAG_COLLECTION, RERANK_CANDIDATES, RERANK_MODEL
from rag.embed import DenseEmbedder, SparseEmbedder
from rag.rerank import Reranker
from rag.store import Hit, Store


class Retriever:
    def __init__(self, collection: str = RAG_COLLECTION, store: Store | None = None,
                 rerank: bool = bool(RERANK_MODEL), sources: list[str] | None = None):
        self.store = store or Store(collection)
        self.sources = sources
        if not self.store.exists():
            raise SystemExit(f"Collection '{collection}' not found. Run first: python -m rag.ingest --src <folder>")
        self.dense, self.sparse = DenseEmbedder(), SparseEmbedder()
        self.rerank = rerank
        self._reranker: Reranker | None = None

    def search(self, query: str, k: int = 6, lang: str | None = None,
               path_contains: str | None = None) -> list[Hit]:
        # with reranking, hybrid search only proposes candidates and the cross-encoder picks k
        n = max(k, RERANK_CANDIDATES) if self.rerank else k
        hits = self.store.search(self.dense.embed_query(query), self.sparse.embed_query(query),
                                 k=n, lang=lang, path_contains=path_contains, sources=self.sources)
        if not self.rerank:
            return hits
        if self._reranker is None:  # ~1 GB model, loaded on first use
            self._reranker = Reranker()
        return self._reranker.rerank(query, hits, k)


def format_hits(hits: list[Hit], max_chars: int = 2500, ids: list[int] | None = None) -> str:
    if not hits:
        return "nothing found"
    parts = []
    for i, h in enumerate(hits):
        p = h.payload
        title = (f"[{ids[i]}] " if ids else "") + h.ref()
        title += f" ({p['symbol']})" if p.get("symbol") else ""
        if p.get("context"):
            title += f" in {p['context']}"
        text = p["text"] if len(p["text"]) <= max_chars else p["text"][:max_chars] + "\n...[truncated]"
        parts.append(f"### {title}\n```{p['lang']}\n{text}\n```")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--collection", default=RAG_COLLECTION)
    ap.add_argument("--lang")
    ap.add_argument("--full", action="store_true", help="Print chunk text")
    ap.add_argument("--no-rerank", action="store_true", help="Hybrid search only")
    args = ap.parse_args()

    hits = Retriever(args.collection, rerank=not args.no_rerank).search(args.query, args.k, args.lang)
    if args.full:
        print(format_hits(hits))
        return
    for h in hits:
        rr = f"{h.rerank_score:6.2f}" if h.rerank_score is not None else "     -"
        print(f"rrf {h.score:.3f}  rerank {rr}  {h.display_ref()}  {h.payload.get('symbol', '')}")


if __name__ == "__main__":
    main()
