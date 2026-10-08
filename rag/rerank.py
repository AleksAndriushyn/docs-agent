"""Second-stage ranking with a cross-encoder.

Hybrid search scores query and chunk separately (bi-encoder + BM25), so it is fast but
coarse. A cross-encoder reads query and chunk together and scores their relevance,
which is much more precise but too slow for the whole index: it only reorders the
top candidates from hybrid search.
"""
from __future__ import annotations

from core.config import RERANK_MODEL
from rag.store import Hit

MAX_DOC_CHARS = 1200  # head of a chunk is enough: on the work benchmark 1200 was as good as 2500, 1.5x faster


def rerank_text(payload: dict) -> str:
    """Path and symbol help the model: 'PluginLoader.cs / LoadAll' is strong evidence."""
    head = payload["path"] + (f" ({payload['symbol']})" if payload.get("symbol") else "")
    return f"{head}\n{payload['text'][:MAX_DOC_CHARS]}"


class Reranker:
    def __init__(self, model: str = RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self.model = TextCrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.rerank(query, [rerank_text(h.payload) for h in hits], batch_size=8)
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:k]
