from __future__ import annotations

from qdrant_client import models

from core.config import EMBED_MODEL, LOCAL_URL

# Ollama fails (400 "EOF") instead of truncating when one input exceeds the physical batch
# (n_ubatch = 2048 tokens). Chars don't map to tokens reliably (escapes/hex can be ~1 char
# per token), so this is a first cut and embed() shrinks any text that still fails.
MAX_EMBED_CHARS = 6000
MIN_EMBED_CHARS = 200

# qwen3-embedding is trained with an instruction on the query side only
QWEN_QUERY_PREFIX = "Instruct: Given a question, retrieve code or documentation passages that answer it\nQuery: "


class DenseEmbedder:
    """Dense vectors from Ollama via its OpenAI-compatible /v1/embeddings."""

    def __init__(self, model: str = EMBED_MODEL, url: str = LOCAL_URL, batch: int = 32):
        from openai import OpenAI
        self.client, self.model, self.batch = OpenAI(base_url=url, api_key="ollama"), model, batch
        self.shrunk = 0  # texts embedded from a shortened prefix

    def _call(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import BadRequestError

        out = []
        for i in range(0, len(texts), self.batch):
            batch = [t[:MAX_EMBED_CHARS] for t in texts[i:i + self.batch]]
            try:
                out.extend(self._call(batch))
            except BadRequestError:  # one oversized text fails the whole batch: isolate it
                out.extend(self._embed_one(t) for t in batch)
        return out

    def _embed_one(self, text: str) -> list[float]:
        from openai import BadRequestError

        shortened = False
        while True:
            try:
                vec = self._call([text])[0]
                self.shrunk += shortened
                return vec
            except BadRequestError:
                if len(text) <= MIN_EMBED_CHARS:
                    raise
                text, shortened = text[:len(text) // 2], True

    def embed_query(self, query: str) -> list[float]:
        prefix = QWEN_QUERY_PREFIX if self.model.startswith("qwen3-embedding") else ""
        return self.embed([prefix + query])[0]


class SparseEmbedder:
    """BM25 term weights; IDF is applied by Qdrant (Modifier.IDF on the collection)."""

    def __init__(self):
        from fastembed import SparseTextEmbedding
        self.model = SparseTextEmbedding("Qdrant/bm25")

    def embed(self, texts: list[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                for e in self.model.embed(texts)]

    def embed_query(self, query: str) -> models.SparseVector:
        e = next(iter(self.model.query_embed(query)))
        return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
