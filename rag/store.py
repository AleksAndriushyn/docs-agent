"""Qdrant collection with two named vectors per chunk: dense (semantic) and bm25 (keywords).
Search runs both and fuses the rankings with RRF."""
from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from core.config import QDRANT_PATH, QDRANT_URL

DENSE, SPARSE = "dense", "bm25"


@dataclass
class Hit:
    score: float  # RRF score from hybrid search
    payload: dict
    rerank_score: float | None = None  # cross-encoder relevance, when reranked

    def ref(self) -> str:
        p = self.payload
        return f"{p['path']}:{p['line_start']}-{p['line_end']}"

    def display_ref(self) -> str:
        """Ref prefixed with the source folder name: several folders can share relative paths."""
        return f"{Path(self.payload['source']).name}/{self.ref()}"


def point_id(source: str, path: str, idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{path}|{idx}"))


class Store:
    def __init__(self, collection: str, client: QdrantClient | None = None):
        self.collection = collection
        self.client = client or (QdrantClient(url=QDRANT_URL) if QDRANT_URL
                                 else QdrantClient(path=QDRANT_PATH))

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure(self, dim: int) -> None:
        if self.exists():
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with warnings.catch_warnings():  # embedded mode ignores payload indexes and warns
            warnings.simplefilter("ignore")
            for field in ("path", "source", "lang"):
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    def drop(self) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)

    def file_hashes(self, source: str) -> dict[str, str]:
        """path -> sha256 of every file already indexed from `source`."""
        hashes, offset = {}, None
        flt = _match(source=source)
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=1000, offset=offset,
                with_payload=["path", "file_sha256"], with_vectors=False)
            for p in points:
                hashes[p.payload["path"]] = p.payload["file_sha256"]
            if offset is None:
                return hashes

    def delete_file(self, source: str, path: str) -> None:
        self.client.delete(self.collection, models.FilterSelector(filter=_match(source=source, path=path)))

    def delete_source(self, source: str) -> None:
        if self.exists():
            self.client.delete(self.collection, models.FilterSelector(filter=_match(source=source)))

    def count(self, source: str | None = None) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, count_filter=_match(source=source), exact=True).count

    def upsert(self, payloads: list[dict], dense: list[list[float]],
               sparse: list[models.SparseVector]) -> None:
        points = [
            models.PointStruct(
                id=point_id(p["source"], p["path"], p["chunk_index"]),
                vector={DENSE: d, SPARSE: s},
                payload=p,
            )
            for p, d, s in zip(payloads, dense, sparse)
        ]
        self.client.upsert(self.collection, points)

    def search(self, dense: list[float], sparse: models.SparseVector, k: int = 6,
               lang: str | None = None, path_contains: str | None = None) -> list[Hit]:
        flt = _match(lang=lang)
        if path_contains:
            cond = models.FieldCondition(key="path", match=models.MatchText(text=path_contains))
            flt = models.Filter(must=[*(flt.must if flt else []), cond])
        prefetch_k = max(k * 4, 20)  # each retriever proposes more candidates than we return
        res = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=prefetch_k, filter=flt),
                models.Prefetch(query=sparse, using=SPARSE, limit=prefetch_k, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [Hit(p.score, p.payload) for p in res.points]

    def source_of(self, path: str) -> str | None:
        points, _ = self.client.scroll(self.collection, scroll_filter=_match(path=path), limit=1,
                                       with_payload=["source"], with_vectors=False)
        return points[0].payload["source"] if points else None


def _match(**fields) -> models.Filter | None:
    must = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in fields.items() if v is not None]
    return models.Filter(must=must) if must else None
