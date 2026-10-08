from qdrant_client import QdrantClient, models

from rag.store import Store


def payload(path, idx, text, sha="s1"):
    return {"source": "/src", "path": path, "lang": "text", "file_sha256": sha,
            "chunk_index": idx, "line_start": 1, "line_end": 2, "symbol": "", "context": "",
            "text": text}


def sv(*idx):
    return models.SparseVector(indices=list(idx), values=[1.0] * len(idx))


def make_store():
    store = Store("t", client=QdrantClient(location=":memory:"))
    store.ensure(3)
    store.upsert(
        [payload("a.txt", 0, "alpha"), payload("b.txt", 0, "beta"), payload("b.txt", 1, "beta2")],
        [[1, 0, 0], [0, 1, 0], [0, 1, 0.1]],
        [sv(1), sv(2), sv(2, 3)],
    )
    return store


def test_hybrid_search_ranks_matching_point_first():
    hits = make_store().search([1, 0, 0], sv(1), k=2)
    assert hits[0].payload["path"] == "a.txt"
    assert hits[0].ref() == "a.txt:1-2"


def test_file_hashes_and_delete():
    store = make_store()
    assert store.file_hashes("/src") == {"a.txt": "s1", "b.txt": "s1"}
    store.delete_file("/src", "b.txt")
    assert store.file_hashes("/src") == {"a.txt": "s1"}
    assert store.source_of("a.txt") == "/src"


def test_lang_filter():
    assert make_store().search([1, 0, 0], sv(1), k=5, lang="python") == []


def test_sources_filter():
    store = make_store()
    assert store.search([1, 0, 0], sv(1), k=5, sources=["/other"]) == []
    assert len(store.search([1, 0, 0], sv(1), k=5, sources=["/src"])) == 3
