import pytest

from app.sources import Registry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr("app.sources.ARGS_DIR", tmp_path / "no-args")
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "a.cs").write_text("class A {}")
    (tmp_path / "proj" / "b.cs").write_text("class B {}")
    return Registry(tmp_path / "data" / "sources.json"), tmp_path / "proj"


def test_files_merge_into_one_folder_entry(reg):
    r, proj = reg
    r.add(str(proj / "a.cs"))
    src = r.add(str(proj / "b.cs"))
    assert len(r.list()) == 1
    assert src.path == str(proj.resolve()) and src.include == ["a.cs", "b.cs"]


def test_adding_folder_widens_file_entry(reg):
    r, proj = reg
    r.add(str(proj / "a.cs"))
    assert r.add(str(proj)).include == []


def test_persisted_and_reloaded(reg, tmp_path):
    r, proj = reg
    sid = r.add(str(proj)).id
    again = Registry(tmp_path / "data" / "sources.json")
    assert again.get(sid).path == str(proj.resolve())


def test_missing_path(reg):
    r, proj = reg
    with pytest.raises(FileNotFoundError):
        r.add(str(proj / "nope"))


def test_ingest_stop_keeps_unvisited_files(tmp_path):
    """A cancelled run must not treat unvisited files as deleted."""
    from qdrant_client import QdrantClient

    from rag.ingest import ingest
    from rag.store import Store

    proj = tmp_path / "p"
    proj.mkdir()
    for name in "abc":
        (proj / f"{name}.txt").write_text(f"file {name}")

    class Dense:
        shrunk = 0
        def embed(self, texts): return [[1.0, 0.0] for _ in texts]

    class Sparse:
        def embed(self, texts):
            from qdrant_client import models
            return [models.SparseVector(indices=[1], values=[1.0]) for _ in texts]

    store = Store("t", client=QdrantClient(location=":memory:"))
    assert ingest(proj, store, Dense(), Sparse()).changed == 3
    (proj / "a.txt").write_text("changed")
    calls = iter([False, True])  # stop before the second file
    st = ingest(proj, store, Dense(), Sparse(), should_stop=lambda: next(calls))
    assert st.cancelled and st.removed == 0
    assert set(store.file_hashes(str(proj))) == {"a.txt", "b.txt", "c.txt"}


def test_indexer_cancel_running_and_queued(tmp_path, monkeypatch):
    import threading
    import time

    from qdrant_client import QdrantClient, models

    from app.indexer import Indexer
    from rag.store import Store

    monkeypatch.setattr("app.sources.ARGS_DIR", tmp_path / "no-args")
    slow, fast = tmp_path / "slow", tmp_path / "fast"
    for d in (slow, fast):
        d.mkdir()
        for i in range(50):
            (d / f"f{i}.txt").write_text(f"text {i}")

    started = threading.Event()

    class Dense:
        shrunk = 0
        def embed(self, texts):
            started.set()
            time.sleep(0.02)
            return [[1.0, 0.0] for _ in texts]

    class Sparse:
        def embed(self, texts):
            return [models.SparseVector(indices=[1], values=[1.0]) for _ in texts]

    reg = Registry(tmp_path / "sources.json")
    idx = Indexer(reg, Store("t", client=QdrantClient(location=":memory:")))
    idx._dense, idx._sparse = Dense(), Sparse()
    a, b = reg.add(str(slow)), reg.add(str(fast))
    idx.enqueue(a.id)
    idx.enqueue(b.id)
    assert started.wait(5)
    idx.cancel(b.id)  # still queued
    idx.cancel(a.id)  # running
    for _ in range(200):
        if a.status == "cancelled":
            break
        time.sleep(0.02)
    assert a.status == "cancelled" and "Stopped at" in a.error
    assert 0 < a.chunks < 50
    assert b.status == "cancelled" and b.chunks == 0
