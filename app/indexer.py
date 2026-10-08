"""Background indexing: one worker thread, one job at a time (Ollama embeds serially anyway)."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable

from app.sources import Registry
from rag.embed import DenseEmbedder, SparseEmbedder
from rag.ingest import ingest
from rag.loaders import iter_files
from rag.store import Store


class Indexer:
    def __init__(self, registry: Registry, store_for: Callable[[str], Store]):
        self.registry, self.store_for = registry, store_for
        self.jobs: queue.Queue[str] = queue.Queue()
        self.progress: dict[str, dict] = {}  # source id -> {done, total, current}
        self.stop_requested: set[str] = set()
        self._dense = self._sparse = None
        threading.Thread(target=self._worker, daemon=True, name="indexer").start()
        threading.Thread(target=self._fill_stats, daemon=True, name="stats").start()

    def _fill_stats(self) -> None:
        """Sources imported from CLI .args files come without counts."""
        for src in self.registry.list():
            if src.status == "ready" and not src.files:
                if Path(src.path).is_dir():
                    src.files = sum(1 for _ in iter_files(Path(src.path), src.include, src.exclude))
                src.chunks = self.store_for(src.collection).count(src.path)
        self.registry.save()

    def enqueue(self, sid: str) -> None:
        src = self.registry.get(sid)
        if src.status in {"queued", "indexing"}:
            return
        src.status, src.error = "queued", ""
        self.jobs.put(sid)

    def cancel(self, sid: str) -> None:
        src = self.registry.get(sid)
        if src.status == "queued":  # the worker skips it when dequeued
            src.status, src.error = "cancelled", "Cancelled before start"
            self.registry.save()
        elif src.status == "indexing":
            self.stop_requested.add(sid)  # checked between files

    def busy(self, sid: str) -> bool:
        return self.registry.get(sid).status in {"queued", "indexing"}

    def _worker(self) -> None:
        while True:
            sid = self.jobs.get()
            try:
                src = self.registry.get(sid)
            except KeyError:
                continue  # removed while queued
            if src.status == "queued":  # not cancelled meanwhile
                self._run(src)

    def _run(self, src) -> None:
        sid = src.id
        src.status = "indexing"
        self.progress[sid] = {"done": 0, "total": 0, "current": ""}
        try:
            if self._dense is None:  # fastembed model load is slow, do it once
                self._dense, self._sparse = DenseEmbedder(), SparseEmbedder()

            def on_file(done: int, total: int, rel: str, status: str) -> None:
                self.progress[sid] = {"done": done, "total": total, "current": rel}

            store = self.store_for(src.collection)
            st = ingest(Path(src.path), store, self._dense, self._sparse,
                        src.include, src.exclude, on_file=on_file,
                        should_stop=lambda: sid in self.stop_requested)
            src.files = st.total
            src.chunks = store.count(src.path)
            src.last_indexed = time.time()
            if st.cancelled:
                done = self.progress.get(sid, {}).get("done", 0)
                src.status = "cancelled"
                src.error = f"Stopped at {done}/{st.total} files; indexed ones stay searchable"
            else:
                src.status = "ready"
                src.error = f"{st.errors} files failed to load" if st.errors else ""
        except Exception as e:
            src.status, src.error = "error", str(e)[:500]
        finally:
            self.stop_requested.discard(sid)
            self.progress.pop(sid, None)
            self.registry.save()
