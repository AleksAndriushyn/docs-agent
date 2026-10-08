"""Registry of collections and the folders indexed into them, persisted in data/sources.json.

One entry per (collection, folder): the index is keyed by folder, and re-indexing a folder
removes files it no longer selects. Adding a single file therefore adds an include pattern
to its folder's entry instead of creating a new one.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.config import PROJECT_DIR

DATA_FILE = PROJECT_DIR / "data" / "sources.json"
ARGS_DIR = PROJECT_DIR / "sources"
DEFAULT_EXCLUDE = ["**/*.Designer.cs"]  # generated WinForms layout, pure noise for search
COLLECTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class Source:
    path: str
    collection: str = "work"
    include: list[str] = field(default_factory=list)  # empty = whole folder
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    status: str = "new"  # new | queued | indexing | ready | cancelled | error
    files: int = 0
    chunks: int = 0
    last_indexed: float | None = None
    error: str = ""

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.collection}|{self.path}".encode()).hexdigest()[:12]

    def to_json(self) -> dict:
        return {"id": self.id, "name": Path(self.path).name, **asdict(self)}


class Registry:
    def __init__(self, file: Path = DATA_FILE, default_collection: str = "work"):
        self.file, self.default = file, default_collection
        self.lock = threading.RLock()
        self.items: dict[str, Source] = {}
        self.empty_collections: set[str] = set()  # created in the UI, nothing indexed yet
        if file.exists():
            data = json.loads(file.read_text())
            if isinstance(data, list):  # format before collections existed
                data = {"sources": data}
            self.empty_collections = set(data.get("collections", []))
            for d in data.get("sources", []):
                d.setdefault("collection", self.default)
                self._put(Source(**d))
        else:
            self._import_args()
        for s in self.items.values():  # a crash mid-run leaves stale states
            if s.status in {"queued", "indexing"}:
                s.status = "new"

    def _put(self, src: Source) -> Source:
        self.items[src.id] = src
        return src

    def _import_args(self) -> None:
        """Seed from the CLI's sources/*.args so folders indexed there aren't redone."""
        for f in sorted(ARGS_DIR.glob("*.args")):
            opts: dict[str, list[str]] = {}
            for line in f.read_text().split():
                key, _, value = line.partition("=")
                opts.setdefault(key, []).append(value)
            if "--src" not in opts:
                continue
            path = str(Path(opts["--src"][0]).expanduser().resolve())
            collection = opts.get("--collection", [self.default])[0]
            self._put(Source(path, collection, opts.get("--include", []), opts.get("--exclude", []),
                             status="ready"))
        self.save()

    def save(self) -> None:
        with self.lock:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            fields = ("path", "collection", "include", "exclude", "status", "files", "chunks",
                      "last_indexed", "error")
            data = {
                "collections": sorted(self.empty_collections - {s.collection for s in self.items.values()}),
                "sources": [{k: getattr(s, k) for k in fields} for s in self.items.values()],
            }
            self.file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # ---------- collections

    def collections(self) -> list[str]:
        with self.lock:
            return sorted({s.collection for s in self.items.values()} | self.empty_collections | {self.default})

    def add_collection(self, name: str) -> None:
        if not COLLECTION_NAME.match(name):
            raise ValueError("Use 1-64 letters, digits, '-' or '_'")
        with self.lock:
            self.empty_collections.add(name)
            self.save()

    def remove_collection(self, name: str) -> list[Source]:
        with self.lock:
            removed = [s for s in self.items.values() if s.collection == name]
            for src in removed:
                del self.items[src.id]
            self.empty_collections.discard(name)
            self.save()
            return removed

    def prune_phantoms(self, collection: str, indexed: dict[str, int]) -> list[Source]:
        """Drop entries that only came from Qdrant discovery and have no points left
        (removed earlier while a stale index value brought them back)."""
        with self.lock:
            gone = [s for s in self.items.values()
                    if s.collection == collection and s.status == "ready"
                    and s.last_indexed is None and not indexed.get(s.path)]
            for src in gone:
                del self.items[src.id]
            if gone:
                self.save()
            return gone

    def discover(self, collection: str, path: str) -> Source | None:
        """Register a folder found in Qdrant (e.g. indexed with the CLI) if it isn't known yet."""
        with self.lock:
            if any(s.collection == collection and s.path == path for s in self.items.values()):
                return None
            src = self._put(Source(path, collection, status="ready"))
            self.save()
            return src

    # ---------- sources

    def list(self, collection: str | None = None) -> list[Source]:
        with self.lock:
            items = [s for s in self.items.values() if collection is None or s.collection == collection]
            return sorted(items, key=lambda s: Path(s.path).name.lower())

    def get(self, sid: str) -> Source:
        with self.lock:
            if sid not in self.items:
                raise KeyError(sid)
            return self.items[sid]

    def add(self, raw_path: str, collection: str | None = None) -> Source:
        collection = collection or self.default
        p = Path(raw_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        folder, pattern = (p, None) if p.is_dir() else (p.parent, p.name)
        with self.lock:
            src = next((s for s in self.items.values()
                        if s.collection == collection and s.path == str(folder)), None)
            if src is None:
                src = self._put(Source(str(folder), collection, include=[pattern] if pattern else []))
            elif pattern is None:
                src.include = []  # the whole folder now
            elif src.include and pattern not in src.include:
                src.include.append(pattern)
            self.save()
            return src

    def update(self, sid: str, include: list[str], exclude: list[str]) -> Source:
        with self.lock:
            src = self.get(sid)
            src.include = [g.strip() for g in include if g.strip()]
            src.exclude = [g.strip() for g in exclude if g.strip()]
            self.save()
            return src

    def remove(self, sid: str) -> Source:
        with self.lock:
            src = self.items.pop(sid)
            self.empty_collections.add(src.collection)  # keep the collection visible
            self.save()
            return src
