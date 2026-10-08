"""Registry of indexed folders, persisted in data/sources.json.

One entry per folder: the index is keyed by folder, and re-indexing a folder removes
files it no longer selects. Adding a single file therefore adds an include pattern
to its folder's entry instead of creating a new one.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.config import PROJECT_DIR

DATA_FILE = PROJECT_DIR / "data" / "sources.json"
ARGS_DIR = PROJECT_DIR / "sources"
DEFAULT_EXCLUDE = ["**/*.Designer.cs"]  # generated WinForms layout, pure noise for search


@dataclass
class Source:
    path: str
    include: list[str] = field(default_factory=list)  # empty = whole folder
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    status: str = "new"  # new | queued | indexing | ready | cancelled | error
    files: int = 0
    chunks: int = 0
    last_indexed: float | None = None
    error: str = ""

    @property
    def id(self) -> str:
        return hashlib.sha1(self.path.encode()).hexdigest()[:12]

    def to_json(self) -> dict:
        return {"id": self.id, "name": Path(self.path).name, **asdict(self)}


class Registry:
    def __init__(self, file: Path = DATA_FILE, collection: str = "work"):
        self.file, self.collection = file, collection
        self.lock = threading.RLock()
        self.items: dict[str, Source] = {}
        if file.exists():
            for d in json.loads(file.read_text()):
                src = Source(**d)
                self.items[src.id] = src
        else:
            self._import_args()
        for s in self.items.values():  # a crash mid-run leaves stale states
            if s.status in {"queued", "indexing"}:
                s.status = "new"

    def _import_args(self) -> None:
        """Seed from the CLI's sources/*.args so folders indexed there aren't redone."""
        for f in sorted(ARGS_DIR.glob("*.args")):
            opts: dict[str, list[str]] = {}
            for line in f.read_text().split():
                key, _, value = line.partition("=")
                opts.setdefault(key, []).append(value)
            if opts.get("--collection", [self.collection])[0] != self.collection or "--src" not in opts:
                continue
            path = str(Path(opts["--src"][0]).expanduser().resolve())
            src = Source(path, opts.get("--include", []), opts.get("--exclude", []), status="ready")
            self.items[src.id] = src
        self.save()

    def save(self) -> None:
        with self.lock:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            fields = ("path", "include", "exclude", "status", "files", "chunks", "last_indexed", "error")
            data = [{k: getattr(s, k) for k in fields} for s in self.items.values()]
            self.file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def list(self) -> list[Source]:
        with self.lock:
            return sorted(self.items.values(), key=lambda s: Path(s.path).name.lower())

    def get(self, sid: str) -> Source:
        with self.lock:
            if sid not in self.items:
                raise KeyError(sid)
            return self.items[sid]

    def add(self, raw_path: str) -> Source:
        p = Path(raw_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        folder, pattern = (p, None) if p.is_dir() else (p.parent, p.name)
        with self.lock:
            src = next((s for s in self.items.values() if s.path == str(folder)), None)
            if src is None:
                src = Source(str(folder), include=[pattern] if pattern else [])
                self.items[src.id] = src
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
            self.save()
            return src
