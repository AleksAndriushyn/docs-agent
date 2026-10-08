from __future__ import annotations

import re
from pathlib import Path

from core.config import MAX_FILE_CHARS, MAX_HITS


class Workspace:
    """File access confined to root: the model can't escape the folder."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SystemExit(f"Folder not found: {self.root}")

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"path outside the working folder: {rel}")
        return p

    def list_dir(self, path: str = ".") -> str:
        p = self.resolve(path)
        items = []
        for c in sorted(p.iterdir()):
            if c.name.startswith("."):
                continue
            items.append(f"{c.name}/" if c.is_dir() else f"{c.name} ({c.stat().st_size} B)")
        return "\n".join(items) or "(empty)"

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        lines = self.resolve(path).read_text(encoding="utf-8", errors="replace").splitlines()
        last = min(end_line or len(lines), len(lines))
        out, size = [], 0
        for i in range(max(start_line, 1), last + 1):
            line = f"{i}: {lines[i - 1]}"
            size += len(line) + 1
            if size > MAX_FILE_CHARS:
                out.append(f"...[truncated, {len(lines)} lines total; continue with start_line={i}]")
                break
            out.append(line)
        return "\n".join(out) or "(empty file)"

    def search(self, pattern: str, glob: str = "**/*") -> str:
        rx = re.compile(pattern, re.IGNORECASE)
        hits = []
        for f in sorted(self.root.glob(glob)):
            if not f.is_file() or f.stat().st_size > 2_000_000:
                continue
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable
            for n, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(self.root)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= MAX_HITS:
                        return "\n".join(hits) + f"\n...[limit {MAX_HITS} reached, narrow the pattern]"
        return "\n".join(hits) or "no matches"
