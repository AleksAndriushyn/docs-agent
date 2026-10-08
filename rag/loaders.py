"""File -> Document. docx/xlsx are converted to markdown-like text so they share
the heading-based chunker; their line numbers refer to that converted text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LANG_BY_EXT = {
    ".py": "python",
    ".cs": "csharp",
    ".c": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
    ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".docx": "docx", ".xlsx": "xlsx",
}
SKIP_DIRS = {".git", ".svn", ".vs", ".idea", ".venv", "venv", "node_modules", "__pycache__",
             "bin", "obj", "build", "dist", "qdrant_data", "qdrant_storage", "sandbox"}
MAX_FILE_BYTES = 2_000_000
MAX_SHEET_ROWS = 5_000


@dataclass
class Document:
    path: str  # relative to the ingest source root
    lang: str
    text: str


def glob_regex(pattern: str) -> re.Pattern:
    """Path glob relative to the source root: `*` stays within one folder, `**` spans folders.
    `GCSViews/**` = everything under GCSViews, `*.cs` = .cs files in the root only."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z", re.IGNORECASE)  # macOS/Windows paths are case-insensitive


def iter_files(src: Path, include: list[str] | None = None,
               exclude: list[str] | None = None) -> Iterator[Path]:
    inc = [glob_regex(g) for g in include or []]
    exc = [glob_regex(g) for g in exclude or []]
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        posix = rel.as_posix()
        if inc and not any(r.match(posix) for r in inc) or any(r.match(posix) for r in exc):
            continue
        if (p.is_file() and p.suffix.lower() in LANG_BY_EXT and not p.name.startswith("~$")
                and p.stat().st_size <= MAX_FILE_BYTES):
            yield p


def load(path: Path, root: Path) -> Document | None:
    lang = LANG_BY_EXT[path.suffix.lower()]
    if lang == "docx":
        text = _docx_text(path)
    elif lang == "xlsx":
        text = _xlsx_text(path)
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp1251", errors="replace")  # legacy Windows files
    if not text.strip():
        return None
    return Document(path.relative_to(root).as_posix(), lang, text)


def _docx_text(path: Path) -> str:
    import docx

    d = docx.Document(str(path))
    lines = []
    for p in d.paragraphs:
        style = (p.style.name or "") if p.style is not None else ""
        if style.startswith("Heading") and style[-1:].isdigit():
            lines.append("#" * int(style[-1]) + " " + p.text)
        elif p.text.strip():
            lines.append(p.text)
    for i, table in enumerate(d.tables, 1):
        lines.append(f"## Table {i}")
        for row in table.rows:
            lines.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(lines)


def _xlsx_text(path: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    lines = []
    for ws in wb.worksheets:
        lines.append(f"## Sheet: {ws.title}")
        header = None
        for n, row in enumerate(ws.iter_rows(values_only=True)):
            if n >= MAX_SHEET_ROWS:
                lines.append(f"...[truncated at {MAX_SHEET_ROWS} rows]")
                break
            cells = ["" if v is None else str(v).strip() for v in row]
            if not any(cells):
                continue
            if header is None:
                header = cells
                continue
            # "column: value" keeps the header next to every value for retrieval
            lines.append("; ".join(f"{h or f'col{i + 1}'}: {v}"
                                   for i, (h, v) in enumerate(zip(header, cells)) if v))
    wb.close()
    return "\n".join(lines)
