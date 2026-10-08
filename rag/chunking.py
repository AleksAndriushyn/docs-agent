"""Document -> chunks with 1-based line ranges.

- markdown/docx/xlsx: split by '#' headings
- rst: split by underlined titles
- code: tree-sitter; definitions that fit stay whole, big ones are split into children
- anything else / oversized pieces: sliding line window
Small neighbouring pieces are merged up to TARGET_CHARS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.loaders import Document

TARGET_CHARS = 1500
WINDOW_LINES, WINDOW_OVERLAP = 60, 10

CODE_LANGS = {"python", "csharp", "cpp"}
# Node types that get a symbol name and act as context for nested chunks.
DEF_TYPES = {
    "function_definition", "class_definition", "decorated_definition",  # python, cpp
    "class_declaration", "struct_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "method_declaration", "constructor_declaration",
    "property_declaration", "namespace_declaration",  # c#
    "class_specifier", "struct_specifier", "namespace_definition", "template_declaration",  # c++
}

_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+:.])\1{2,}\s*$")
_MD_HEADING = re.compile(r"^#{1,6}\s+\S")


@dataclass
class Chunk:
    text: str
    line_start: int
    line_end: int
    symbol: str = ""   # function/class name or section title
    context: str = ""  # enclosing class/namespace for code


@dataclass
class _Seg:
    start: int  # 0-based, inclusive
    end: int
    symbol: str = ""
    context: str = ""


def chunk(doc: Document) -> list[Chunk]:
    lines = doc.text.splitlines()
    if not lines:
        return []
    if doc.lang in CODE_LANGS:
        segs = _code_segments(doc.text, doc.lang, lines)
    elif doc.lang == "rst":
        segs = _heading_segments(lines, _rst_titles(lines))
    elif doc.lang in {"markdown", "docx", "xlsx"}:
        segs = _heading_segments(
            lines, {i: lines[i].lstrip("#").strip() for i, l in enumerate(lines) if _MD_HEADING.match(l)})
    else:
        segs = [_Seg(0, len(lines) - 1)]
    segs = [s for seg in _normalize(segs, len(lines)) for s in _split_big(seg, lines)]
    return [_to_chunk(group, lines) for group in _merge(segs, lines)]


# ---------- prose ----------

def _rst_titles(lines: list[str]) -> dict[int, str]:
    titles = {}
    for i in range(len(lines) - 1):
        title, under = lines[i].strip(), lines[i + 1].strip()
        if title and not _RST_UNDERLINE.match(lines[i]) and _RST_UNDERLINE.match(lines[i + 1]) \
                and len(under) >= len(title):
            start = i - 1 if i > 0 and _RST_UNDERLINE.match(lines[i - 1]) else i  # overline style
            titles[start] = title
    return titles


def _heading_segments(lines: list[str], titles: dict[int, str]) -> list[_Seg]:
    starts = sorted(set(titles) | {0})
    bounds = starts[1:] + [len(lines)]
    return [_Seg(s, e - 1, titles.get(s, "")) for s, e in zip(starts, bounds)]


# ---------- code ----------

def _code_segments(text: str, lang: str, lines: list[str]) -> list[_Seg]:
    try:
        from tree_sitter_language_pack import get_parser
        root = get_parser(lang).parse(text.encode("utf-8")).root_node
    except Exception:
        return [_Seg(0, len(lines) - 1)]  # no grammar: fall back to windows

    segs: list[_Seg] = []

    def size(node) -> int:
        return node.end_byte - node.start_byte

    def visit(node, context: str) -> None:
        # a whole definition is worth more than its pieces, so it gets double budget
        limit = TARGET_CHARS * 2 if node.type in DEF_TYPES else TARGET_CHARS
        if size(node) <= limit or not node.named_children:
            segs.append(_Seg(node.start_point[0], node.end_point[0], _symbol(node), context))
            return
        # too big: descend; definitions become context for their members
        inner = context
        if node.type in DEF_TYPES:
            header = lines[node.start_point[0]].strip()[:120]
            inner = f"{context} > {header}" if context else header
        for child in node.named_children:
            visit(child, inner)

    for child in root.named_children:
        visit(child, "")
    return segs or [_Seg(0, len(lines) - 1)]


def _symbol(node) -> str:
    if node.type not in DEF_TYPES:
        return ""
    if node.type == "decorated_definition":
        node = node.child_by_field_name("definition") or node
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", "replace")
    first = node.text.decode("utf-8", "replace").split("(")[0].split("{")[0]
    return " ".join(first.split())[-80:]  # e.g. C++ "void Foo::bar"


# ---------- shared ----------

def _normalize(segs: list[_Seg], n_lines: int) -> list[_Seg]:
    """Sort, remove overlaps, attach uncovered lines (blank gaps) to the previous segment."""
    out, nxt = [], 0
    for s in sorted(segs, key=lambda s: (s.start, s.end)):
        if s.end < nxt:
            continue
        start = max(s.start, nxt)
        if out:
            out[-1].end = start - 1
        else:
            start = 0
        out.append(_Seg(start, s.end, s.symbol, s.context))
        nxt = s.end + 1
    if nxt < n_lines:
        if out:
            out[-1].end = n_lines - 1
        else:
            out.append(_Seg(0, n_lines - 1))
    return out


def _chars(seg: _Seg, lines: list[str]) -> int:
    return sum(len(l) + 1 for l in lines[seg.start:seg.end + 1])


def _split_big(seg: _Seg, lines: list[str]) -> list[_Seg]:
    if _chars(seg, lines) <= TARGET_CHARS * 2:
        return [seg]
    out, i = [], seg.start
    while i <= seg.end:
        # window is capped by lines and by chars (tables/URLs make lines very long)
        j, size = i, len(lines[i]) + 1
        while (j < seg.end and j + 1 - i < WINDOW_LINES
               and size + len(lines[j + 1]) + 1 <= TARGET_CHARS):
            j += 1
            size += len(lines[j]) + 1
        out.append(_Seg(i, j, seg.symbol, seg.context))
        if j == seg.end:
            break
        i = max(j + 1 - min(WINDOW_OVERLAP, (j - i) // 4), i + 1)  # overlap, but always advance
    return out


def _merge(segs: list[_Seg], lines: list[str]) -> list[list[_Seg]]:
    groups, cur, cur_size = [], [], 0
    for s in segs:
        n = _chars(s, lines)
        # window pieces overlap, merging them would duplicate lines
        overlapping = cur and s.start <= cur[-1].end
        if cur and (cur_size + n > TARGET_CHARS or overlapping):
            groups.append(cur)
            cur, cur_size = [], 0
        cur.append(s)
        cur_size += n
    if cur:
        groups.append(cur)
    return groups


def _to_chunk(group: list[_Seg], lines: list[str]) -> Chunk:
    start, end = group[0].start, max(s.end for s in group)
    symbols = list(dict.fromkeys(s.symbol for s in group if s.symbol))
    context = group[0].context
    return Chunk(
        text="\n".join(lines[start:end + 1]),
        line_start=start + 1,
        line_end=end + 1,
        symbol=", ".join(symbols[:3]) + (" …" if len(symbols) > 3 else ""),
        context=context,
    )
