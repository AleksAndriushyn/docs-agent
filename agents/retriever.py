from __future__ import annotations

import re
from pathlib import Path

from agents.base import Agent
from agents.citations import Citations
from core.tools import Tool, schema
from core.workspace import Workspace
from rag.search import Retriever, format_hits
from rag.store import Store

MAX_K = 8  # keeps the local model's context window from overflowing
# repeated in every tool result: small models follow a fresh reminder better than the system prompt
CITE_HINT = "\n\n(Cite as [n] after each claim.)"

SYSTEM = """You answer questions using an indexed knowledge base of work files (code and documents).
- Search with rag_search. If results are weak, rephrase: synonyms, English terms,
  parameter/function names. Usually 2-3 searches are needed.
- If a chunk is not enough, read more of the file with open_file and a line range.
- Use ONLY the retrieved text, not your own knowledge: every parameter name, value and
  step in the answer must appear in a retrieved chunk. Similar projects use different
  names for similar things, so never fill gaps from memory.
- Every chunk header starts with a number like [3]. Put that number right after each
  claim it supports, e.g. "THR_FS_VALUE defaults to 950 [3]." Write only [n]: no file
  paths, no URLs. The list of sources is appended automatically.
- If the knowledge base has no answer, say so.
- Answer in the user's language."""


def rag_tools(retriever: Retriever, cites: Citations) -> list[Tool]:
    def rag_search(query: str, k: int = 6, lang: str | None = None,
                   path_contains: str | None = None) -> str:
        hits = retriever.search(query, min(k, MAX_K), lang, path_contains)
        if not hits:
            return format_hits(hits)
        return format_hits(hits, ids=[cites.add(h.display_ref()) for h in hits]) + CITE_HINT

    def open_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        source = retriever.store.source_of(path)
        if source is None:
            raise ValueError(f"{path} is not in the knowledge base")
        text = Workspace(Path(source)).read_file(path, start_line, end_line)
        numbered = re.findall(r"^(\d+): ", text, re.M)  # lines actually returned (may be truncated)
        n = cites.add(f"{Path(source).name}/{path}:{start_line}-{numbered[-1] if numbered else start_line}")
        return f"[{n}] {path}\n{text}{CITE_HINT}"

    return [
        Tool("rag_search",
             "Hybrid (semantic + keyword) search over the knowledge base. Returns numbered "
             "chunks: [n] path:line_start-line_end.",
             schema({
                 "query": {"type": "string", "description": "Natural-language query or key terms"},
                 "k": {"type": "integer", "minimum": 1, "maximum": MAX_K},
                 "lang": {"type": "string", "enum": ["python", "csharp", "cpp", "markdown", "rst",
                                                     "text", "docx", "xlsx"]},
                 "path_contains": {"type": "string", "description": "Substring filter on the path"},
             }, ["query"]),
             rag_search),
        Tool("open_file", "Read lines of an indexed file (path as shown by rag_search).",
             schema({"path": {"type": "string"},
                     "start_line": {"type": "integer", "minimum": 1},
                     "end_line": {"type": "integer", "minimum": 1}}, ["path"]),
             open_file),
    ]


def retriever_agent(collection: str, store: Store | None = None) -> Agent:
    cites = Citations()
    return Agent(
        name="retriever",
        description="Semantic search over the work-file knowledge base (C#/C++/Python code, docs, "
                    "docx/xlsx). Best for how/where/why questions when the exact "
                    "file name is unknown.",
        system=SYSTEM,
        tools=rag_tools(Retriever(collection, store), cites),
        citations=cites,
    )
