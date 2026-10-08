"""Researcher: one agent for both ways of finding things.

- rag_search: semantic + keyword search over the indexed knowledge base (reranked)
- grep / list_dir: exact regex and structure in the working folder, including files
  the index doesn't cover (.param, .xml, .json, logs) and edits not yet reindexed
- read_file: the working folder first, then indexed sources (paths from rag_search)

One agent instead of two sub-agents saves a full LLM loop per question on a local model.
"""
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

SYSTEM = """You answer questions about work files: code and documents.
Tools:
- rag_search: search by meaning in the indexed knowledge base. Start here for
  how/where/why questions. If results are weak, rephrase: synonyms, English terms,
  parameter/function names. Usually 2-3 searches are needed.
- grep: exact regex over the working folder. Use it for known identifiers ("all uses of
  FDMenuMap"), for files the index doesn't cover (.param, .xml, .json, logs) and to
  confirm that a name really exists.
- read_file: read more of a file, by a path from rag_search or grep, with a line range.
- list_dir: folder structure.
Rules:
- Use ONLY what the tools returned, not your own knowledge: every parameter name, value and
  step in the answer must appear in tool output. Similar projects use different names for
  similar things, so never fill gaps from memory.
- Results of rag_search and read_file start with a number like [3]. Put that number right
  after each claim it supports, e.g. "THR_FS_VALUE defaults to 950 [3]." Write only [n]:
  no file paths, no URLs. The list of sources is appended automatically.
- If nothing answers the question, say so.
- Answer in the user's language."""

SYSTEM_NO_FOLDER = SYSTEM.replace(
    "- grep:", "- (no working folder: grep and list_dir are unavailable)\n- grep:", 1)


def workspace_tools(ws: Workspace) -> list[Tool]:
    """Plain folder tools (no citations); the tester uses them too."""
    return [
        Tool("list_dir", "List files and folders at a relative path inside the working folder.",
             schema({"path": {"type": "string", "description": "E.g. '.' or 'src/'"}}),
             ws.list_dir),
        Tool("read_file", "Read a text file with line numbers. Long files are truncated, "
                          "continue with start_line.",
             schema({"path": {"type": "string"},
                     "start_line": {"type": "integer", "minimum": 1},
                     "end_line": {"type": "integer", "minimum": 1}}, ["path"]),
             ws.read_file),
        Tool("grep", "Case-insensitive regex search over file contents in the working folder. "
                     "Returns path:line: text.",
             schema({"pattern": {"type": "string"},
                     "glob": {"type": "string", "description": "File filter, e.g. '**/*.cs'"}},
                    ["pattern"]),
             ws.search),
    ]


def research_tools(ws: Workspace | None, retriever: Retriever | None, cites: Citations) -> list[Tool]:
    tools: list[Tool] = []
    if retriever is not None:
        def rag_search(query: str, k: int = 6, lang: str | None = None,
                       path_contains: str | None = None) -> str:
            hits = retriever.search(query, min(k, MAX_K), lang, path_contains)
            if not hits:
                return format_hits(hits)
            return format_hits(hits, ids=[cites.add(h.display_ref()) for h in hits]) + CITE_HINT

        tools.append(Tool(
            "rag_search",
            "Hybrid (semantic + keyword) search over the knowledge base. Returns numbered "
            "chunks: [n] path:line_start-line_end.",
            schema({
                "query": {"type": "string", "description": "Natural-language query or key terms"},
                "k": {"type": "integer", "minimum": 1, "maximum": MAX_K},
                "lang": {"type": "string", "enum": ["python", "csharp", "cpp", "markdown", "rst",
                                                    "text", "docx", "xlsx"]},
                "path_contains": {"type": "string", "description": "Substring filter on the path"},
            }, ["query"]),
            rag_search))

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        root = None
        if ws is not None and ws.resolve(path).is_file():
            root = ws.root
        elif retriever is not None:
            source = retriever.store.source_of(path)
            root = Path(source) if source else None
        if root is None:
            raise ValueError(f"{path} not found in the working folder or the knowledge base")
        text = Workspace(root).read_file(path, start_line, end_line)
        numbered = re.findall(r"^(\d+): ", text, re.M)  # lines actually returned (may be truncated)
        n = cites.add(f"{root.name}/{path}:{start_line}-{numbered[-1] if numbered else start_line}")
        return f"[{n}] {path}\n{text}{CITE_HINT}"

    tools.append(Tool(
        "read_file",
        "Read lines of a file: from the working folder, or an indexed file by the path "
        "rag_search showed. Returns numbered lines; long files are truncated.",
        schema({"path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1}}, ["path"]),
        read_file))
    if ws is not None:
        tools += [t for t in workspace_tools(ws) if t.name != "read_file"]
    return tools


def researcher_agent(ws: Workspace | None = None, collection: str | None = None,
                     store: Store | None = None, sources: list[str] | None = None) -> Agent:
    retriever = None
    if collection:
        try:
            retriever = Retriever(collection, store, sources=sources)
        except SystemExit as e:  # no index yet: folder tools only
            print(f"[researcher] {e}. Working without the knowledge base.")
    if retriever is None and ws is None:
        raise SystemExit("researcher needs a working folder or an indexed collection")
    cites = Citations()
    return Agent(
        name="researcher",
        description="Finds answers in work files: semantic search over the indexed knowledge base "
                    "(C#/C++/Python code, docs, docx/xlsx) plus exact regex search and reading in "
                    "the working folder. Give it the question with any known names or paths.",
        system=SYSTEM if ws is not None else SYSTEM_NO_FOLDER,
        tools=research_tools(ws, retriever, cites),
        citations=cites,
    )
