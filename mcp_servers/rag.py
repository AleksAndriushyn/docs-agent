"""MCP server over the RAG index: any MCP client (Claude Code, Claude Desktop, a LangChain
agent) can search the Qdrant collections and read the indexed files.

    python -m mcp_servers.rag                      # stdio, started by the client
    claude mcp add docs-rag -- <venv>/bin/python -m mcp_servers.rag

stdout is the protocol channel over stdio: nothing here may print to it.
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from core.config import RAG_COLLECTION
from core.workspace import Workspace
from rag.search import Retriever, format_hits
from rag.store import Store

MAX_K = 10

mcp = MCPServer(
    "docs-rag",
    log_level="WARNING",  # INFO logs every Qdrant/Ollama HTTP request
    instructions="Search indexed code and documentation (MissionPlanner, MpRpc, ArduPilot docs). "
                 "Call list_collections first if unsure which collection to use, then rag_search; "
                 "read_file shows more of a file by the path rag_search returned.",
)
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_retrievers: dict[str, Retriever] = {}  # embedders and the reranker load once per collection


def _retriever(collection: str) -> Retriever:
    if collection not in _retrievers:
        try:
            _retrievers[collection] = Retriever(collection)
        except SystemExit as e:  # Retriever exits on a missing collection; a server must not
            raise ToolError(str(e)) from None
    return _retrievers[collection]


@mcp.tool(annotations=READ_ONLY)
def list_collections() -> str:
    """List Qdrant collections with the indexed folders and their chunk counts."""
    store = Store(RAG_COLLECTION)
    lines = []
    for c in sorted(c.name for c in store.client.get_collections().collections):
        sources = Store(c, client=store.client).sources()
        folders = ", ".join(f"{Path(s).name} ({n} chunks)" for s, n in sorted(sources.items()))
        lines.append(f"- {c}: {folders or 'empty'}")
    return "\n".join(lines) or "no collections"


@mcp.tool(annotations=READ_ONLY)
def rag_search(query: str, collection: str = RAG_COLLECTION, k: int = 6,
               lang: str | None = None, path_contains: str | None = None) -> str:
    """Hybrid (semantic + BM25) search with reranking over one collection.

    Returns chunks titled `path:line_start-line_end`. Rephrase with English terms or
    identifiers if results are weak.

    Args:
        query: Natural-language question or key terms.
        collection: Collection name from list_collections.
        k: Number of chunks, 1-10.
        lang: Filter: python, csharp, cpp, markdown, rst, text, docx, xlsx.
        path_contains: Substring filter on the file path, e.g. 'GCSViews/'.
    """
    hits = _retriever(collection).search(query, max(1, min(k, MAX_K)), lang, path_contains)
    return format_hits(hits)


@mcp.tool(annotations=READ_ONLY)
def read_file(path: str, collection: str = RAG_COLLECTION, start_line: int = 1,
              end_line: int | None = None) -> str:
    """Read numbered lines of an indexed file; long files are truncated, continue with start_line.

    Args:
        path: Path as rag_search printed it (relative to the indexed folder).
        collection: Collection the file was found in.
        start_line: First line, 1-based.
        end_line: Last line, inclusive.
    """
    source = _retriever(collection).store.source_of(path)
    if source is None:
        raise ToolError(f"{path} is not indexed in collection '{collection}'")
    return Workspace(source).read_file(path, start_line, end_line)


if __name__ == "__main__":
    mcp.run("stdio")
