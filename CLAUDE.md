# Docs Agent: project context

A learning project for moving into AI engineering. Python agents that answer questions
about files in a folder via tool calling.

## What exists
- `agent.py`: CLI, `--agent {research,research-lg,tester,supervisor}`.
- `core/llm.py`: one streaming agent loop for two providers:
  - `--provider anthropic`: Anthropic SDK.
  - `--provider local`: Ollama via the OpenAI-compatible API (`localhost:11434/v1`),
    model in `LOCAL_MODEL` (qwen3.6, thinking; temperature 0.6, it loops at low temperatures).
- `rag/`: indexing into Qdrant (server in Docker via `docker compose`, `QDRANT_URL` in `.env`;
  embedded without it), chunking by tree-sitter/headings, hybrid dense+BM25 (RRF), include/exclude.
  Reranking: jina-reranker-v2 multilingual over 16 candidates; benchmark in `evals/retrieval.py`.
  Indexing speed is bound by Ollama embeddings (~30 chunks/s, independent of batch size);
  chunking and BM25 take well under a second, so threading the Python side doesn't help.
- `agents/`: researcher (rag_search + grep/read_file/list_dir), tester (pytest in `sandbox/`),
  supervisor (sub-agents as tools), citations as `[n]` -> `Sources:` (agents/citations.py).
- `agents/langgraph_agent.py`: the same researcher on LangGraph (`--agent research-lg`,
  `--framework langgraph` in evals); prompt, tools and citations are shared with `run_loop`.
- `mcp_servers/rag.py`: MCP server (mcp 2.x `MCPServer`, stdio) with `list_collections`,
  `rag_search`, `read_file` over `rag/`; run with `python -m mcp_servers.rag`.
- `evals/`: retrieval (hit@k, MRR) and answer evals with an LLM judge (local, Claude API,
  or Claude Code on the subscription); results in `evals/results/`.
- `app/`: desktop app `python -m app` (FastAPI + pywebview, React in `app/web`): sources in
  `data/sources.json`, indexing with progress/cancel, chat streamed over SSE. Chats are saved per
  (collection, folder) in `data/chats/`; a run is detached from the request, so the UI can
  reattach to an answer in progress; Stop cancels on the next streamed token.
- Docker: `docker compose --profile app up -d --build` -> http://127.0.0.1:8765 (app + Qdrant;
  Ollama stays on the host via `host.docker.internal`; `~/Documents` is mounted read-only at the
  same path because sources are stored as absolute paths). Without `--profile app` only Qdrant starts.
- Work collection `work`: MissionPlanner (main code + plugins) and MpRpc, see `sources/*.args`.
- Settings via `.env` (see `.env.example`).

## Plan (by week)
1. Done: streaming, `usage`/cost output, real questions on ArduPilot docs on both providers.
2. Done: RAG with embeddings, Qdrant, hybrid search, reranking.
3. Done: evals with reference answers and a grading script, local vs anthropic.
4. MCP: the RAG server is done; next is an MCP server for flight logs (.bin/.tlog via pymavlink).
5. Prod: Docker (local) and the React UI with streaming are done; Langfuse and deployment remain.

## How to work with me
- Chat with me in Russian; everything in code (comments, descriptions, prompts, CLI strings,
  docs) is in English.
- This is a learning project: explain key decisions briefly, but write the code right away.
- Show only the changed parts unless I ask for the whole file.
- Ollama may run on Windows while the script runs in WSL: mind the localhost difference.
