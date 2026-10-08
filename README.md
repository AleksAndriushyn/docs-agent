# Docs Agent

CLI agents that answer questions about your files via tool calling. Fully local mode:
Ollama (LLM + embeddings) and Qdrant in embedded mode, nothing leaves the machine.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull qwen3-embedding:0.6b                 # embeddings for RAG
OLLAMA_CONTEXT_LENGTH=32768 ollama serve          # default context is too small for agents

docker compose up -d                              # Qdrant server, dashboard: http://localhost:6333/dashboard
# and QDRANT_URL=http://localhost:6333 in .env; without it an embedded store in ./qdrant_data is used
```

## Agents

| `--agent`    | What it does                                                         |
|--------------|----------------------------------------------------------------------|
| `files`      | `list_dir` / `read_file` / regex `search` over `--root` (default)    |
| `rag`        | hybrid search over the Qdrant index + `open_file` for more context   |
| `tester`     | writes pytest tests for code in `--root`, runs them in `sandbox/`    |
| `supervisor` | delegates to the three agents above (`ask_retriever`, `ask_files`, `ask_tester`) |

```bash
python agent.py --root ./docs -q "Which parameter enables throttle failsafe?"
python agent.py --agent rag --provider local -v
python agent.py --agent tester --root ./src --provider local
python agent.py --agent supervisor --root ./src --provider local -v
```

`-v` shows tool calls, sub-agent activity and the model's reasoning.
The tester asks `[y/N]` before every pytest run.

## RAG index

```bash
python -m rag.ingest --src ~/work --collection docs     # incremental: unchanged files are skipped
python -m rag.ingest --src ~/work --reset                # rebuild from scratch
python -m rag.search "throttle failsafe" -k 5 --full     # debug retrieval without an LLM
```

Supported: `.py .cs .c/.cpp/.h/.hpp` (chunked by functions/classes via tree-sitter),
`.md .rst .txt` (by headings), `.docx .xlsx`.
Each chunk is stored with a dense vector (Ollama) and a BM25 sparse vector; search fuses both with RRF.
The top 16 hybrid candidates are then reordered by a multilingual cross-encoder
(`jina-reranker-v2-base-multilingual`, ~1.9 s per query on CPU); `--no-rerank` in `rag.search` compares.

```bash
python -m evals.retrieval evals/retrieval_work.jsonl --collection work   # hit@k / MRR, hybrid vs reranked
```

## Desktop app

```bash
cd app/web && npm install && npm run build && cd ../..   # once, and after UI changes
python -m app                                          # native window
python -m app --browser                                # same UI in the browser
```

- Left panel: add folders or files (native pickers), see indexing progress, edit include/exclude
  globs, reindex or remove a source. Sources live in `data/sources.json`; on first start the
  ones from `sources/*.args` are imported, so nothing indexed via the CLI is redone.
- Chat: the supervisor on the local model, streamed, with agent activity, model reasoning,
  clickable `[n]` citations (open the file in VS Code at the line) and Run/Decline for tests.
- The selected source is the working folder for the `files` and `tester` agents.

UI development: `python -m app --browser` for the API, then `npm run dev` in `app/web` (proxies `/api`).

## Layout

```
agent.py        CLI
core/           agent loop for both providers, tools, workspace, console output
rag/            loaders, chunking, embeddings, Qdrant store, ingest, search
agents/         files, retriever, tester, supervisor
app/            desktop app: FastAPI + pywebview, React UI in app/web
tests/          pytest
```

## Tests

```bash
pytest
```
