# Docs Agent

[![CI](https://github.com/AleksAndriushyn/docs-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/AleksAndriushyn/docs-agent/actions/workflows/ci.yml)

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
| `research`   | `rag_search` over the Qdrant index (reranked) + `grep` / `read_file` / `list_dir` in `--root` (default) |
| `tester`     | writes pytest tests for code in `--root`, runs them in `sandbox/`    |
| `supervisor` | delegates to the two agents above (`ask_researcher`, `ask_tester`)   |

```bash
python agent.py --root ./docs -q "Which parameter enables throttle failsafe?"
python agent.py --agent research --collection work --root ~/Documents/MissionPlanner --provider local -v
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

## Evals

Two levels:

- **Retrieval** (`evals/retrieval.py`): does the right file reach the top-k? hit@k and MRR, hybrid vs reranked.
- **Answers** (`evals/run.py`): the researcher (index only) answers each question in `evals/qa_*.jsonl`, an
  LLM judge compares the answer with a hand-checked reference fact by fact, and citations are
  checked against the expected files. Questions without an answer in the base check for
  hallucinations: the agent should say it found nothing.

```bash
python -m evals.run evals/qa_docs.jsonl --collection docs --provider local
python -m evals.run evals/qa_work.jsonl --collection work --source ~/Documents/MissionPlanner --provider anthropic
python -m evals.run evals/results/<run>.json --rejudge --judge claude   # re-grade saved answers
python -m evals.report evals/results/*.json --md evals/REPORT.md
```

The judge is the local model by default (free); `--judge-model gemma4:26b` picks another local
model, `--judge claude-code` grades through Claude Code on the Claude subscription (no API
credits), `--judge claude` uses the paid API.
`--source` limits retrieval to one indexed folder, so other folders in a shared collection
never reach the API. Every answer also records tokens, cost and latency.

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
agents/         researcher, tester, supervisor
app/            desktop app: FastAPI + pywebview, React UI in app/web
tests/          pytest
```

## Tests

```bash
pytest
```
