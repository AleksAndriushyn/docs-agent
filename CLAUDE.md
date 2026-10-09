# Docs Agent: контекст проекта

Учебный проект для перехода в AI engineering. CLI-агент на Python, который
отвечает на вопросы по файлам в папке через tool calling.

## Что уже есть
- `agent.py`: CLI, `--agent {research,tester,supervisor}`.
- `core/llm.py`: один агентный цикл со стримингом для двух провайдеров:
  - `--provider anthropic`: Anthropic SDK.
  - `--provider local`: Ollama через OpenAI-совместимый API (`localhost:11434/v1`),
    модель в `LOCAL_MODEL` (qwen3.6, thinking; температура 0.6, на низкой зацикливается).
- `rag/`: индексация в Qdrant (сервер в Docker через `docker compose`, `QDRANT_URL` в `.env`;
  без него embedded), чанкинг tree-sitter/заголовки, гибрид dense+BM25 (RRF), include/exclude.
  Реранкинг: jina-reranker-v2 multilingual поверх 16 кандидатов; бенчмарк `evals/retrieval.py`.
- `agents/`: researcher (rag_search + grep/read_file/list_dir), tester (pytest в `sandbox/`),
  supervisor (саб-агенты как инструменты),
  ссылки через номера `[n]` -> `Sources:` (agents/citations.py).
- `agents/langgraph_agent.py`: тот же researcher на LangGraph (`--agent research-lg`,
  в evals `--framework langgraph`): промпт, инструменты и цитаты общие с `run_loop`.
- `mcp_servers/rag.py`: MCP-сервер (mcp 2.x, `MCPServer`, stdio) с `list_collections`,
  `rag_search`, `read_file` поверх `rag/`; запуск `python -m mcp_servers.rag`.
- `app/`: десктоп-приложение `python -m app` (FastAPI + pywebview, React в `app/web`):
  источники в `data/sources.json`, индексация с прогрессом/остановкой, чат со стримингом (SSE).
- Рабочая коллекция `work`: MissionPlanner (основной код + плагины) и MpRpc, см. `sources/*.args`.
- Настройки через `.env` (см. `.env.example`).
- Всё в коде (комментарии, descriptions, промпты, строки CLI) на английском.

## План (по неделям)
1. Сейчас: стриминг ответа, вывод `usage`/стоимости, прогон 10 реальных вопросов
   по документации ArduPilot на обоих провайдерах.
2. RAG: эмбеддинги, pgvector/Qdrant, гибридный поиск, реранкинг.
3. Evals: набор вопросов с эталонами и скрипт оценки, сравнение local vs anthropic.
4. MCP-сервер для логов полёта (.bin/.tlog через pymavlink).
5. Прод: Langfuse, Docker, деплой, React-фронт со стримингом.

## Как работать со мной
- Это учебный проект: объясняй ключевые решения кратко, но код пиши сразу.
- Показывай только изменённые части, если не прошу весь файл.
- Ollama может быть на Windows, а скрипт в WSL: учитывай разницу localhost.
