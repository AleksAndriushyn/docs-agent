# The app (FastAPI + built React UI) without the native window. Ollama stays on the host.
#   docker compose --profile app up -d --build      -> http://127.0.0.1:8765

FROM node:24-slim AS web
WORKDIR /web
COPY app/web/package.json app/web/package-lock.json ./
RUN npm ci
COPY app/web/ ./
RUN npm run build

FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # BM25 and the ~1 GB reranker download here on first use: keep it in a volume
    FASTEMBED_CACHE_PATH=/cache/fastembed
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY agents agents
COPY app app
COPY core core
COPY mcp_servers mcp_servers
COPY rag rag
COPY --from=web /web/dist app/web/dist
EXPOSE 8765
# uvicorn directly: `python -m app` would try to open a pywebview window
CMD ["uvicorn", "app.server:api", "--host", "0.0.0.0", "--port", "8765"]
