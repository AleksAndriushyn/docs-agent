"""HTTP API for the desktop UI. Chat answers stream as Server-Sent Events."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from app.chat import ChatSession
from app.indexer import Indexer
from app.sources import Registry
from core.config import EMBED_MODEL, LOCAL_MODEL, LOCAL_URL, QDRANT_URL
from core.llm import make_llm
from rag.store import Store

COLLECTION = os.getenv("APP_COLLECTION", "work")
WEB_DIST = Path(__file__).parent / "web" / "dist"

store = Store(COLLECTION)  # one client for everything: embedded Qdrant allows only one
registry = Registry(collection=COLLECTION)
indexer = Indexer(registry, store)
chat = ChatSession(make_llm("local"), store, COLLECTION)

api = FastAPI(title="Docs Agent")


def source_json(src) -> dict:
    return {**src.to_json(), "progress": indexer.progress.get(src.id)}


# ---------- status

@api.get("/api/status")
def status() -> dict:
    try:
        ollama = httpx.get(f"{LOCAL_URL}/models", timeout=2).is_success
    except httpx.HTTPError:
        ollama = False
    try:
        points, qdrant = store.count(), True
    except Exception:
        points, qdrant = 0, False
    return {"model": LOCAL_MODEL, "embed_model": EMBED_MODEL, "collection": COLLECTION,
            "points": points, "ollama": ollama, "qdrant": qdrant,
            "qdrant_mode": "server" if QDRANT_URL else "embedded",
            "root": chat.root}


# ---------- sources

class AddSource(BaseModel):
    path: str


class EditSource(BaseModel):
    include: list[str]
    exclude: list[str]


@api.get("/api/sources")
def list_sources() -> list[dict]:
    return [source_json(s) for s in registry.list()]


@api.post("/api/sources")
def add_source(body: AddSource) -> dict:
    try:
        src = registry.add(body.path)
    except FileNotFoundError as e:
        raise HTTPException(404, f"Not found: {e}")
    indexer.enqueue(src.id)
    return source_json(src)


@api.patch("/api/sources/{sid}")
def edit_source(sid: str, body: EditSource) -> dict:
    _require(sid)
    if indexer.busy(sid):
        raise HTTPException(409, "Source is being indexed")
    src = registry.update(sid, body.include, body.exclude)
    indexer.enqueue(sid)
    return source_json(src)


@api.post("/api/sources/{sid}/reindex")
def reindex(sid: str) -> dict:
    _require(sid)
    indexer.enqueue(sid)
    return source_json(registry.get(sid))


@api.post("/api/sources/{sid}/cancel")
def cancel_indexing(sid: str) -> dict:
    _require(sid)
    indexer.cancel(sid)
    return source_json(registry.get(sid))


@api.delete("/api/sources/{sid}")
def remove_source(sid: str) -> dict:
    _require(sid)
    if indexer.busy(sid):
        raise HTTPException(409, "Source is being indexed")
    src = registry.remove(sid)
    store.delete_source(src.path)
    if chat.root == src.path:
        chat.reset(root="")
    return {"ok": True}


def _require(sid: str) -> None:
    try:
        registry.get(sid)
    except KeyError:
        raise HTTPException(404, "Unknown source")


# ---------- open a cited file

class OpenRef(BaseModel):
    ref: str  # "<source folder name>/<path>:<line_start>-<line_end>" as in "Sources:"


@api.post("/api/open")
def open_ref(body: OpenRef) -> dict:
    name, _, rest = body.ref.partition("/")
    rel, _, lines = rest.rpartition(":")
    line = lines.split("-")[0] if lines else "1"
    for src in registry.list():
        target = Path(src.path) / rel
        if Path(src.path).name == name and target.is_file():
            subprocess.Popen(_open_cmd(target, line))
            return {"ok": True, "path": str(target)}
    raise HTTPException(404, f"Cannot resolve {body.ref}")


def _open_cmd(target: Path, line: str) -> list[str]:
    """VS Code at the cited line if available, otherwise the default app."""
    if shutil.which("code"):
        return ["code", "-g", f"{target}:{line}"]
    if Path("/Applications/Visual Studio Code.app").exists():
        return ["open", f"vscode://file{quote(str(target))}:{line}"]
    return ["open", str(target)]


# ---------- chat

class ChatRequest(BaseModel):
    message: str
    root: str  # working folder for the files/tester agents


class Approval(BaseModel):
    id: str
    approved: bool


@api.post("/api/chat")
def chat_stream(body: ChatRequest) -> StreamingResponse:
    if not Path(body.root).is_dir():
        raise HTTPException(400, "Pick a working folder first")
    if body.root != chat.root:
        chat.reset(root=body.root)  # different folder = new conversation

    def sse():
        for event in chat.run(body.message):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # the sync generator blocks on a queue; run it off the event loop
    return StreamingResponse(iterate_in_threadpool(sse()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api.post("/api/chat/reset")
def chat_reset() -> dict:
    chat.reset()
    return {"ok": True}


@api.post("/api/approve")
def approve(body: Approval) -> dict:
    if not chat.approve(body.id, body.approved):
        raise HTTPException(404, "No pending approval with this id")
    return {"ok": True}


# ---------- UI

if WEB_DIST.exists():
    api.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @api.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")
