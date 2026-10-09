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

from app.chat import ChatManager
from app.indexer import Indexer
from app.sources import Registry
from core.config import EMBED_MODEL, LOCAL_MODEL, LOCAL_URL, QDRANT_URL
from core.llm import make_llm
from rag.store import Store

DEFAULT_COLLECTION = os.getenv("APP_COLLECTION", "work")
WEB_DIST = Path(__file__).parent / "web" / "dist"

_base = Store(DEFAULT_COLLECTION)  # one client for everything: embedded Qdrant allows only one
_stores: dict[str, Store] = {}


def store_for(collection: str) -> Store:
    if collection not in _stores:
        _stores[collection] = Store(collection, client=_base.client)
    return _stores[collection]


def qdrant_collections() -> list[str]:
    return [c.name for c in _base.client.get_collections().collections]


registry = Registry(default_collection=DEFAULT_COLLECTION)
try:  # folders indexed elsewhere (CLI) show up in the UI without reindexing
    for name in qdrant_collections():
        indexed = store_for(name).sources()
        registry.prune_phantoms(name, indexed)
        for path in indexed:
            registry.discover(name, path)
except Exception as e:
    print(f"Qdrant not reachable at startup: {e}")
indexer = Indexer(registry, store_for)
chats = ChatManager(make_llm("local"), store_for)

api = FastAPI(title="Docs Agent")


def source_json(src) -> dict:
    return {**src.to_json(), "progress": indexer.progress.get(src.id)}


# ---------- status

@api.get("/api/status")
def status(collection: str = DEFAULT_COLLECTION) -> dict:
    try:
        ollama = httpx.get(f"{LOCAL_URL}/models", timeout=2).is_success
    except httpx.HTTPError:
        ollama = False
    try:
        points, qdrant = store_for(collection).count(), True
    except Exception:
        points, qdrant = 0, False
    return {"model": LOCAL_MODEL, "embed_model": EMBED_MODEL, "collection": collection,
            "points": points, "ollama": ollama, "qdrant": qdrant,
            "qdrant_mode": "server" if QDRANT_URL else "embedded",
            "running_chat": chats.running()}


# ---------- collections

class NewCollection(BaseModel):
    name: str


@api.get("/api/collections")
def list_collections() -> list[dict]:
    try:
        names = set(qdrant_collections())
    except Exception:
        names = set()
    out = []
    for name in sorted(names | set(registry.collections())):
        try:
            points = store_for(name).count()
        except Exception:
            points = 0
        out.append({"name": name, "points": points, "sources": len(registry.list(name))})
    return out


@api.post("/api/collections")
def create_collection(body: NewCollection) -> dict:
    try:
        registry.add_collection(body.name)  # the Qdrant collection appears on first indexing
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": body.name, "points": 0, "sources": 0}


# ---------- sources

@api.delete("/api/collections/{name}")
def delete_collection(name: str) -> dict:
    sources = registry.list(name)
    if any(indexer.busy(src.id) for src in sources):
        raise HTTPException(409, "Stop indexing in this collection first")
    store_for(name).drop()  # vectors are gone for good; files on disk are untouched
    registry.remove_collection(name)
    _stores.pop(name, None)  # chats stay on disk and come back if the collection is recreated
    return {"ok": True, "sources": len(sources)}


class AddSource(BaseModel):
    path: str
    collection: str = DEFAULT_COLLECTION


class EditSource(BaseModel):
    include: list[str]
    exclude: list[str]


@api.get("/api/sources")
def list_sources(collection: str | None = None) -> list[dict]:
    return [source_json(s) for s in registry.list(collection)]


@api.post("/api/sources")
def add_source(body: AddSource) -> dict:
    try:
        src = registry.add(body.path, body.collection)
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
    store_for(src.collection).delete_source(src.path)  # its chats stay, like the files
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
            try:
                subprocess.Popen(_open_cmd(target, line))
            except FileNotFoundError:  # in Docker there is neither `code` nor `open`
                raise HTTPException(501, f"No editor here to open {target}")
            return {"ok": True, "path": str(target)}
    raise HTTPException(404, f"Cannot resolve {body.ref}")


def _open_cmd(target: Path, line: str) -> list[str]:
    """VS Code at the cited line if available, otherwise the default app."""
    if shutil.which("code"):
        return ["code", "-g", f"{target}:{line}"]
    if Path("/Applications/Visual Studio Code.app").exists():
        return ["open", f"vscode://file{quote(str(target))}:{line}"]
    return ["open", str(target)]


# ---------- chats

class NewChat(BaseModel):
    root: str  # working folder (a source of the collection)
    collection: str = DEFAULT_COLLECTION


class Message(BaseModel):
    message: str


class Approval(BaseModel):
    id: str
    approved: bool


def _chat(cid: str):
    try:
        return chats.get(cid)
    except KeyError:
        raise HTTPException(404, "Unknown chat")


@api.get("/api/chats")
def list_chats(collection: str, root: str) -> list[dict]:
    return chats.list(collection, root)


@api.post("/api/chats")
def create_chat(body: NewChat) -> dict:
    if not Path(body.root).is_dir():
        raise HTTPException(400, "Pick a working folder first")
    return chats.create(body.collection, body.root).to_json()


@api.get("/api/chats/{cid}")
def get_chat(cid: str) -> dict:
    return _chat(cid).to_json()


@api.delete("/api/chats/{cid}")
def delete_chat(cid: str) -> dict:
    _chat(cid)
    try:
        chats.delete(cid)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@api.post("/api/chats/{cid}/messages")
def ask(cid: str, body: Message) -> dict:
    chat = _chat(cid)
    if not Path(chat.root).is_dir():
        raise HTTPException(400, f"Working folder not found: {chat.root}")
    try:
        chats.ask(cid, body.message)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return chat.to_json()


@api.get("/api/chats/{cid}/events")
def chat_events(cid: str, after: int = 0) -> StreamingResponse:
    """SSE of the live run from event index `after` (the `offset` of GET /api/chats/{cid}).
    A closed connection just stops the iteration; the run itself goes on."""
    chat = _chat(cid)

    def sse():
        for event in chat.stream(after):
            yield ": ping\n\n" if event is None else f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # the sync generator blocks on a condition; run it off the event loop
    return StreamingResponse(iterate_in_threadpool(sse()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api.post("/api/chats/{cid}/stop")
def stop_chat(cid: str) -> dict:
    _chat(cid)
    chats.stop(cid)
    return {"ok": True}


@api.post("/api/chats/{cid}/approve")
def approve(cid: str, body: Approval) -> dict:
    _chat(cid)
    if not chats.approve(cid, body.id, body.approved):
        raise HTTPException(404, "No pending approval with this id")
    return {"ok": True}


# ---------- UI

if WEB_DIST.exists():
    api.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @api.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")
