"""Chats: each one belongs to a working folder in a collection and is saved in data/chats/.

A run is detached from the HTTP request: the agent works in its own thread and appends events
to the chat, so the UI can leave and reattach to an answer in progress. Events are folded into
the chat's turns on the server too, which is what a reattaching UI loads before streaming the rest.
Only one run at a time: there is one local model.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Iterator

from agents.base import Agent
from agents.supervisor import supervisor_agent
from core.config import PROJECT_DIR
from core.console import Console
from core.llm import LLM
from core.workspace import Workspace
from rag.store import Store

CHATS_DIR = PROJECT_DIR / "data" / "chats"
APPROVAL_TIMEOUT_S = 600
TITLE_CHARS = 80

Emit = Callable[[dict], None]


class Cancelled(BaseException):
    """BaseException, not Exception: tool wrappers catch Exception and would hand a stop
    inside a sub-agent back to the supervisor as an ordinary tool error."""


class EventConsole(Console):
    def __init__(self, emit: Emit, cancel: threading.Event, agent: str = "supervisor",
                 quiet: bool = False):
        super().__init__(verbose=True, quiet=quiet)
        self.emit, self.cancel, self.agent = emit, cancel, agent

    def _check(self) -> None:
        # called on every streamed token: leaving the stream closes the request to the model
        if self.cancel.is_set():
            raise Cancelled()

    def text(self, s: str) -> None:
        self._check()
        if not self.quiet and s:
            self.emit({"type": "text", "text": s})

    def thinking(self, s: str) -> None:
        self._check()
        if not self.quiet and s:
            self.emit({"type": "thinking", "agent": self.agent, "text": s})

    def newline(self) -> None:
        pass  # the UI handles layout

    def log(self, s: str) -> None:
        self._check()
        self.emit({"type": "log", "agent": self.agent, "text": s})

    def tool_call(self, name: str, args, size: int) -> None:
        self._check()
        if isinstance(args, str):  # OpenAI-style JSON string
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                pass
        self.emit({"type": "tool", "agent": self.agent, "name": name, "args": args, "chars": size})

    def child(self, name: str) -> Console:
        return EventConsole(self.emit, self.cancel, name, quiet=True)


def new_turn(question: str) -> dict:
    return {"question": question, "answer": "", "thinking": "", "steps": [], "approvals": [],
            "error": None, "status": "running", "started": time.time(), "finished": None}


def apply_event(turn: dict, e: dict) -> None:
    """Fold one event into a turn; the UI's reducer mirrors this."""
    kind = e["type"]
    if kind == "text":
        turn["answer"] += e["text"]
    elif kind == "thinking":
        turn["thinking"] += e["text"]
    elif kind == "log":
        turn["steps"].append({"agent": e["agent"], "text": e["text"]})
    elif kind == "tool":
        turn["steps"].append({k: e[k] for k in ("agent", "name", "args", "chars")})
    elif kind == "approval":
        turn["approvals"].append({"id": e["id"], "prompt": e["prompt"], "state": "pending"})
    elif kind == "approval_done":
        for a in turn["approvals"]:
            if a["id"] == e["id"]:
                a["state"] = "approved" if e["approved"] else "declined"
    elif kind in ("done", "error", "stopped"):
        turn["status"] = kind
        turn["error"] = e.get("message")
        turn["finished"] = time.time()


class Chat:
    def __init__(self, data: dict, file: Path):
        self.id: str = data["id"]
        self.collection: str = data["collection"]
        self.root: str = data["root"]
        self.title: str = data.get("title", "")
        self.created: float = data["created"]
        self.updated: float = data["updated"]
        self.turns: list[dict] = data.get("turns", [])
        self.messages: list = data.get("messages", [])  # LLM history (OpenAI-style dicts)
        self.file = file
        for t in self.turns:  # the app was closed mid-answer
            if t["status"] == "running":
                apply_event(t, {"type": "stopped", "message": "Interrupted: the app was closed"})
        self.agent: Agent | None = None
        self.had_index = False
        self.events: list[dict] = []  # of the current or last run; streams index into it
        self.cond = threading.Condition()
        self.cancel = threading.Event()
        self.approvals: dict[str, dict] = {}

    @property
    def running(self) -> bool:
        return bool(self.turns) and self.turns[-1]["status"] == "running"

    def summary(self) -> dict:
        return {"id": self.id, "collection": self.collection, "root": self.root, "title": self.title,
                "created": self.created, "updated": self.updated, "turns": len(self.turns),
                "running": self.running}

    def to_json(self) -> dict:
        """Full chat for the UI; `offset` is where a stream of the live run continues."""
        with self.cond:
            return {**self.summary(), "items": self.turns, "offset": len(self.events)}

    def save(self) -> None:
        with self.cond:
            data = {**self.summary(), "turns": self.turns, "messages": self.messages}
            data.pop("running")
            text = json.dumps(data, ensure_ascii=False, default=_jsonable)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.file)  # atomic: a crash mid-write keeps the previous version

    def emit(self, e: dict) -> None:
        with self.cond:
            apply_event(self.turns[-1], e)
            self.events.append(e)
            self.cond.notify_all()

    def stream(self, after: int) -> Iterator[dict | None]:
        """Events from index `after` until the run ends; None is a keep-alive tick."""
        i = after
        while True:
            with self.cond:
                if i >= len(self.events) and self.running:
                    self.cond.wait(timeout=1.0)
                new, i = self.events[i:], max(i, len(self.events))
                finished = not self.running
            yield from new
            if finished and not new:
                return
            if not new:
                yield None  # keep-alive tick


def _jsonable(o):
    if hasattr(o, "model_dump"):  # SDK objects in Anthropic-style history
        return o.model_dump()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class ChatManager:
    def __init__(self, llm: LLM, store_for: Callable[[str], Store], folder: Path = CHATS_DIR):
        self.llm, self.store_for, self.folder = llm, store_for, folder
        self.chats: dict[str, Chat] = {}
        self.run_lock = threading.Lock()
        for f in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                chat = Chat(json.loads(f.read_text(encoding="utf-8")), f)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping broken chat {f.name}: {e}")
                continue
            self.chats[chat.id] = chat

    def list(self, collection: str, root: str) -> list[dict]:
        chats = [c for c in self.chats.values() if (c.collection, c.root) == (collection, root)]
        return [c.summary() for c in sorted(chats, key=lambda c: c.updated, reverse=True)]

    def get(self, cid: str) -> Chat:
        return self.chats[cid]  # KeyError -> 404

    def create(self, collection: str, root: str) -> Chat:
        now, cid = time.time(), uuid.uuid4().hex[:12]
        chat = Chat({"id": cid, "collection": collection, "root": root,
                     "created": now, "updated": now}, self.folder / f"{cid}.json")
        self.chats[cid] = chat  # saved with the first question: empty chats leave no files
        return chat

    def delete(self, cid: str) -> None:
        chat = self.get(cid)
        if chat.running:
            raise RuntimeError("Stop the answer first")
        del self.chats[cid]
        chat.file.unlink(missing_ok=True)

    def running(self) -> str | None:
        return next((c.id for c in self.chats.values() if c.running), None)

    def ask(self, cid: str, message: str) -> None:
        """Starts a run in the background; events go to the chat."""
        chat = self.get(cid)
        if not self.run_lock.acquire(blocking=False):
            raise RuntimeError("Another answer is still running")
        with chat.cond:
            chat.events = []
            chat.cancel.clear()
            chat.turns.append(new_turn(message))
            chat.title = chat.title or message.strip().splitlines()[0][:TITLE_CHARS]
            chat.updated = time.time()
        chat.save()
        threading.Thread(target=self._work, args=(chat, message), daemon=True, name="chat").start()

    def stop(self, cid: str) -> None:
        chat = self.get(cid)
        chat.cancel.set()
        for slot in list(chat.approvals.values()):  # a pending approval would block for minutes
            slot["event"].set()

    def approve(self, cid: str, aid: str, approved: bool) -> bool:
        slot = self.get(cid).approvals.get(aid)
        if not slot:
            return False
        slot["approved"] = approved
        slot["event"].set()
        return True

    def _agent(self, chat: Chat) -> Agent:
        # rebuild when the index appeared after the first message
        store = self.store_for(chat.collection)
        if chat.agent is None or (not chat.had_index and store.exists()):
            chat.had_index = store.exists()
            chat.agent = supervisor_agent(self.llm, Workspace(chat.root), chat.collection,
                                          verbose=True, confirm=lambda p: self._confirm(chat, p),
                                          store=store)
        return chat.agent

    def _confirm(self, chat: Chat, prompt: str) -> bool:
        aid = uuid.uuid4().hex[:8]
        slot = {"event": threading.Event(), "approved": False}
        chat.approvals[aid] = slot
        chat.emit({"type": "approval", "id": aid, "prompt": prompt})
        slot["event"].wait(APPROVAL_TIMEOUT_S)
        chat.approvals.pop(aid, None)
        chat.emit({"type": "approval_done", "id": aid, "approved": slot["approved"]})
        if chat.cancel.is_set():
            raise Cancelled()
        return slot["approved"]

    def _work(self, chat: Chat, message: str) -> None:
        history_len = len(chat.messages)
        try:
            agent = self._agent(chat)
            chat.messages.append({"role": "user", "content": message})
            agent.run(self.llm, chat.messages, EventConsole(chat.emit, chat.cancel))
            chat.emit({"type": "done"})
        except Cancelled:
            # the model never saw a finished exchange: keep its history as before the question
            del chat.messages[history_len:]
            chat.emit({"type": "stopped"})
        except Exception as e:
            del chat.messages[history_len:]
            chat.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            chat.updated = time.time()
            chat.save()
            self.run_lock.release()
