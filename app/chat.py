"""Chat session: runs the supervisor in a worker thread and turns its console output
into events for the UI (text, thinking, tool calls per agent, approvals, done/error)."""
from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Callable, Iterator

from agents.base import Agent
from agents.supervisor import supervisor_agent
from core.console import Console
from core.llm import LLM
from core.workspace import Workspace
from rag.store import Store

APPROVAL_TIMEOUT_S = 600

Emit = Callable[[dict], None]


class EventConsole(Console):
    def __init__(self, emit: Emit, agent: str = "supervisor", quiet: bool = False):
        super().__init__(verbose=True, quiet=quiet)
        self.emit, self.agent = emit, agent

    def text(self, s: str) -> None:
        if not self.quiet and s:
            self.emit({"type": "text", "text": s})

    def thinking(self, s: str) -> None:
        if not self.quiet and s:
            self.emit({"type": "thinking", "agent": self.agent, "text": s})

    def newline(self) -> None:
        pass  # the UI handles layout

    def log(self, s: str) -> None:
        self.emit({"type": "log", "agent": self.agent, "text": s})

    def tool_call(self, name: str, args, size: int) -> None:
        if isinstance(args, str):  # OpenAI-style JSON string
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                pass
        self.emit({"type": "tool", "agent": self.agent, "name": name, "args": args, "chars": size})

    def child(self, name: str) -> Console:
        return EventConsole(self.emit, name, quiet=True)


class ChatSession:
    """Single-user session: one conversation at a time, tied to one working folder."""

    def __init__(self, llm: LLM, store_for: Callable[[str], Store], collection: str):
        self.llm, self.store_for, self.collection = llm, store_for, collection
        self.root: str | None = None
        self.messages: list = []
        self.agent: Agent | None = None
        self.had_index = False
        self.run_lock = threading.Lock()
        self.approvals: dict[str, dict] = {}
        self._emit: Emit = lambda e: None

    def use(self, collection: str, root: str) -> None:
        """Another collection or working folder starts a new conversation."""
        if (collection, root) != (self.collection, self.root):
            self.collection = collection
            self.reset(root=root)

    def reset(self, root: str | None = None) -> None:
        self.messages = []
        self.agent = None
        if root is not None:
            self.root = root

    def _agent(self) -> Agent:
        # rebuild when the index appeared after the first message
        store = self.store_for(self.collection)
        if self.agent is None or (not self.had_index and store.exists()):
            self.had_index = store.exists()
            self.agent = supervisor_agent(self.llm, Workspace(self.root), self.collection,
                                          verbose=True, confirm=self._confirm, store=store)
        return self.agent

    def _confirm(self, prompt: str) -> bool:
        aid = uuid.uuid4().hex[:8]
        slot = {"event": threading.Event(), "approved": False}
        self.approvals[aid] = slot
        self._emit({"type": "approval", "id": aid, "prompt": prompt})
        slot["event"].wait(APPROVAL_TIMEOUT_S)
        self.approvals.pop(aid, None)
        self._emit({"type": "approval_done", "id": aid, "approved": slot["approved"]})
        return slot["approved"]

    def approve(self, aid: str, approved: bool) -> bool:
        slot = self.approvals.get(aid)
        if not slot:
            return False
        slot["approved"] = approved
        slot["event"].set()
        return True

    def run(self, message: str) -> Iterator[dict]:
        """Blocking iterator of events; the agent runs in its own thread."""
        if not self.run_lock.acquire(blocking=False):
            yield {"type": "error", "message": "A previous request is still running."}
            return
        events: queue.Queue[dict | None] = queue.Queue()
        self._emit = events.put

        def work() -> None:
            try:
                agent = self._agent()
                self.messages.append({"role": "user", "content": message})
                agent.run(self.llm, self.messages, EventConsole(events.put))
                events.put({"type": "done"})
            except Exception as e:
                events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                events.put(None)
                # released here, not by the reader: a closed browser tab must not keep it locked
                self.run_lock.release()

        threading.Thread(target=work, daemon=True, name="chat").start()
        while (event := events.get()) is not None:
            yield event
