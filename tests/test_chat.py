import json
import threading
import time

import pytest

from app.chat import ChatManager


class FakeStore:
    def exists(self):
        return True


class FakeAgent:
    """Streams a few tokens; with `block`, keeps thinking until the run is cancelled."""
    def __init__(self, block: threading.Event | None = None):
        self.block = block

    def run(self, llm, messages, out):
        out.thinking("hmm")
        while self.block is not None and not self.block.is_set():
            out.thinking(".")  # the console raises here once stop() is called
            time.sleep(0.01)
        out.tool_call("rag_search", {"query": "x"}, 10)
        out.text("answer")
        messages.append({"role": "assistant", "content": "answer"})
        return "answer"


@pytest.fixture
def manager(tmp_path, monkeypatch):
    agent = {"value": FakeAgent()}
    monkeypatch.setattr("app.chat.supervisor_agent", lambda *a, **k: agent["value"])
    m = ChatManager(llm=None, store_for=lambda c: FakeStore(), folder=tmp_path / "chats")
    return m, agent, tmp_path


def wait_idle(m: ChatManager):
    for _ in range(300):
        if m.running() is None and not m.run_lock.locked():
            return
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def test_chat_is_saved_per_folder_and_reloaded(manager, tmp_path):
    m, _, _ = manager
    chat = m.create("work", str(tmp_path))
    m.ask(chat.id, "How are plugins loaded?\nsecond line")
    wait_idle(m)

    again = ChatManager(llm=None, store_for=lambda c: FakeStore(), folder=tmp_path / "chats")
    [summary] = again.list("work", str(tmp_path))
    assert summary["title"] == "How are plugins loaded?"
    assert again.list("docs", str(tmp_path)) == [] and again.list("work", "/other") == []
    turn = again.get(chat.id).turns[0]
    assert turn["status"] == "done" and turn["answer"] == "answer" and turn["thinking"] == "hmm"
    assert turn["steps"][0]["name"] == "rag_search"
    assert [msg["role"] for msg in again.get(chat.id).messages] == ["user", "assistant"]


def test_stop_rolls_back_model_history(manager, tmp_path):
    m, agent, _ = manager
    gate = threading.Event()  # closed: only stop ends the first run
    agent["value"] = FakeAgent(block=gate)
    chat = m.create("work", str(tmp_path))
    m.ask(chat.id, "q")
    with pytest.raises(RuntimeError):
        m.ask(chat.id, "second question while running")
    for _ in range(300):  # stop once the model is streaming
        if chat.turns[-1]["thinking"]:
            break
        time.sleep(0.01)
    m.stop(chat.id)
    wait_idle(m)
    assert chat.turns[-1]["status"] == "stopped" and chat.turns[-1]["thinking"].startswith("hmm")
    assert chat.messages == []  # the unfinished exchange is not sent to the model next time
    gate.set()  # the chat keeps its agent; let it finish this time
    m.ask(chat.id, "next")  # the lock was released
    wait_idle(m)
    assert chat.turns[-1]["status"] == "done"


def test_reattach_streams_the_rest(manager, tmp_path):
    m, agent, _ = manager
    gate = threading.Event()
    agent["value"] = FakeAgent(block=gate)
    chat = m.create("work", str(tmp_path))
    m.ask(chat.id, "q")
    time.sleep(0.05)
    snapshot = chat.to_json()  # what a UI coming back to the chat loads
    assert snapshot["running"] and snapshot["items"][-1]["thinking"].startswith("hmm")
    gate.set()
    rest = [e for e in chat.stream(snapshot["offset"]) if e is not None]
    assert [e["type"] for e in rest][-3:] == ["tool", "text", "done"]
    assert not any(e["type"] == "thinking" and e["text"] == "hmm" for e in rest)  # no replay


def test_running_turn_from_a_closed_app_is_marked_stopped(tmp_path, monkeypatch):
    folder = tmp_path / "chats"
    folder.mkdir()
    (folder / "c1.json").write_text(json.dumps({
        "id": "c1", "collection": "work", "root": "/r", "title": "t", "created": 1, "updated": 1,
        "turns": [{"question": "q", "answer": "", "thinking": "", "steps": [], "approvals": [],
                   "error": None, "status": "running", "started": 1, "finished": None}],
        "messages": [],
    }))
    m = ChatManager(llm=None, store_for=lambda c: FakeStore(), folder=folder)
    assert m.get("c1").turns[0]["status"] == "stopped"
    assert m.running() is None
