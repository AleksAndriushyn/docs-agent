from types import SimpleNamespace as NS

from core import llm as llm_mod
from core.console import Console
from core.llm import EMPTY_ANSWER_NOTE, FINAL_STEP_NOTE, LLM, run_loop
from core.tools import Tool, schema


class FakeStream(list):
    """Like openai.Stream: iterable and a context manager that closes the response."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class EndlessSearcher:
    """Fake OpenAI client: calls the tool whenever tools are offered, answers otherwise."""

    def __init__(self):
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kw):
        self.requests.append(kw)
        if "tools" in kw:
            call = NS(index=0, id=f"c{len(self.requests)}",
                      function=NS(name="search", arguments='{"q": "x"}'))
            delta = NS(content=None, tool_calls=[call])
        else:
            delta = NS(content="Not found in the knowledge base.", tool_calls=None)
        return FakeStream([NS(usage=None, choices=[NS(delta=delta)])])


def test_last_step_forces_an_answer(monkeypatch):
    monkeypatch.setattr(llm_mod, "MAX_STEPS", 4)
    client = EndlessSearcher()
    tool = Tool("search", "search", schema({"q": {"type": "string"}}), lambda q: "nothing")
    messages = [{"role": "user", "content": "question"}]
    answer = run_loop(LLM("local", client, "fake"), "sys", [tool], messages, Console())
    assert answer == "Not found in the knowledge base."
    assert len(client.requests) == 4
    final = client.requests[-1]
    assert "tools" not in final and final["messages"][-1]["content"] == FINAL_STEP_NOTE
    assert all(m.get("content") != FINAL_STEP_NOTE for m in messages)  # not kept in history


class SilentThinker:
    """Fake OpenAI client: first reply is empty (only reasoning), the retry answers."""

    def __init__(self):
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kw):
        self.requests.append(kw)
        text = "" if len(self.requests) == 1 else "Not found."
        return FakeStream([NS(usage=None, choices=[NS(delta=NS(content=text or None, tool_calls=None))])])


def test_empty_answer_is_retried_once_without_tools():
    client = SilentThinker()
    tool = Tool("search", "search", schema({"q": {"type": "string"}}), lambda q: "nothing")
    answer = run_loop(LLM("local", client, "fake"), "sys", [tool],
                      [{"role": "user", "content": "q"}], Console())
    assert answer == "Not found."
    assert len(client.requests) == 2
    assert "tools" not in client.requests[1]
    assert client.requests[1]["messages"][-1]["content"] == EMPTY_ANSWER_NOTE
