from langchain_core.messages import AIMessage

from agents.base import Agent
from agents.citations import Citations
from agents.langgraph_agent import LangGraphAgent
from core.console import Console
from core.llm import LLM
from core.tools import Tool, schema


class FakeModel:
    """Scripted replies; records what each call was given."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


def make_agent(cites):
    def lookup(name: str) -> str:
        return f"[{cites.add(f'src/{name}.py:1-5')}] body of {name}"
    tool = Tool("lookup", "Look up a name.", schema({"name": {"type": "string"}}, ["name"]), lookup)
    return Agent("t", "test agent", "system prompt", [tool], cites)


def test_tool_call_then_answer_with_sources(monkeypatch):
    model = FakeModel([
        AIMessage("", tool_calls=[{"name": "lookup", "args": {"name": "foo"}, "id": "c1"}]),
        AIMessage("foo is defined in src [1]."),
    ])
    monkeypatch.setattr("agents.langgraph_agent.chat_model", lambda llm: model)
    history = [{"role": "user", "content": "where is foo?"}]
    answer = LangGraphAgent(make_agent(Citations())).run(LLM("local", None, "m"), history, Console(quiet=True))
    assert answer.startswith("foo is defined in src [1].")
    assert "Sources:\n[1] src/foo.py:1-5" in answer
    assert "body of foo" in model.calls[1][-1].content  # tool output went back to the model
    assert history[-1] == {"role": "assistant", "content": "foo is defined in src [1]."}


def test_empty_reply_is_retried(monkeypatch):
    model = FakeModel([AIMessage(""), AIMessage("not found")])
    monkeypatch.setattr("agents.langgraph_agent.chat_model", lambda llm: model)
    answer = LangGraphAgent(make_agent(Citations())).run(
        LLM("local", None, "m"), [{"role": "user", "content": "q"}], Console(quiet=True))
    assert answer == "not found"
    assert "empty" in model.calls[1][-1].content


def test_last_step_forces_an_answer(monkeypatch):
    from core.config import MAX_STEPS
    # distinct messages: add_messages dedupes by id, a reused object would replace itself
    calls = [AIMessage("", tool_calls=[{"name": "lookup", "args": {"name": "x"}, "id": f"c{i}"}])
             for i in range(MAX_STEPS - 1)]
    model = FakeModel(calls + [AIMessage("gave up: not found")])
    monkeypatch.setattr("agents.langgraph_agent.chat_model", lambda llm: model)
    answer = LangGraphAgent(make_agent(Citations())).run(
        LLM("local", None, "m"), [{"role": "user", "content": "q"}], Console(quiet=True))
    assert answer.startswith("gave up: not found")
    assert len(model.calls) == MAX_STEPS
    assert "Tool budget exhausted" in model.calls[-1][-1].content


def test_mcp_read_file_unknown_path(monkeypatch):
    import pytest
    from mcp.server.mcpserver.exceptions import ToolError

    from mcp_servers import rag

    class Store:
        def source_of(self, path): return None

    class Retriever:
        store = Store()

    monkeypatch.setitem(rag._retrievers, "c", Retriever())
    with pytest.raises(ToolError, match="not indexed"):
        rag.read_file("nope.cs", "c")
