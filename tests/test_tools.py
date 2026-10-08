import pytest

from core.tools import Tool, call_tool, schema


def make():
    return Tool("add", "sum", schema({"a": {"type": "integer"}, "b": {"type": "integer"}}, ["a"]),
                lambda a, b=0: str(a + b))


def test_formats():
    t = make()
    assert t.anthropic()["input_schema"]["required"] == ["a"]
    assert t.openai()["function"]["parameters"]["properties"]["b"]["type"] == "integer"


def test_call_tool_accepts_dict_and_json_string():
    reg = {"add": make()}
    assert call_tool(reg, "add", {"a": 1, "b": 2}) == "3"
    assert call_tool(reg, "add", '{"a": 5}') == "5"
    with pytest.raises(ValueError):
        call_tool(reg, "nope", {})
