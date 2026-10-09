"""The same agent on LangGraph instead of core.llm.run_loop, for a side-by-side comparison.

Prompt, tools, citations and tool execution are shared with the hand-written loop, so evals
measure the framework, not other differences. The graph is the loop made explicit:

    START -> model --(tool calls)--> tools -> model ... --(text)--> END

Kept from run_loop: the last allowed step answers without tools, an empty reply is asked once more.
"""
from __future__ import annotations

import operator
from typing import Annotated, cast

from langchain_core.messages import (AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage,
                                     ToolMessage, convert_to_messages)
from langgraph.graph import END, START, MessagesState, StateGraph

from agents.base import Agent
from core.config import LOCAL_MODEL, LOCAL_TEMPERATURE, LOCAL_URL, MAX_STEPS
from core.console import CURRENT, Console
from core.llm import ANTHROPIC_MAX_TOKENS, EMPTY_ANSWER_NOTE, FINAL_STEP_NOTE, LLM, _run_tool


class State(MessagesState):
    steps: Annotated[int, operator.add]  # model calls so far


def last_ai(state: State) -> AIMessage:
    """The graph only routes on the model's reply, which is always an AIMessage."""
    msg = state["messages"][-1]
    assert isinstance(msg, AIMessage), type(msg)
    return msg


def chat_model(llm: LLM):
    if llm.provider == "local":
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr
        return ChatOpenAI(model=LOCAL_MODEL, base_url=LOCAL_URL, api_key=SecretStr("ollama"),
                          temperature=LOCAL_TEMPERATURE, top_p=0.95,
                          streaming=True, stream_usage=True)
    from langchain_anthropic import ChatAnthropic
    # model_validate, not the constructor: pyright takes pydantic aliases (model_name,
    # max_tokens_to_sample) for the only parameter names; at runtime both names work
    return ChatAnthropic.model_validate({
        "model": llm.model, "max_tokens": ANTHROPIC_MAX_TOKENS, "streaming": True,
        # same automatic prompt caching as the hand-written loop
        "model_kwargs": {"cache_control": {"type": "ephemeral"}},
    })


def record_usage(llm: LLM, msg: AIMessage) -> None:
    u = msg.usage_metadata
    if not u:
        return
    details = u.get("input_token_details") or {}
    read, write = details.get("cache_read") or 0, details.get("cache_creation") or 0
    if llm.provider == "local":  # run_loop counts Ollama's prompt cache as plain input
        read = write = 0
    # LangChain counts cached tokens into input_tokens; Usage keeps them apart
    llm.usage.add(llm.model, u["input_tokens"] - read - write, u["output_tokens"], read, write)


class LangGraphAgent:
    def __init__(self, agent: Agent):
        self.agent = agent

    def _graph(self, llm: LLM, out: Console):
        model = chat_model(llm)
        schemas = [t.openai() for t in self.agent.tools]
        with_tools = model.bind_tools(schemas)
        # Anthropic needs tools defined while history has tool calls; Ollama ignores tool_choice
        final = model.bind_tools(schemas, tool_choice="none") if llm.provider == "anthropic" else model
        registry = {t.name: t for t in self.agent.tools}
        system = SystemMessage(self.agent.system)

        def call_model(state: State) -> dict:
            last = state["steps"] >= MAX_STEPS - 1
            history = [system, *state["messages"]]
            resp = (final if last else with_tools).invoke(
                history + [HumanMessage(FINAL_STEP_NOTE)] if last else history)
            record_usage(llm, resp)
            if not resp.tool_calls and not resp.text.strip():
                resp = final.invoke(history + [resp, HumanMessage(EMPTY_ANSWER_NOTE)])
                record_usage(llm, resp)
            return {"messages": [resp], "steps": 1}

        def call_tools(state: State) -> dict:
            out.newline()
            results = []
            for call in last_ai(state).tool_calls:
                output, is_error = _run_tool(registry, call["name"], call["args"], out)
                results.append(ToolMessage(output, tool_call_id=call["id"],
                                           status="error" if is_error else "success"))
            return {"messages": results}

        def route(state: State) -> str:
            return "tools" if last_ai(state).tool_calls else END

        g = StateGraph(State)
        g.add_node("model", call_model)
        g.add_node("tools", call_tools)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", route, ["tools", END])
        g.add_edge("tools", "model")
        return g.compile()

    def run(self, llm: LLM, messages: list, out: Console) -> str:
        """Same contract as Agent.run: streams to `out`, returns the answer with sources."""
        if self.agent.citations:
            self.agent.citations.reset()
        token = CURRENT.set(out)
        try:
            graph = self._graph(llm, out)
            state = State(messages=cast("list[AnyMessage]", convert_to_messages(messages)), steps=0)
            final_state: State | None = None
            # "messages" streams model tokens, "values" yields the state after each step
            for mode, data in graph.stream(state, stream_mode=["messages", "values"],
                                           config={"recursion_limit": 2 * MAX_STEPS + 2}):
                # with several modes the payload type depends on `mode`, which pyright can't follow
                if mode == "messages":
                    chunk, meta = cast("tuple[BaseMessage, dict]", data)
                    if meta.get("langgraph_node") == "model" and isinstance(chunk, AIMessage):
                        out.text(chunk.text)
                else:
                    final_state = cast(State, data)
        finally:
            CURRENT.reset(token)
        out.newline()
        answer = final_state["messages"][-1].text if final_state else ""
        messages.append({"role": "assistant", "content": answer})  # text-only history between questions
        if self.agent.citations:
            extra = self.agent.citations.footer(answer)
            out.text(extra)
            out.newline()
            answer += extra
        return answer
