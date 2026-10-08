"""One agent loop (model -> tools -> model) for both providers.

Anthropic and OpenAI-compatible APIs differ only in message format and in how the
stream is assembled; everything else is shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.config import LOCAL_MODEL, LOCAL_TEMPERATURE, LOCAL_URL, MAX_STEPS, MODEL
from core.console import Console
from core.pricing import Usage
from core.tools import Tool, call_tool


# Opus 5.5 always thinks; a small cap would cut answers mid-thought. Streaming avoids timeouts.
ANTHROPIC_MAX_TOKENS = 32000


@dataclass
class LLM:
    provider: str  # "anthropic" | "local"
    client: object
    model: str
    usage: Usage = field(default_factory=Usage)  # accumulated over the LLM's lifetime


def make_llm(provider: str) -> LLM:
    if provider == "local":
        from openai import OpenAI
        return LLM("local", OpenAI(base_url=LOCAL_URL, api_key="ollama"), LOCAL_MODEL)
    import anthropic
    return LLM("anthropic", anthropic.Anthropic(), MODEL)


def run_loop(llm: LLM, system: str, tools: list[Tool], messages: list, out: Console) -> str:
    """Runs until the model answers without tool calls. Mutates `messages` (history)."""
    step = _step_anthropic if llm.provider == "anthropic" else _step_openai
    registry = {t.name: t for t in tools}
    for _ in range(MAX_STEPS):
        answer = step(llm, system, registry, messages, out)
        if answer is not None:
            out.newline()
            return answer
    return f"Agent did not finish within {MAX_STEPS} steps."


def _run_tool(registry: dict[str, Tool], name: str, args, out: Console) -> tuple[str, bool]:
    try:
        output, is_error = call_tool(registry, name, args), False
    except Exception as e:  # the model sees the error and can retry
        output, is_error = f"Error: {e}", True
    out.tool_call(name, args, len(output))
    return output, is_error


def _step_anthropic(llm, system, registry, messages, out) -> str | None:
    with llm.client.messages.stream(
        model=llm.model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=system,
        tools=[t.anthropic() for t in registry.values()],
        messages=messages,
        # each loop step resends the whole history: cache the growing prefix
        cache_control={"type": "ephemeral"},
    ) as stream:
        for text in stream.text_stream:
            out.text(text)
        resp = stream.get_final_message()
    u = resp.usage
    llm.usage.add(llm.model, u.input_tokens, u.output_tokens,
                  u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0)
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason != "tool_use":
        return "".join(b.text for b in resp.content if b.type == "text")

    out.newline()
    results = []
    for block in resp.content:
        if block.type == "tool_use":
            output, is_error = _run_tool(registry, block.name, block.input, out)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output, "is_error": is_error})
    messages.append({"role": "user", "content": results})
    return None


def _step_openai(llm, system, registry, messages, out) -> str | None:
    stream = llm.client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "system", "content": system}, *messages],
        tools=[t.openai() for t in registry.values()],
        temperature=LOCAL_TEMPERATURE,
        top_p=0.95,
        stream=True,
        stream_options={"include_usage": True},  # usage arrives in a final chunk without choices
    )
    # No final-message helper in the OpenAI stream: rebuild it from deltas.
    content, calls, thinking = "", {}, False
    for chunk in stream:
        if chunk.usage:
            llm.usage.add(llm.model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # Thinking models (qwen3.x on Ollama) stream reasoning in a non-standard field.
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            out.thinking(reasoning)
            thinking = True
        if thinking and (delta.content or delta.tool_calls):
            out.thinking("\n\n")
            thinking = False
        if delta.content:
            out.text(delta.content)
            content += delta.content
        for tc in delta.tool_calls or []:
            # arguments come as JSON fragments, joined by call index
            c = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            c["id"] = tc.id or c["id"]
            if tc.function:
                c["name"] += tc.function.name or ""
                c["arguments"] += tc.function.arguments or ""

    if not calls:
        messages.append({"role": "assistant", "content": content})
        return content

    out.newline()
    tool_calls = [
        {"id": c["id"], "type": "function",
         "function": {"name": c["name"], "arguments": c["arguments"]}}
        for _, c in sorted(calls.items())
    ]
    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
    for call in tool_calls:
        fn = call["function"]
        output, _ = _run_tool(registry, fn["name"], fn["arguments"], out)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
    return None
