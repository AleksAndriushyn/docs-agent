from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[..., str]

    def anthropic(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}

    def openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.schema}}


def call_tool(tools: dict[str, Tool], name: str, args: dict | str) -> str:
    if name not in tools:
        raise ValueError(f"unknown tool: {name}")
    if isinstance(args, str):  # OpenAI-style arguments arrive as a JSON string
        args = json.loads(args or "{}")
    return tools[name].fn(**args)


def schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}
