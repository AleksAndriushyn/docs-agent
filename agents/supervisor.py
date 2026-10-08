from __future__ import annotations

from typing import Callable

from agents.base import Agent
from agents.citations import Citations
from agents.researcher import researcher_agent
from agents.tester import tester_agent
from core.llm import LLM
from core.workspace import Workspace
from rag.store import Store

SYSTEM = """You are a coordinator. You don't read files yourself: you delegate to agent tools
and assemble the final answer.
- ask_researcher: any question about the work files (code and docs): it searches the
  knowledge base by meaning and the working folder by regex, and reads files.
- ask_tester: write and run tests for a Python module.
Simple questions need one ask_researcher call; split only genuinely separate subtasks.
Rules:
- Agents don't see the conversation: make each task self-contained, with paths and names.
- Independent subtasks can go to different agents. Don't repeat the same task.
- Agent answers contain source markers like [4]. Keep each marker next to the claim it
  supports, exactly as given. Never invent markers or file paths; the source list is
  appended automatically. Don't add facts the agents didn't report.
- Answer in the user's language."""


def supervisor_agent(llm: LLM, ws: Workspace, collection: str, verbose: bool,
                     confirm: Callable[[str], bool] | None = None,
                     store: Store | None = None) -> Agent:
    subagents = [researcher_agent(ws, collection, store), tester_agent(ws, confirm)]
    cites = Citations()
    return Agent(
        name="supervisor",
        description="Coordinator",
        system=SYSTEM,
        tools=[a.as_tool(llm, verbose, cites) for a in subagents],
        citations=cites,
    )
