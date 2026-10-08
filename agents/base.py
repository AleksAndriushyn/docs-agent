from __future__ import annotations

from dataclasses import dataclass, field

from agents.citations import Citations

from core.console import CURRENT, Console
from core.llm import LLM, run_loop
from core.tools import Tool, schema


@dataclass
class Agent:
    name: str
    description: str  # shown to the supervisor when the agent is used as a tool
    system: str
    tools: list[Tool] = field(default_factory=list)
    citations: Citations | None = None  # set when tools emit numbered sources

    def _answer(self, llm: LLM, messages: list, out: Console) -> str:
        if self.citations:
            self.citations.reset()
        token = CURRENT.set(out)
        try:
            return run_loop(llm, self.system, self.tools, messages, out)
        finally:
            CURRENT.reset(token)

    def run(self, llm: LLM, messages: list, out: Console) -> str:
        answer = self._answer(llm, messages, out)
        if self.citations:
            # appended after streaming: a caller (supervisor) gets exact refs too
            extra = self.citations.footer(answer)
            out.text(extra)
            out.newline()
            answer += extra
        return answer

    def as_tool(self, llm: LLM, verbose: bool, cites: Citations | None = None) -> Tool:
        """Each call is a fresh conversation: raw tool output stays out of the caller's context.
        With `cites`, [n] markers are renumbered into the caller's registry instead of a footer."""
        def call(task: str) -> str:
            parent = CURRENT.get()
            out = parent.child(self.name) if parent else Console(verbose, f"[{self.name}] ", quiet=True)
            out.log(f"task: {task}")
            messages = [{"role": "user", "content": task}]
            if cites is not None and self.citations:
                return self.citations.remap(self._answer(llm, messages, out), cites)
            return self.run(llm, messages, out)

        return Tool(
            name=f"ask_{self.name}",
            description=self.description,
            schema=schema({"task": {"type": "string",
                                    "description": "Self-contained task with all needed context"}},
                          ["task"]),
            fn=call,
        )
