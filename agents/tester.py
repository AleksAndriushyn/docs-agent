"""Writes pytest tests for code under --root and runs them in sandbox/<run_id>/.
The model may only write inside the sandbox; every pytest run needs a "y" from the user."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Callable

from agents.base import Agent
from agents.files import workspace_tools
from core.config import SANDBOX_DIR
from core.tools import Tool, schema
from core.workspace import Workspace

TIMEOUT_S = 60
MAX_OUTPUT = 8000

SYSTEM = """You write and run pytest tests for Python code in the working folder.
Workflow:
1. Read the code (read_file, search) and understand the public behaviour.
2. copy_to_sandbox the modules under test (paths are preserved, the sandbox is on
   sys.path, import as usual: `from pkg.mod import f`).
3. write_test: test_*.py files in the sandbox. Cover behaviour, edge cases, errors.
   No network, no writes outside tmp_path.
4. run_pytest. If a test fails, decide whether the test or the code is wrong. Fix tests,
   never the code; report bugs you found.
Finish with: what is covered, what failed and why, the sandbox path.
Answer in the user's language."""


class Sandbox:
    def __init__(self, ws: Workspace, confirm: Callable[[str], bool] | None = None):
        self.ws = ws
        self.confirm = confirm or _confirm
        self.dir = SANDBOX_DIR / time.strftime("%Y%m%d-%H%M%S")
        self.box = None  # created lazily, so an unused tester leaves nothing behind

    def _box(self) -> Workspace:
        if self.box is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.box = Workspace(self.dir)
        return self.box

    def copy_to_sandbox(self, path: str) -> str:
        src = self.ws.resolve(path)
        dst = self._box().resolve(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return f"copied: {path}"

    def write_test(self, path: str, content: str) -> str:
        dst = self._box().resolve(path)
        if not dst.name.startswith(("test_", "conftest")) or dst.suffix != ".py":
            raise ValueError("only test_*.py or conftest.py may be written")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def run_pytest(self, args: str = "") -> str:
        cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
               *shlex.split(args)]
        box = self._box()
        if not self.confirm(f"Run in {box.root}:\n  {shlex.join(cmd[1:])}\nFiles:\n"
                            + box.list_dir()):
            return "The user declined the run."
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(box.root), "PYTHONPATH": str(box.root)}
        try:
            r = subprocess.run(cmd, cwd=box.root, env=env, capture_output=True, text=True,
                               timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"Timed out after {TIMEOUT_S}s: likely an infinite loop or waiting for input."
        out = (r.stdout + r.stderr)[-MAX_OUTPUT:]
        return f"exit code {r.returncode}\n{out}"


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False  # no human to approve
    return input(f"\n{prompt}\n[y/N] ").strip().lower() in {"y", "yes"}


def tester_agent(ws: Workspace, confirm: Callable[[str], bool] | None = None) -> Agent:
    sb = Sandbox(ws, confirm)
    tools = workspace_tools(ws) + [
        Tool("copy_to_sandbox", "Copy a file or folder from the working folder into the sandbox.",
             schema({"path": {"type": "string"}}, ["path"]), sb.copy_to_sandbox),
        Tool("write_test", "Write a test file (test_*.py or conftest.py) into the sandbox.",
             schema({"path": {"type": "string"}, "content": {"type": "string"}},
                    ["path", "content"]), sb.write_test),
        Tool("run_pytest", "Run pytest in the sandbox (requires user approval). "
                           "Returns the exit code and output.",
             schema({"args": {"type": "string", "description": "Extra arguments, e.g. "
                                                               "'test_x.py -k name'"}}),
             sb.run_pytest),
    ]
    return Agent(
        name="tester",
        description="Writes pytest tests for Python code in the working folder and runs them "
                    "in a sandbox. Give it the module path and what to check.",
        system=SYSTEM,
        tools=tools,
    )
