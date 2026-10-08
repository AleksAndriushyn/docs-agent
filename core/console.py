import sys
from contextvars import ContextVar

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


class Console:
    """Output sink for an agent run. Sub-agents use quiet=True: their answer goes
    back to the supervisor, only tool-call logs are shown (with -v)."""

    def __init__(self, verbose: bool = False, prefix: str = "", quiet: bool = False):
        self.verbose, self.prefix, self.quiet = verbose, prefix, quiet
        self._midline = False

    def _write(self, s: str) -> None:
        if s:
            print(s, end="", flush=True)
            self._midline = not s.endswith("\n")

    def newline(self) -> None:
        if self._midline:
            self._write("\n")

    def text(self, s: str) -> None:
        if not self.quiet:
            self._write(s)

    def thinking(self, s: str) -> None:
        if self.verbose and not self.quiet:
            self._write(f"{DIM}{s}{RESET}")

    def log(self, s: str) -> None:
        if self.verbose:
            self.newline()
            self._write(f"{DIM}{self.prefix}{s}{RESET}\n")

    def tool_call(self, name: str, args, size: int) -> None:
        self.log(f"-> {name}({args}) [{size} chars]")

    def child(self, name: str) -> "Console":
        """Console for a sub-agent: its answer goes to the caller, only logs are shown."""
        return Console(self.verbose, prefix=f"[{name}] ", quiet=True)


# Console of the agent run in progress; sub-agents called as tools derive theirs from it.
CURRENT: ContextVar[Console | None] = ContextVar("current_console", default=None)
