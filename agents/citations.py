"""Numbered citations. Small models skip or mangle long `path:line` refs, but reliably
write `[n]`; the code then maps the numbers back to exact refs, so refs can't be invented."""
from __future__ import annotations

import re

_MARK = re.compile(r"\[(\d+)\]")


class Citations:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.refs: list[str] = []  # id n -> refs[n - 1]

    def add(self, ref: str) -> int:
        if ref not in self.refs:
            self.refs.append(ref)
        return self.refs.index(ref) + 1

    def remap(self, answer: str, target: "Citations") -> str:
        """Rewrite [n] markers into `target`'s numbering (sub-agent -> supervisor)."""
        def sub(m: re.Match) -> str:
            n = int(m.group(1))
            return f"[{target.add(self.refs[n - 1])}]" if 0 < n <= len(self.refs) else ""
        return _MARK.sub(sub, answer)

    def footer(self, answer: str) -> str:
        cited = [int(n) for n in dict.fromkeys(_MARK.findall(answer))
                 if 0 < int(n) <= len(self.refs)]
        if cited:
            return "\n\nSources:\n" + "\n".join(f"[{n}] {self.refs[n - 1]}" for n in cited)
        if self.refs:  # be explicit rather than silently dropping provenance
            return "\n\nRetrieved (not cited):\n" + "\n".join(f"- {r}" for r in self.refs)
        return ""
