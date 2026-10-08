"""USD per 1M tokens (Claude API first-party rates, checked 2026-10-08).
Cache writes cost 1.25x input (5-minute TTL). Local models are free."""
from __future__ import annotations

from dataclasses import dataclass, field

PRICES = {  # model: (input, output, cache_read)
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0
    by_model: dict[str, list[int]] = field(default_factory=dict)  # model -> [in, out, cr, cw]

    def add(self, model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0) -> None:
        self.input += inp
        self.output += out
        self.cache_read += cache_read
        self.cache_write += cache_write
        self.calls += 1
        row = self.by_model.setdefault(model, [0, 0, 0, 0])
        for i, v in enumerate((inp, out, cache_read, cache_write)):
            row[i] += v

    def cost(self) -> float | None:
        """None when a model has no known price (e.g. a local one): free, not unknown."""
        total, priced = 0.0, False
        for model, (inp, out, cr, cw) in self.by_model.items():
            if model not in PRICES:
                continue
            p_in, p_out, p_cr = PRICES[model]
            total += (inp * p_in + out * p_out + cr * p_cr + cw * p_in * 1.25) / 1e6
            priced = True
        return total if priced else None

    def snapshot(self) -> dict:
        return {"input": self.input, "output": self.output, "cache_read": self.cache_read,
                "cache_write": self.cache_write, "calls": self.calls, "cost_usd": self.cost()}

    def __sub__(self, other: "Usage") -> "Usage":
        d = Usage()
        for model, row in self.by_model.items():
            prev = other.by_model.get(model, [0, 0, 0, 0])
            d.add(model, *(a - b for a, b in zip(row, prev)))
        d.calls = self.calls - other.calls
        return d

    def copy(self) -> "Usage":
        c = Usage()
        for model, row in self.by_model.items():
            c.add(model, *row)
        c.calls = self.calls
        return c
