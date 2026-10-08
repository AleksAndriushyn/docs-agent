"""Deterministic check next to the LLM judge: the hard tokens of each reference fact
(parameter names, code identifiers, numbers, file extensions) must appear in the answer.

A local judge grading its own model tends to be lenient; a missing `THR_FS_VALUE` or `950`
can't be argued away. Facts with no hard tokens ("prefer no pulses") are left to the judge.
"""
from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"\b[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b"),  # THR_FS_VALUE, SERVOx_FUNCTION
    re.compile(r"\b[A-Z]{3,}[0-9]*\b"),                      # RTL, CIRCLE, MAVFTP
    re.compile(r"\b[A-Za-z]*[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b"),  # getParamList, FDMenuMap (not "RC")
    re.compile(r"\b[A-Za-z_]\w*(?=\(\))"),                    # Init(), Exit()
    re.compile(r"(?<![\w.])\.(?:dll|cs|py|rst|cpp|h)\b"),    # .dll, .cs
    re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])"),          # 950, 0.5
]
# all-caps words that are prose, not identifiers
_STOP = {"AND", "THE", "FOR", "NOT", "GCS", "API", "USB", "ON", "OFF"}


def terms(fact: str) -> list[str]:
    found: list[str] = []
    for rx in _PATTERNS:
        for m in rx.findall(fact):
            if m not in _STOP and m not in found:
                found.append(m)
    return found


def _normalize(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", ".", text)  # "0,5" in Russian prose is 0.5


def present(term: str, answer: str) -> bool:
    text = _normalize(answer)
    if re.fullmatch(r"\d+(?:\.\d+)?", term):
        return re.search(rf"(?<![\d.]){re.escape(term)}(?![\d]|\.\d)", text) is not None
    # "SERVOx_FUNCTION": x stands for a channel number, so SERVO5_FUNCTION also counts
    pattern = re.sub(r"(?<=[A-Z])x(?=_)", r"(?:\\d+|x)", re.escape(term))
    return re.search(rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])", text, re.IGNORECASE) is not None


def check(facts: list[str], answer: str) -> list[dict]:
    """Per fact: its hard terms and which of them are missing from the answer."""
    out = []
    for fact in facts:
        ts = terms(fact)
        out.append({"fact": fact, "terms": ts, "missing": [t for t in ts if not present(t, answer)]})
    return out


def score(facts: list[str], answer: str) -> dict:
    """Summary for one answer: hard terms found / total, plus the per-fact detail."""
    per_fact = check(facts, answer)
    total = sum(len(f["terms"]) for f in per_fact)
    found = total - sum(len(f["missing"]) for f in per_fact)
    return {"found": found, "total": total, "facts": per_fact}
