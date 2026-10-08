"""LLM-as-judge: grades one answer against the reference and its key facts.

The judge never sees retrieved chunks, only the question, the reference and the answer:
it measures correctness against ground truth, not faithfulness to whatever was retrieved.
"""
from __future__ import annotations

import json

from core.config import LOCAL_MODEL, LOCAL_URL

CLAUDE_JUDGE = "claude-opus-5-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"fact": {"type": "string"}, "present": {"type": "boolean"}},
                "required": ["fact", "present"],
                "additionalProperties": False,
            },
        },
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "refused": {"type": "boolean"},
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "comment": {"type": "string"},
    },
    "required": ["facts", "contradictions", "refused", "verdict", "comment"],
    "additionalProperties": False,
}

PROMPT = """You grade an AI assistant's answer to a question about technical documentation or code.
Use only the reference below as ground truth. The answer may be in another language than
the reference; judge meaning, not wording. Parameter names, numbers and identifiers must match.

Question:
{question}

Reference answer:
{reference}

Key facts the answer should contain:
{facts}

Answer to grade:
<answer>
{answer}
</answer>

Return:
- facts: for each key fact (same order, copied verbatim), whether the answer states it.
  Paraphrases count; a wrong value or name for that fact does not.
- contradictions: claims in the answer that contradict the reference, or names/values that
  look invented (e.g. a parameter the reference names differently). Empty if none.
  Extra correct-sounding details that are merely absent from the reference are not contradictions.
- refused: true if the answer says the information was not found / is not in the knowledge base.
- verdict: {verdict_rule}
- comment: one short sentence."""

RULE_ANSWERABLE = ('"correct" if all key facts are present and there are no contradictions; '
                   '"partial" if some facts are missing but nothing contradicts the reference; '
                   'otherwise "incorrect".')
RULE_UNANSWERABLE = ('the reference says the knowledge base has no answer. "correct" if the answer '
                     'says so and does not invent an answer; "partial" if it says so but still '
                     'speculates; "incorrect" if it presents an answer as fact.')


# The judge prompt is a few thousand tokens; a small window keeps a second model's KV cache small.
JUDGE_NUM_CTX = 8192


class Judge:
    def __init__(self, kind: str = "local", model: str | None = None):
        self.kind = kind  # "claude" only on request: it needs paid API credits
        if kind == "claude":
            import anthropic
            self.client, self.model = anthropic.Anthropic(), CLAUDE_JUDGE
        elif kind == "claude-code":
            self.client, self.model = None, model or "opus"
        else:
            from openai import OpenAI
            self.client, self.model = OpenAI(base_url=LOCAL_URL, api_key="ollama"), model or LOCAL_MODEL
        self.usage = {"input": 0, "output": 0}

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.model}"

    def grade(self, case: dict, answer: str) -> dict:
        prompt = PROMPT.format(
            question=case["question"],
            reference=case["reference"],
            facts="\n".join(f"- {f}" for f in case["facts"]) or "(none: the question is unanswerable)",
            answer=answer or "(empty answer)",
            verdict_rule=RULE_ANSWERABLE if case["answerable"] else RULE_UNANSWERABLE,
        )
        if self.kind == "claude":
            text = self._claude(prompt)
        elif self.kind == "claude-code":
            return self._claude_code(prompt)
        elif self.model == LOCAL_MODEL:
            # same model as the agent: reuse its loaded instance (another num_ctx would reload it)
            text = self._local(prompt)
        else:
            text = self._ollama(prompt)
        return json.loads(text)

    def _claude(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        self.usage["input"] += resp.usage.input_tokens
        self.usage["output"] += resp.usage.output_tokens
        # output_config.format guarantees the text block is valid JSON
        return next(b.text for b in resp.content if b.type == "text")

    def _local(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "grade", "schema": SCHEMA, "strict": True}},
        )
        if resp.usage:
            self.usage["input"] += resp.usage.prompt_tokens
            self.usage["output"] += resp.usage.completion_tokens
        return resp.choices[0].message.content

    def _ollama(self, prompt: str) -> str:
        """Native Ollama API: unlike the OpenAI-compatible one, it accepts num_ctx."""
        import httpx

        base = LOCAL_URL.removesuffix("/").removesuffix("/v1")
        resp = httpx.post(f"{base}/api/chat", timeout=600, json={
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": SCHEMA,
            "stream": False,
            # temperature 0: grading should be repeatable; num_predict caps a runaway generation
            "options": {"num_ctx": JUDGE_NUM_CTX, "temperature": 0, "num_predict": 2048},
            "think": False,  # the verdict schema is the whole output; reasoning only costs time
        })
        resp.raise_for_status()
        data = resp.json()
        self.usage["input"] += data.get("prompt_eval_count", 0)
        self.usage["output"] += data.get("eval_count", 0)
        return data["message"]["content"]

    def _claude_code(self, prompt: str) -> dict:
        """Claude Code in headless mode: runs on the Claude subscription, not on API credits."""
        import os
        import subprocess
        import tempfile

        # an API key in the environment (loaded from .env) would take precedence over the login
        env = {k: v for k, v in os.environ.items()
               if k not in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}}
        cmd = ["claude", "-p", prompt, "--model", self.model, "--output-format", "json",
               "--json-schema", json.dumps(SCHEMA), "--tools", "", "--no-session-persistence",
               "--strict-mcp-config", "--setting-sources", ""]
        with tempfile.TemporaryDirectory() as cwd:  # no project files or CLAUDE.md in context
            r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"claude exited {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
        data = json.loads(r.stdout)
        if data.get("is_error") or not isinstance(data.get("structured_output"), dict):
            raise RuntimeError(f"no structured output: {str(data.get('result'))[:200]}")
        for m in (data.get("modelUsage") or {}).values():
            self.usage["input"] += m.get("inputTokens", 0) + m.get("cacheReadInputTokens", 0)
            self.usage["output"] += m.get("outputTokens", 0)
        return data["structured_output"]
