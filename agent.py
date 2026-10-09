"""CLI for the agents.

    python agent.py --root ./docs -q "Which parameters control failsafe?"     # researcher (default)
    python agent.py --agent tester --root ./src --provider local                 # pytest in sandbox
    python agent.py --agent research-lg --root ./docs --provider local           # researcher on LangGraph
    python agent.py --agent supervisor --root ./src --provider local -v          # all of the above
"""
from __future__ import annotations

import argparse

from core.config import RAG_COLLECTION
from core.console import DIM, RESET, Console
from core.llm import make_llm
from core.workspace import Workspace


def build_agent(name: str, llm, ws: Workspace, collection: str, verbose: bool):
    if name == "research":
        from agents.researcher import researcher_agent
        return researcher_agent(ws, collection)
    if name == "research-lg":
        from agents.langgraph_agent import LangGraphAgent
        from agents.researcher import researcher_agent
        return LangGraphAgent(researcher_agent(ws, collection))
    if name == "tester":
        from agents.tester import tester_agent
        return tester_agent(ws)
    from agents.supervisor import supervisor_agent
    return supervisor_agent(llm, ws, collection, verbose)


def print_usage(llm, before=None) -> None:
    u = llm.usage - before if before else llm.usage
    cost = u.cost()
    price = f", ${cost:.4f}" if cost is not None else " (local, free)"
    cache = f", cache read {u.cache_read:,}" if u.cache_read else ""
    print(f"{DIM}[{u.calls} calls · in {u.input:,} · out {u.output:,}{cache}{price}]{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["research", "research-lg", "tester", "supervisor"], default="research")
    ap.add_argument("--root", default=".", help="Folder the agent works in")
    ap.add_argument("--collection", default=RAG_COLLECTION, help="Qdrant collection for rag/supervisor")
    ap.add_argument("-q", "--question", help="Single question, non-interactive")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show tool calls and model reasoning")
    ap.add_argument("--provider", choices=["anthropic", "local"], default="anthropic")
    args = ap.parse_args()

    if args.provider == "anthropic" and args.agent in {"research", "research-lg", "supervisor"}:
        print("Warning: retrieved work-file chunks will be sent to the Anthropic API. "
              "For fully local operation use --provider local.")

    ws = Workspace(args.root)
    llm = make_llm(args.provider)
    agent = build_agent(args.agent, llm, ws, args.collection, args.verbose)
    out = Console(args.verbose)
    messages: list = []

    if args.question:
        messages.append({"role": "user", "content": args.question})
        agent.run(llm, messages, out)  # streamed to the terminal
        print_usage(llm)
        return

    print(f"Agent: {args.agent}, folder: {ws.root}\nAsk a question (empty line to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        messages.append({"role": "user", "content": q})  # history persists between questions
        print()
        before = llm.usage.copy()
        agent.run(llm, messages, out)
        print_usage(llm, before)


if __name__ == "__main__":
    main()
