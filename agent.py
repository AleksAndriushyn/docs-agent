"""CLI for the agents.

    python agent.py --root ./docs -q "Which parameters control failsafe?"     # files (default)
    python agent.py --agent rag --provider local -v                              # vector DB
    python agent.py --agent tester --root ./src --provider local                 # pytest in sandbox
    python agent.py --agent supervisor --root ./src --provider local -v          # all of the above
"""
from __future__ import annotations

import argparse

from core.config import RAG_COLLECTION
from core.console import Console
from core.llm import make_llm
from core.workspace import Workspace


def build_agent(name: str, llm, ws: Workspace, collection: str, verbose: bool):
    # imports are lazy so the files agent doesn't load qdrant/fastembed
    if name == "files":
        from agents.files import files_agent
        return files_agent(ws)
    if name == "rag":
        from agents.retriever import retriever_agent
        return retriever_agent(collection)
    if name == "tester":
        from agents.tester import tester_agent
        return tester_agent(ws)
    from agents.supervisor import supervisor_agent
    return supervisor_agent(llm, ws, collection, verbose)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["files", "rag", "tester", "supervisor"], default="files")
    ap.add_argument("--root", default=".", help="Folder the agent works in")
    ap.add_argument("--collection", default=RAG_COLLECTION, help="Qdrant collection for rag/supervisor")
    ap.add_argument("-q", "--question", help="Single question, non-interactive")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show tool calls and model reasoning")
    ap.add_argument("--provider", choices=["anthropic", "local"], default="anthropic")
    args = ap.parse_args()

    if args.provider == "anthropic" and args.agent in {"rag", "supervisor"}:
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
        agent.run(llm, messages, out)


if __name__ == "__main__":
    main()
