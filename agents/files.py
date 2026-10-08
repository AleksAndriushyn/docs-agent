from agents.base import Agent
from core.tools import Tool, schema
from core.workspace import Workspace

SYSTEM = """You answer questions about the files in the working folder.
- Explore the files with tools first, then answer. Never invent file contents.
- Use search for content lookup and list_dir for structure.
- Cite the files and lines you rely on (path:line).
- If the files don't contain the answer, say so.
- Answer in the user's language."""


def workspace_tools(ws: Workspace) -> list[Tool]:
    return [
        Tool("list_dir", "List files and folders at a relative path inside the working folder.",
             schema({"path": {"type": "string", "description": "E.g. '.' or 'src/'"}}),
             ws.list_dir),
        Tool("read_file", "Read a text file with line numbers. Long files are truncated, "
                          "continue with start_line.",
             schema({"path": {"type": "string"},
                     "start_line": {"type": "integer", "minimum": 1},
                     "end_line": {"type": "integer", "minimum": 1}}, ["path"]),
             ws.read_file),
        Tool("search", "Case-insensitive regex search over file contents. "
                       "Returns path:line: text.",
             schema({"pattern": {"type": "string"},
                     "glob": {"type": "string", "description": "File filter, e.g. '**/*.md'"}},
                    ["pattern"]),
             ws.search),
    ]


def files_agent(ws: Workspace) -> Agent:
    return Agent(
        name="files",
        description="Reads and regex-searches files in the working folder directly. Best for exact "
                    "names, paths, folder structure and reading a specific file.",
        system=SYSTEM,
        tools=workspace_tools(ws),
    )
