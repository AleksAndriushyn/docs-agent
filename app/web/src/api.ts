export type Progress = { done: number; total: number; current: string };

export type Source = {
  id: string;
  name: string;
  path: string;
  include: string[];
  exclude: string[];
  status: "new" | "queued" | "indexing" | "ready" | "cancelled" | "error";
  files: number;
  chunks: number;
  last_indexed: number | null;
  error: string;
  progress: Progress | null;
};

export type Status = {
  model: string;
  embed_model: string;
  collection: string;
  points: number;
  ollama: boolean;
  qdrant: boolean;
  qdrant_mode: "server" | "embedded";
};

export type ChatEvent =
  | { type: "text"; text: string }
  | { type: "thinking"; agent: string; text: string }
  | { type: "log"; agent: string; text: string }
  | { type: "tool"; agent: string; name: string; args: unknown; chars: number }
  | { type: "approval"; id: string; prompt: string }
  | { type: "approval_done"; id: string; approved: boolean }
  | { type: "done" }
  | { type: "error"; message: string };

declare global {
  interface Window {
    pywebview?: { api: { pick_folder(): Promise<string[]>; pick_files(): Promise<string[]> } };
  }
}

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail).catch(() => res.statusText);
    throw new Error(String(detail));
  }
  return res.json();
}

export const api = {
  status: () => call<Status>("GET", "/api/status"),
  sources: () => call<Source[]>("GET", "/api/sources"),
  addSource: (path: string) => call<Source>("POST", "/api/sources", { path }),
  editSource: (id: string, include: string[], exclude: string[]) =>
    call<Source>("PATCH", `/api/sources/${id}`, { include, exclude }),
  reindex: (id: string) => call<Source>("POST", `/api/sources/${id}/reindex`),
  cancel: (id: string) => call<Source>("POST", `/api/sources/${id}/cancel`),
  removeSource: (id: string) => call("DELETE", `/api/sources/${id}`),
  resetChat: () => call("POST", "/api/chat/reset"),
  approve: (id: string, approved: boolean) => call("POST", "/api/approve", { id, approved }),
  openRef: (ref: string) => call("POST", "/api/open", { ref }),
};

/** POST + Server-Sent Events (EventSource can't send a body). */
export async function streamChat(message: string, root: string, onEvent: (e: ChatEvent) => void) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, root }),
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().then((j) => j.detail).catch(() => res.statusText);
    onEvent({ type: "error", message: String(detail) });
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      if (frame.startsWith("data: ")) onEvent(JSON.parse(frame.slice(6)));
    }
  }
}

/** Native pickers in the desktop window; a path prompt in a plain browser. */
export async function pickPaths(kind: "folder" | "files"): Promise<string[]> {
  if (window.pywebview) {
    return kind === "folder" ? window.pywebview.api.pick_folder() : window.pywebview.api.pick_files();
  }
  const path = window.prompt(kind === "folder" ? "Folder path" : "File path");
  return path ? [path] : [];
}
