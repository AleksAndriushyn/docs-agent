export type Progress = { done: number; total: number; current: string };

export type Source = {
  id: string;
  name: string;
  path: string;
  collection: string;
  include: string[];
  exclude: string[];
  status: "new" | "queued" | "indexing" | "ready" | "cancelled" | "error";
  files: number;
  chunks: number;
  last_indexed: number | null;
  error: string;
  progress: Progress | null;
};

export type Collection = { name: string; points: number; sources: number };

export type Status = {
  model: string;
  embed_model: string;
  collection: string;
  points: number;
  ollama: boolean;
  qdrant: boolean;
  qdrant_mode: "server" | "embedded";
  running_chat: string | null;
};

export type Step = { agent: string; name?: string; args?: unknown; chars?: number; text?: string };
export type Approval = { id: string; prompt: string; state: "pending" | "approved" | "declined" };

export type Turn = {
  question: string;
  answer: string;
  thinking: string;
  steps: Step[];
  approvals: Approval[];
  error: string | null;
  status: "running" | "done" | "error" | "stopped";
  started: number;  // unix seconds
  finished: number | null;
};

export type ChatSummary = {
  id: string;
  collection: string;
  root: string;
  title: string;
  created: number;
  updated: number;
  turns: number;
  running: boolean;
};

/** `offset`: index in the live run's events where a stream continues after `items`. */
export type ChatFull = ChatSummary & { items: Turn[]; offset: number };

export type ChatEvent =
  | { type: "text"; text: string }
  | { type: "thinking"; agent: string; text: string }
  | { type: "log"; agent: string; text: string }
  | { type: "tool"; agent: string; name: string; args: unknown; chars: number }
  | { type: "approval"; id: string; prompt: string }
  | { type: "approval_done"; id: string; approved: boolean }
  | { type: "done" }
  | { type: "stopped"; message?: string }
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
  status: (collection: string) => call<Status>("GET", `/api/status?collection=${encodeURIComponent(collection)}`),
  collections: () => call<Collection[]>("GET", "/api/collections"),
  createCollection: (name: string) => call<Collection>("POST", "/api/collections", { name }),
  deleteCollection: (name: string) => call("DELETE", `/api/collections/${encodeURIComponent(name)}`),
  sources: (collection: string) =>
    call<Source[]>("GET", `/api/sources?collection=${encodeURIComponent(collection)}`),
  addSource: (path: string, collection: string) => call<Source>("POST", "/api/sources", { path, collection }),
  editSource: (id: string, include: string[], exclude: string[]) =>
    call<Source>("PATCH", `/api/sources/${id}`, { include, exclude }),
  reindex: (id: string) => call<Source>("POST", `/api/sources/${id}/reindex`),
  cancel: (id: string) => call<Source>("POST", `/api/sources/${id}/cancel`),
  removeSource: (id: string) => call("DELETE", `/api/sources/${id}`),
  chats: (collection: string, root: string) =>
    call<ChatSummary[]>("GET", `/api/chats?collection=${encodeURIComponent(collection)}&root=${encodeURIComponent(root)}`),
  chat: (id: string) => call<ChatFull>("GET", `/api/chats/${id}`),
  createChat: (collection: string, root: string) => call<ChatFull>("POST", "/api/chats", { collection, root }),
  deleteChat: (id: string) => call("DELETE", `/api/chats/${id}`),
  ask: (id: string, message: string) => call<ChatFull>("POST", `/api/chats/${id}/messages`, { message }),
  stop: (id: string) => call("POST", `/api/chats/${id}/stop`),
  approve: (chat: string, id: string, approved: boolean) =>
    call("POST", `/api/chats/${chat}/approve`, { id, approved }),
  openRef: (ref: string) => call("POST", "/api/open", { ref }),
};

/** Server-Sent Events of a chat's live run, from event index `after`; ends with the run.
 * fetch rather than EventSource: EventSource reconnects on its own and would replay events. */
export async function streamChat(id: string, after: number, onEvent: (e: ChatEvent) => void,
                                 signal: AbortSignal) {
  const res = await fetch(`/api/chats/${id}/events?after=${after}`, { signal });
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
      if (frame.startsWith("data: ")) onEvent(JSON.parse(frame.slice(6)));  // ": ping" frames are skipped
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
