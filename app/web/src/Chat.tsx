import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  api, streamChat, type Approval, type ChatEvent, type ChatFull, type ChatSummary, type Source, type Turn,
} from "./api";

const FOOTER = /\n*(Sources|Retrieved \(not cited\)):\n([\s\S]*)$/;
const STICK_PX = 40;  // this close to the bottom counts as "at the bottom"
const LAST_CHAT_KEY = "docs-agent.chat.";  // + collection|root: the chat to reopen for a folder

function remembered(key: string): string | null {
  try {
    return localStorage.getItem(LAST_CHAT_KEY + key);
  } catch {
    return null;
  }
}

function remember(key: string, id: string | null) {
  try {
    if (id) localStorage.setItem(LAST_CHAT_KEY + key, id);
    else localStorage.removeItem(LAST_CHAT_KEY + key);
  } catch {
    /* storage unavailable */
  }
}

/** Mirrors apply_event in app/chat.py: the server folds the same events into saved turns. */
function applyEvent(t: Turn, e: ChatEvent): Turn {
  switch (e.type) {
    case "text": return { ...t, answer: t.answer + e.text };
    case "thinking": return { ...t, thinking: t.thinking + e.text };
    case "log": return { ...t, steps: [...t.steps, { agent: e.agent, text: e.text }] };
    case "tool": return { ...t, steps: [...t.steps, { agent: e.agent, name: e.name, args: e.args, chars: e.chars }] };
    case "approval":
      return { ...t, approvals: [...t.approvals, { id: e.id, prompt: e.prompt, state: "pending" }] };
    case "approval_done":
      return {
        ...t,
        approvals: t.approvals.map((a) => a.id === e.id ? { ...a, state: e.approved ? "approved" : "declined" } : a),
      };
    case "done":
    case "stopped":
    case "error":
      return { ...t, status: e.type, error: "message" in e ? e.message ?? null : null, finished: Date.now() / 1000 };
  }
}

export function Chat({ root, collection, sources }: { root: string; collection: string; sources: Source[] }) {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [notice, setNotice] = useState("");
  const [stopping, setStopping] = useState(false);
  const stream = useRef<AbortController | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);  // follow new output only while the view is at the bottom
  const folderKey = `${collection}|${root}`;
  const rootName = sources.find((s) => s.path === root)?.name;
  const ready = !!rootName;  // root belongs to this collection's loaded sources
  const running = turns.some((t) => t.status === "running");

  useEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  function onScroll() {
    const el = box.current;
    // growing content doesn't fire scroll events, so only the user's own scrolling changes this
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_PX;
  }

  // a folder's chats; reopen the one used last, or the most recent
  useEffect(() => {
    detach();
    setTurns([]);
    setChatId(null);
    setChats([]);
    if (!ready) return;
    let stale = false;
    api.chats(collection, root).then((list) => {
      if (stale) return;
      setChats(list);
      const last = remembered(folderKey);
      const pick = list.find((c) => c.id === last) ?? list[0];
      if (pick) open(pick.id);
    }).catch((e) => setNotice(String(e.message ?? e)));
    return () => { stale = true; detach(); };
  }, [folderKey, ready]);

  function detach() {
    stream.current?.abort();
    stream.current = null;
  }

  function show(chat: ChatFull) {
    stick.current = true;  // an opened chat or a new question starts at the bottom
    setChatId(chat.id);
    setTurns(chat.items);
    remember(folderKey, chat.id);
    setChats((cs) => [chat, ...cs.filter((c) => c.id !== chat.id)].sort((a, b) => b.updated - a.updated));
    if (chat.running) attach(chat.id, chat.offset);
  }

  async function open(id: string) {
    detach();
    setNotice("");
    try {
      show(await api.chat(id));
    } catch (e) {
      setNotice(String((e as Error).message));
    }
  }

  function attach(id: string, after: number) {
    detach();
    const ctl = new AbortController();
    stream.current = ctl;
    const update = (e: ChatEvent) => {
      setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? applyEvent(t, e) : t)));
      if (e.type === "done" || e.type === "stopped" || e.type === "error") {
        setStopping(false);
        setChats((cs) => cs.map((c) => c.id === id ? { ...c, running: false, updated: Date.now() / 1000 } : c));
      }
    };
    streamChat(id, after, update, ctl.signal).catch(() => { /* aborted: switched chat or folder */ });
  }

  function newChat() {
    detach();
    setChatId(null);
    setTurns([]);
    setNotice("");
    remember(folderKey, null);
  }

  async function send() {
    const question = input.trim();
    if (!question || running || !ready) return;
    setNotice("");
    try {
      const id = chatId ?? (await api.createChat(collection, root)).id;
      const chat = await api.ask(id, question);
      setInput("");
      show(chat);
    } catch (e) {
      setNotice(String((e as Error).message));
    }
  }

  async function stop() {
    if (!chatId) return;
    setStopping(true);
    await api.stop(chatId).catch((e) => setNotice(String(e.message)));
  }

  async function remove() {
    if (!chatId || running || !window.confirm("Delete this chat?")) return;
    await api.deleteChat(chatId);
    setChats((cs) => cs.filter((c) => c.id !== chatId));
    newChat();
  }

  return (
    <main className="chat">
      <header className="chat-head">
        <div>
          <span className="muted">Knowledge base: </span><strong>{collection}</strong>
          <span className="muted"> · working folder: </span>
          <strong>{rootName ?? "add a source on the left"}</strong>
        </div>
        <div className="row">
          <select value={chatId ?? ""} disabled={!chats.length}
            onChange={(e) => e.target.value ? open(e.target.value) : newChat()}>
            <option value="">{chats.length ? `History (${chats.length})` : "No saved chats"}</option>
            {chats.map((c) => (
              <option key={c.id} value={c.id}>
                {c.running ? "● " : ""}{c.title || "Untitled"} · {formatDate(c.updated)}
              </option>
            ))}
          </select>
          <button onClick={remove} disabled={!chatId || running} title="Delete this chat">Delete</button>
          <button onClick={newChat} disabled={!chatId}>New chat</button>
        </div>
      </header>

      <div className="messages" ref={box} onScroll={onScroll}>
        {turns.length === 0 && (
          <div className="welcome muted">
            <p>Ask about your indexed files. The supervisor delegates to agents:</p>
            <ul>
              <li><b>researcher</b> searches the knowledge base by meaning, greps and reads the working folder</li>
              <li><b>tester</b> writes and runs pytest tests (asks before running)</li>
            </ul>
            <p>Chats are saved per working folder.</p>
          </div>
        )}
        {turns.map((t, i) => <TurnView key={i} turn={t} chatId={chatId} />)}
      </div>

      <footer className="composer">
        {notice && <div className="error small">{notice}</div>}
        <div className="composer-row">
          <textarea
            value={input}
            placeholder={ready ? "Ask a question… (Enter to send, Shift+Enter for a new line)" : "Pick a working folder first"}
            disabled={!ready}
            rows={3}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
          />
          {running ? (
            <button className="danger-solid" onClick={stop} disabled={stopping}>{stopping ? "Stopping…" : "Stop"}</button>
          ) : (
            <button className="primary" onClick={send} disabled={!input.trim() || !ready}>Send</button>
          )}
        </div>
      </footer>
    </main>
  );
}

/** Seconds since `since`, ticking while mounted. */
function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, []);
  return <>{formatDuration(now - since)}</>;
}

function lastActivity(t: Turn): string {
  const pending = t.approvals.find((a) => a.state === "pending");
  if (pending) return "waiting for your approval";
  if (t.answer) return "writing the answer";
  const step = [...t.steps].reverse().find((s) => s.name);
  if (step) return `${step.agent} → ${step.name}`;
  return t.thinking ? "reasoning" : "waiting for the model";
}

function TurnView({ turn: t, chatId }: { turn: Turn; chatId: string | null }) {
  const isRunning = t.status === "running";
  const m = t.answer.match(FOOTER);
  const body = m ? t.answer.slice(0, m.index) : t.answer;
  const refs = m ? parseRefs(m[2]) : [];
  const cited = m?.[1] === "Sources";
  // [n] markers become links the renderer turns into chips
  const markdown = body.replace(/\[(\d+)\](?!\()/g, "[$1](#cite-$1)");
  const byNumber = new Map(refs.map((r) => [r.n, r.ref]));

  return (
    <section className="turn">
      <div className="question">{t.question}</div>

      {(t.steps.length > 0 || t.thinking) && (
        <details className="activity" open={isRunning}>
          <summary>
            {isRunning ? <span className="spinner" /> : null}
            Agent activity · {t.steps.filter((s) => s.name).length} tool calls
          </summary>
          <ol className="steps">
            {t.steps.map((s, i) => (
              <li key={i}>
                <span className={`agent agent-${s.agent}`}>{s.agent}</span>
                {s.name ? (
                  <><code>{s.name}</code> <span className="muted">{formatArgs(s.args)}</span>
                    <span className="muted small"> · {s.chars} chars</span></>
                ) : <span className="muted">{s.text}</span>}
              </li>
            ))}
          </ol>
          {t.thinking && (
            <details className="thinking">
              <summary>Model reasoning</summary>
              <pre>{t.thinking}</pre>
            </details>
          )}
        </details>
      )}

      {t.approvals.map((a) => <ApprovalCard key={a.id} approval={a} chatId={chatId} />)}

      {body && (
        <div className="answer">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
            a: ({ href, children }) => {
              const n = href?.startsWith("#cite-") ? Number(href.slice(6)) : null;
              if (n !== null) {
                const ref = byNumber.get(n);
                return <button className="cite" title={ref} onClick={() => ref && api.openRef(ref)}>{n}</button>;
              }
              return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
            },
          }}>{markdown}</ReactMarkdown>
        </div>
      )}
      {isRunning && (
        <div className="live muted small">
          <span className="spinner" /> {lastActivity(t)} · <Elapsed since={t.started} />
        </div>
      )}
      {t.status === "stopped" && <div className="muted small">{t.error ?? "Stopped"}</div>}

      {refs.length > 0 && (
        <div className="refs">
          <span className="muted small">{cited ? "Sources" : "Retrieved, not cited"}</span>
          {refs.map((r) => (
            <button key={r.ref} className="ref" onClick={() => api.openRef(r.ref)} title="Open in editor">
              {r.n !== null && <b>{r.n}</b>} {r.ref}
            </button>
          ))}
        </div>
      )}
      {t.status === "error" && t.error && <div className="error">{t.error}</div>}
    </section>
  );
}

function ApprovalCard({ approval: a, chatId }: { approval: Approval; chatId: string | null }) {
  return (
    <div className={`approval ${a.state}`}>
      <div className="approval-title">The tester wants to run tests</div>
      <pre>{a.prompt}</pre>
      {a.state === "pending" ? (
        <div className="row">
          <button className="primary" onClick={() => chatId && api.approve(chatId, a.id, true)}>Run</button>
          <button onClick={() => chatId && api.approve(chatId, a.id, false)}>Decline</button>
        </div>
      ) : <div className="muted small">{a.state}</div>}
    </div>
  );
}

function parseRefs(block: string): { n: number | null; ref: string }[] {
  return block.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const m = l.match(/^\[(\d+)\]\s+(.+)$/);
    return m ? { n: Number(m[1]), ref: m[2] } : { n: null, ref: l.replace(/^-\s*/, "") };
  });
}

function formatArgs(args: unknown): string {
  if (!args || typeof args !== "object") return String(args ?? "");
  const text = Object.values(args as Record<string, unknown>).map((v) => JSON.stringify(v)).join(", ");
  return text.length > 140 ? text.slice(0, 140) + "…" : text;
}

function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function formatDate(unix: number): string {
  const d = new Date(unix * 1000);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { day: "numeric", month: "short" });
}
