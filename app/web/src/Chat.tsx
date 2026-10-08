import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, streamChat, type ChatEvent, type Source } from "./api";

type Step = { agent: string; name?: string; args?: unknown; chars?: number; text?: string };
type Approval = { id: string; prompt: string; state: "pending" | "approved" | "declined" };

type Turn = {
  question: string;
  answer: string;
  thinking: string;
  steps: Step[];
  approvals: Approval[];
  error?: string;
  running: boolean;
};

const FOOTER = /\n*(Sources|Retrieved \(not cited\)):\n([\s\S]*)$/;

export function Chat({ root, collection, sources }: { root: string; collection: string; sources: Source[] }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const running = turns.some((t) => t.running);
  const end = useRef<HTMLDivElement>(null);
  const rootName = sources.find((s) => s.path === root)?.name;

  useEffect(() => {
    // braces matter: scrollIntoView may return a Promise, and an effect's return value is its cleanup
    end.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  async function newChat() {
    await api.resetChat();
    setTurns([]);
  }

  async function send() {
    const question = input.trim();
    if (!question || running || !root) return;
    setInput("");
    setTurns((ts) => [...ts, { question, answer: "", thinking: "", steps: [], approvals: [], running: true }]);
    const update = (fn: (t: Turn) => Turn) =>
      setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? fn(t) : t)));

    await streamChat(question, root, collection, (e: ChatEvent) => {
      switch (e.type) {
        case "text": return update((t) => ({ ...t, answer: t.answer + e.text }));
        case "thinking": return update((t) => ({ ...t, thinking: t.thinking + e.text }));
        case "log": return update((t) => ({ ...t, steps: [...t.steps, { agent: e.agent, text: e.text }] }));
        case "tool": return update((t) => ({ ...t, steps: [...t.steps, e] }));
        case "approval":
          return update((t) => ({ ...t, approvals: [...t.approvals, { id: e.id, prompt: e.prompt, state: "pending" }] }));
        case "approval_done":
          return update((t) => ({
            ...t,
            approvals: t.approvals.map((a) => a.id === e.id ? { ...a, state: e.approved ? "approved" : "declined" } : a),
          }));
        case "error": return update((t) => ({ ...t, error: e.message, running: false }));
        case "done": return update((t) => ({ ...t, running: false }));
      }
    });
    update((t) => ({ ...t, running: false }));
  }

  return (
    <main className="chat">
      <header className="chat-head">
        <div>
          <span className="muted">Knowledge base: </span><strong>{collection}</strong>
          <span className="muted"> · working folder: </span>
          <strong>{rootName ?? "add a source on the left"}</strong>
        </div>
        <button onClick={newChat} disabled={running || turns.length === 0}>New chat</button>
      </header>

      <div className="messages">
        {turns.length === 0 && (
          <div className="welcome muted">
            <p>Ask about your indexed files. The supervisor delegates to agents:</p>
            <ul>
              <li><b>researcher</b> searches the knowledge base by meaning, greps and reads the working folder</li>
              <li><b>tester</b> writes and runs pytest tests (asks before running)</li>
            </ul>
          </div>
        )}
        {turns.map((t, i) => <TurnView key={i} turn={t} />)}
        <div ref={end} />
      </div>

      <footer className="composer">
        <textarea
          value={input}
          placeholder={root ? "Ask a question… (Enter to send, Shift+Enter for a new line)" : "Pick a working folder first"}
          disabled={!root}
          rows={3}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button className="primary" onClick={send} disabled={running || !input.trim() || !root}>
          {running ? "Working…" : "Send"}
        </button>
      </footer>
    </main>
  );
}

function TurnView({ turn: t }: { turn: Turn }) {
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
        <details className="activity" open={t.running}>
          <summary>
            {t.running ? <span className="spinner" /> : null}
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

      {t.approvals.map((a) => <ApprovalCard key={a.id} approval={a} />)}

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
      {t.running && !body && t.approvals.every((a) => a.state !== "pending") && (
        <div className="muted small"><span className="spinner" /> thinking…</div>
      )}

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
      {t.error && <div className="error">{t.error}</div>}
    </section>
  );
}

function ApprovalCard({ approval: a }: { approval: Approval }) {
  return (
    <div className={`approval ${a.state}`}>
      <div className="approval-title">The tester wants to run tests</div>
      <pre>{a.prompt}</pre>
      {a.state === "pending" ? (
        <div className="row">
          <button className="primary" onClick={() => api.approve(a.id, true)}>Run</button>
          <button onClick={() => api.approve(a.id, false)}>Decline</button>
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
