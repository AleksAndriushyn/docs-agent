import { useState } from "react";
import { api, pickPaths, type Source, type Status } from "./api";

type Props = {
  sources: Source[];
  status: Status | null;
  root: string;
  onRoot: (path: string) => void;
  onChange: () => void;
};

export function Sidebar({ sources, status, root, onRoot, onChange }: Props) {
  const [error, setError] = useState("");

  async function add(kind: "folder" | "files") {
    setError("");
    try {
      for (const path of await pickPaths(kind)) await api.addSource(path);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
    onChange();
  }

  return (
    <aside className="sidebar">
      <header className="sidebar-head">
        <h1>Knowledge base</h1>
        {status && (
          <div className="health">
            <Dot ok={status.ollama} label={`Ollama · ${status.model}`} />
            <Dot ok={status.qdrant} label={`Qdrant (${status.qdrant_mode})`} />
            <span className="muted">{status.points.toLocaleString()} chunks in “{status.collection}”</span>
          </div>
        )}
        <div className="row">
          <button onClick={() => add("folder")}>+ Folder</button>
          <button onClick={() => add("files")}>+ Files</button>
        </div>
        {error && <p className="error">{error}</p>}
      </header>

      <ul className="sources">
        {sources.length === 0 && <li className="muted empty">Add a folder or files to index them.</li>}
        {sources.map((s) => (
          <SourceItem key={s.id} source={s} active={s.path === root} onRoot={onRoot} onChange={onChange} />
        ))}
      </ul>
    </aside>
  );
}

function SourceItem({ source: s, active, onRoot, onChange }: {
  source: Source; active: boolean; onRoot: (p: string) => void; onChange: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [include, setInclude] = useState(s.include.join("\n"));
  const [exclude, setExclude] = useState(s.exclude.join("\n"));
  const busy = s.status === "queued" || s.status === "indexing";
  const pct = s.progress && s.progress.total ? Math.round((s.progress.done / s.progress.total) * 100) : 0;

  async function act(fn: () => Promise<unknown>) {
    try {
      await fn();
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
    onChange();
  }

  return (
    <li className={`source ${active ? "active" : ""}`}>
      <button className="source-main" onClick={() => onRoot(s.path)} title="Use as working folder for chat">
        <span className="source-name">{s.name}</span>
        <span className={`badge ${s.status}`}>{s.status}</span>
      </button>
      <div className="source-path muted" title={s.path}>
        {/* LRM marks keep slashes in place inside the rtl box used for start-ellipsis */}
        {"\u200E" + s.path + "\u200E"}
      </div>

      {busy && (
        <div className="progress">
          <div className="bar"><div style={{ width: `${pct}%` }} /></div>
          <span className="muted">
            {s.progress ? `${s.progress.done}/${s.progress.total} · ${s.progress.current}` : "waiting…"}
          </span>
        </div>
      )}
      {!busy && s.status !== "new" && (
        <div className="muted small">
          {s.files} files · {s.chunks.toLocaleString()} chunks
          {s.include.length > 0 && ` · ${s.include.length} include rules`}
        </div>
      )}
      {s.error && <div className="error small">{s.error}</div>}

      <div className="row small-buttons">
        {busy
          ? <button className="danger" onClick={() => act(() => api.cancel(s.id))}>Stop</button>
          : <button onClick={() => act(() => api.reindex(s.id))}>
              {s.status === "cancelled" ? "Resume" : "Reindex"}
            </button>}
        <button disabled={busy} onClick={() => setEditing(!editing)}>Filters</button>
        <button className="danger" disabled={busy}
          onClick={() => confirm(`Remove ${s.name} from the index?`) && act(() => api.removeSource(s.id))}>
          Remove
        </button>
      </div>

      {editing && (
        <div className="filters">
          <label>Include (one glob per line, empty = whole folder)
            <textarea value={include} onChange={(e) => setInclude(e.target.value)} rows={4}
              placeholder={"GCSViews/**\n*.cs"} />
          </label>
          <label>Exclude
            <textarea value={exclude} onChange={(e) => setExclude(e.target.value)} rows={3}
              placeholder="**/*.Designer.cs" />
          </label>
          <p className="muted small"><code>*</code> stays in one folder, <code>**</code> spans folders.</p>
          <button onClick={() => act(async () => {
            await api.editSource(s.id, include.split("\n"), exclude.split("\n"));
            setEditing(false);
          })}>Save &amp; reindex</button>
        </div>
      )}
    </li>
  );
}

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return <span className="dot-label"><span className={`dot ${ok ? "ok" : "bad"}`} />{label}</span>;
}
