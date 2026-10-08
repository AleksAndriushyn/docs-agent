import { useState } from "react";
import { api, pickPaths, type Collection, type Source, type Status } from "./api";

type Props = {
  collections: Collection[];
  collection: string;
  onCollection: (name: string) => void;
  sources: Source[];
  status: Status | null;
  root: string;
  onRoot: (path: string) => void;
  onChange: () => void;
};

export function Sidebar({ collections, collection, onCollection, sources, status, root, onRoot, onChange }: Props) {
  const [error, setError] = useState("");

  async function add(kind: "folder" | "files") {
    setError("");
    try {
      for (const path of await pickPaths(kind)) await api.addSource(path, collection);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
    onChange();
  }

  return (
    <aside className="sidebar">
      <header className="sidebar-head">
        <h1>Knowledge base</h1>
        <CollectionPicker collections={collections} value={collection} onChange={onCollection}
          onCreated={onChange} onError={setError} />
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
        {sources.length === 0 && (
          <li className="muted empty">“{collection}” is empty. Add a folder or files to index them.</li>
        )}
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

function CollectionPicker({ collections, value, onChange, onCreated, onError }: {
  collections: Collection[]; value: string; onChange: (name: string) => void;
  onCreated: () => void; onError: (msg: string) => void;
}) {
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [name, setName] = useState("");
  const names = collections.some((c) => c.name === value)
    ? collections : [...collections, { name: value, points: 0, sources: 0 }];
  const current = names.find((c) => c.name === value);

  async function create() {
    const n = name.trim();
    if (!n) return;
    try {
      await api.createCollection(n);
      onChange(n);
      onCreated();
      setCreating(false);
      setName("");
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }

  async function remove() {
    try {
      await api.deleteCollection(value);
      setDeleting(false);
      const next = collections.find((c) => c.name !== value);
      onChange(next?.name ?? "work");
      onCreated();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }

  if (creating) {
    return (
      <div className="row collection-picker">
        <input autoFocus value={name} placeholder="new-base-name" aria-label="New knowledge base name"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") create(); if (e.key === "Escape") setCreating(false); }} />
        <button className="primary" onClick={create} disabled={!name.trim()}>Create</button>
        <button onClick={() => setCreating(false)}>Cancel</button>
      </div>
    );
  }

  return (
    <div className="row collection-picker">
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label="Knowledge base">
        {names.map((c) => (
          <option key={c.name} value={c.name}>
            {c.name} · {c.sources} {c.sources === 1 ? "source" : "sources"}
          </option>
        ))}
      </select>
      <button onClick={() => setCreating(true)} title="New knowledge base">New</button>
      <button className="danger" onClick={() => setDeleting(true)} title="Delete this knowledge base">Delete</button>
      {deleting && (
        <div className="confirm">
          <span>Delete knowledge base <b>{value}</b> with {current?.sources ?? 0} sources and{" "}
            {(current?.points ?? 0).toLocaleString()} chunks? Files on disk are not touched,
            but reindexing takes time.</span>
          <div className="row">
            <button className="danger-solid" onClick={remove}>Delete</button>
            <button onClick={() => setDeleting(false)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}
