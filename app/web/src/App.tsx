import { useCallback, useEffect, useState } from "react";
import { api, type Source, type Status } from "./api";
import { Chat } from "./Chat";
import { Sidebar } from "./Sidebar";

const ROOT_KEY = "docs-agent.root";

function loadRoot(): string {
  try {
    return localStorage.getItem(ROOT_KEY) ?? "";
  } catch {
    return "";
  }
}

export function App() {
  const [sources, setSources] = useState<Source[]>([]);
  const [status, setStatus] = useState<Status | null>(null);
  const [root, setRoot] = useState(loadRoot);

  const refresh = useCallback(async () => {
    const [src, st] = await Promise.all([api.sources(), api.status()]);
    setSources(src);
    setStatus(st);
  }, []);

  // poll fast while something is indexing, slowly otherwise
  const indexing = sources.some((s) => s.status === "queued" || s.status === "indexing");
  useEffect(() => {
    refresh().catch(() => {});
    const timer = setInterval(() => refresh().catch(() => {}), indexing ? 1000 : 5000);
    return () => clearInterval(timer);
  }, [refresh, indexing]);

  // default to the first source; drop a root that was removed
  useEffect(() => {
    if (sources.length && !sources.some((s) => s.path === root)) choose(sources[0].path);
  }, [sources, root]);

  function choose(path: string) {
    setRoot(path);
    try {
      localStorage.setItem(ROOT_KEY, path);
    } catch {
      /* storage unavailable */
    }
  }

  return (
    <div className="app">
      <Sidebar sources={sources} status={status} root={root} onRoot={choose} onChange={() => refresh()} />
      <Chat root={root} sources={sources} />
    </div>
  );
}
