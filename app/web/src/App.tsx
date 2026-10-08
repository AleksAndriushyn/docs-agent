import { useCallback, useEffect, useState } from "react";
import { api, type Collection, type Source, type Status } from "./api";
import { Chat } from "./Chat";
import { Sidebar } from "./Sidebar";

const ROOT_KEY = "docs-agent.root";
const COLLECTION_KEY = "docs-agent.collection";

function load(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
}

export function App() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [collection, setCollection] = useState(() => load(COLLECTION_KEY, "work"));
  const [sources, setSources] = useState<Source[]>([]);
  const [loadedFor, setLoadedFor] = useState("");  // collection the sources belong to
  const [status, setStatus] = useState<Status | null>(null);
  const [root, setRoot] = useState(() => load(ROOT_KEY, ""));

  const refresh = useCallback(async () => {
    const [cols, src, st] = await Promise.all([api.collections(), api.sources(collection), api.status(collection)]);
    setCollections(cols);
    setSources(src);
    setLoadedFor(collection);
    setStatus(st);
  }, [collection]);

  // poll fast while something is indexing, slowly otherwise
  const indexing = sources.some((s) => s.status === "queued" || s.status === "indexing");
  useEffect(() => {
    refresh().catch(() => {});
    const timer = setInterval(() => refresh().catch(() => {}), indexing ? 1000 : 5000);
    return () => clearInterval(timer);
  }, [refresh, indexing]);

  // the working folder must belong to the current collection: default to its first source
  useEffect(() => {
    if (loadedFor !== collection) return;  // stale or not yet fetched
    if (!sources.length) {
      if (root) choose("");
    } else if (!sources.some((s) => s.path === root)) {
      choose(sources[0].path);
    }
  }, [sources, root, loadedFor, collection]);

  function choose(path: string) {
    setRoot(path);
    store(ROOT_KEY, path);
  }

  function switchCollection(name: string) {
    setCollection(name);
    store(COLLECTION_KEY, name);
    setSources([]);  // don't show the previous collection's sources until the fetch lands
    setLoadedFor("");
  }

  return (
    <div className="app">
      <Sidebar collections={collections} collection={collection} onCollection={switchCollection}
        sources={sources} status={status} root={root} onRoot={choose} onChange={() => refresh()} />
      <Chat key={collection} root={root} collection={collection} sources={sources} />
    </div>
  );
}
