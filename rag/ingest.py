"""Index a folder into Qdrant. Incremental: unchanged files (same sha256) are skipped,
files that are deleted or no longer selected by --include/--exclude are removed.

    python -m rag.ingest --src ~/work --collection work
    python -m rag.ingest @sources/missionplanner.args      # arguments from a file, one per line
"""
from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.config import EMBED_MODEL, RAG_COLLECTION
from rag.chunking import chunk
from rag.embed import DenseEmbedder, SparseEmbedder
from rag.loaders import iter_files, load
from rag.store import Store


def embed_text(payload: dict) -> str:
    """Text that gets embedded: path and code context help retrieval, the stored text stays raw."""
    head = payload["path"]
    if payload["context"]:
        head += f"\n{payload['context']}"
    if payload["symbol"]:
        head += f"\n{payload['symbol']}"
    return f"{head}\n\n{payload['text']}"


@dataclass
class Stats:
    total: int = 0
    changed: int = 0
    chunks: int = 0
    removed: int = 0
    unchanged: int = 0
    errors: int = 0
    shrunk: int = 0
    seconds: float = 0.0
    cancelled: bool = False


# on_file(done, total, rel_path, status): status is "indexed", "skipped", "removed" or "error: ..."
ProgressFn = Callable[[int, int, str, str], None]


def ingest(src: Path, store: Store, dense: DenseEmbedder, sparse: SparseEmbedder,
           include: list[str] | None = None, exclude: list[str] | None = None,
           on_file: ProgressFn | None = None,
           should_stop: Callable[[], bool] | None = None) -> Stats:
    source = str(src)
    indexed = store.file_hashes(source) if store.exists() else {}
    files = list(iter_files(src, include, exclude))
    st, seen, t0, shrunk0 = Stats(total=len(files)), set(), time.time(), dense.shrunk
    report = on_file or (lambda *a: None)

    for done, f in enumerate(files, 1):
        if should_stop and should_stop():
            # stale-file cleanup is skipped: unvisited files would look deleted
            st.cancelled = True
            break
        rel = f.relative_to(src).as_posix()
        seen.add(rel)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if indexed.get(rel) == sha:
            st.unchanged += 1
            report(done, st.total, rel, "skipped")
            continue
        try:
            doc = load(f, src)
            chunks = chunk(doc) if doc else []
        except Exception as e:  # one broken docx shouldn't stop the whole run
            st.errors += 1
            report(done, st.total, rel, f"error: {e}")
            continue
        if not chunks:
            report(done, st.total, rel, "skipped")
            continue

        payloads = [{
            "source": source, "path": rel, "lang": doc.lang, "file_sha256": sha,
            "chunk_index": i, "line_start": c.line_start, "line_end": c.line_end,
            "symbol": c.symbol, "context": c.context, "text": c.text,
        } for i, c in enumerate(chunks)]
        texts = [embed_text(p) for p in payloads]
        dense_vecs = dense.embed(texts)
        store.ensure(len(dense_vecs[0]))
        if rel in indexed:
            store.delete_file(source, rel)  # chunk count may differ from the old version
        store.upsert(payloads, dense_vecs, sparse.embed(texts))
        st.changed += 1
        st.chunks += len(chunks)
        report(done, st.total, rel, "indexed")

    for rel in set() if st.cancelled else set(indexed) - seen:
        store.delete_file(source, rel)
        st.removed += 1
        report(st.total, st.total, rel, "removed")
    st.shrunk = dense.shrunk - shrunk0
    st.seconds = time.time() - t0
    return st


def print_progress(done: int, total: int, rel: str, status: str) -> None:
    if status == "indexed":
        print(f"  + {rel}")
    elif status == "removed":
        print(f"  - {rel}")
    elif status.startswith("error"):
        print(f"  ! {rel}: {status}")


def dry_run(src: Path, include: list[str] | None, exclude: list[str] | None) -> None:
    by_dir: Counter[str] = Counter()
    size = 0
    for f in iter_files(src, include, exclude):
        rel = f.relative_to(src)
        by_dir[rel.parts[0] if len(rel.parts) > 1 else "(root)"] += 1
        size += f.stat().st_size
    for d, n in by_dir.most_common():
        print(f"{n:6}  {d}")
    # ~1 chunk per 1.2 KB and ~20 chunks/s on docs/ (qwen3-embedding:0.6b, M5 Pro)
    print(f"Total: {sum(by_dir.values())} files, {size / 1e6:.1f} MB, "
          f"~{size / 1200:.0f} chunks, ~{size / 1200 / 20 / 60:.0f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description="Index a folder into Qdrant", fromfile_prefix_chars="@")
    ap.add_argument("--src", required=True, help="Folder to index")
    ap.add_argument("--collection", default=RAG_COLLECTION)
    ap.add_argument("--include", action="append", help="Glob relative to --src, repeatable "
                                                        "(e.g. 'GCSViews/**', '*.cs')")
    ap.add_argument("--exclude", action="append", help="Glob to skip, repeatable (e.g. '**/*.Designer.cs')")
    ap.add_argument("--dry-run", action="store_true", help="Only list what would be indexed")
    ap.add_argument("--reset", action="store_true", help="Drop the collection and reindex from scratch")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}")
    if args.dry_run:
        dry_run(src, args.include, args.exclude)
        return
    store = Store(args.collection)
    if args.reset:
        store.drop()
    print(f"{src} -> collection '{args.collection}' (embeddings: {EMBED_MODEL})")
    st = ingest(src, store, DenseEmbedder(), SparseEmbedder(), args.include, args.exclude,
                on_file=print_progress)
    print(f"Done in {st.seconds:.1f}s: changed files {st.changed}, chunks {st.chunks}, "
          f"removed files {st.removed}, unchanged {st.unchanged}, errors {st.errors}.")
    if st.shrunk:
        print(f"{st.shrunk} chunks were too token-dense and got embedded from a shortened prefix.")


if __name__ == "__main__":
    main()
