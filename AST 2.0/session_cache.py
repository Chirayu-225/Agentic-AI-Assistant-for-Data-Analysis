"""
session_cache.py — an in-memory, session-scoped cache of "already fetched
and already cleaned" small/medium CSV uploads, so a follow-up question
against the SAME upload doesn't pay for a Blob fetch and a fresh
pd.read_csv() + rule_based_clean() pass every single time.

WHY THIS EXISTS
    Before this, every /analyze request re-fetched the file from Blob
    storage (a real network round-trip) and re-parsed + re-cleaned it
    with pandas, even when it was the fifth question in a row about the
    exact same upload — only the query text changed between requests,
    never the file. That's real, repeated, wasted work on the single
    most common interaction pattern this app has (ask several questions
    about one dataset), not a rare edge case.

WHY KEYED ON (session_id, file_url), NOT JUST session_id OR just a
content hash
    - session_id alone would collide across different files the same
      session uploads over time (or in the multi-table/RAG tabs, which
      this cache doesn't touch at all — see SCOPE below).
    - A content hash needs the bytes in hand to compute, which means
      fetching first — defeating the actual point, which is skipping the
      fetch itself. file_url is already known BEFORE any fetch (the
      frontend sends it straight from Blob storage), so it's the only
      key that lets api.py decide whether to fetch at all.
    - file_url alone (no session_id) would leak one session's cached
      data to a different session that somehow reused the exact same
      URL — not how Vercel Blob URLs work (see below) but the extra key
      costs nothing and removes any doubt.

WHY THIS IS SAFE WITHOUT EXPLICIT INVALIDATION
    Every upload gets a new, unique, unguessable Blob URL (see the
    Next.js repo's README: "Known trade-off: Blob access is public" —
    the URL itself is what makes a given upload addressable). A
    "corrected" re-upload of the same filename gets a DIFFERENT file_url,
    so it's automatically a cache miss — there's no scenario where the
    same (session_id, file_url) key should ever legitimately map to two
    different files, so there's nothing to invalidate.

SCOPE — deliberately narrow
    This cache holds ONLY the single-dataset /analyze path's cleaned
    DataFrame, for files under file_reader.LARGE_FILE_THRESHOLD_BYTES.
    It does NOT cover:
      - The large-CSV path (analysis_service._run_large_csv_analysis) —
        that path already avoids a full pandas parse by design (see
        sandbox_limits.run_large_csv_sql_sandboxed's docstring); caching
        would mean keeping a large temp file (or its Arrow table) alive
        across requests, a real feature with its own memory/disk
        lifecycle questions, not a small addition to this cache.
      - Multi-table joins or RAG document indexing — different call
        patterns (multiple files per request; RAG already keeps its own
        per-document index), out of scope for what was actually asked.

CONCURRENCY AND LIFETIME
    This is a plain in-memory dict, guarded by a lock for FastAPI's
    threaded request handling — consistent with this project's existing
    single-process assumption (see sandbox_limits.py's own docstring:
    "Streamlit/Flask is typically single-process, multi-session"). It is
    NOT shared across multiple server processes/replicas — if this ever
    runs behind more than one uvicorn worker or Render instance, a
    request can land on a process that never saw the earlier fetch, and
    silently falls back to a cache miss (fetches again) rather than
    breaking — see get()'s docstring. Entries expire after TTL_SECONDS
    and the cache is capped at MAX_ENTRIES total (oldest evicted first)
    so a long-running server doesn't accumulate unbounded memory across
    many sessions/files over its lifetime.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

TTL_SECONDS = 30 * 60   # 30 minutes — long enough to cover a real back-and-forth Q&A session
MAX_ENTRIES = 100       # bounds total memory; oldest entries evicted first once exceeded


@dataclass
class CachedFile:
    filename: str
    csv_bytes: bytes
    cleaned_df: pd.DataFrame
    cached_at: float


_cache: dict[tuple[str, str], CachedFile] = {}
_lock = threading.Lock()


def _is_expired(entry: CachedFile) -> bool:
    return (time.time() - entry.cached_at) > TTL_SECONDS


def get(session_id: str, file_url: str) -> Optional[CachedFile]:
    """Returns the cached (filename, csv_bytes, cleaned_df) for this exact
    session + file, or None on a miss (never seen before, expired, or —
    on a multi-process deployment — cached by a different process). A
    None here is ALWAYS treated as "just fetch and parse it normally" by
    every caller, never as an error, so a cache that's cold, wrong-process,
    or disabled degrades to the pre-caching behavior, not a broken one."""
    key = (session_id, file_url)
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        if _is_expired(entry):
            del _cache[key]
            return None
        return entry


def put(session_id: str, file_url: str, filename: str, csv_bytes: bytes, cleaned_df: pd.DataFrame) -> None:
    """Stores the ALREADY-cleaned DataFrame — the whole point is that the
    next call to get() hands back something query routing + sandboxed
    exec can run against directly, with parsing and cleaning already
    paid for."""
    key = (session_id, file_url)
    with _lock:
        if key not in _cache and len(_cache) >= MAX_ENTRIES:
            oldest_key = min(_cache, key=lambda k: _cache[k].cached_at)
            del _cache[oldest_key]
        _cache[key] = CachedFile(
            filename=filename, csv_bytes=csv_bytes, cleaned_df=cleaned_df, cached_at=time.time(),
        )


def clear() -> None:
    """Test-only convenience — real callers never need to clear this
    themselves; TTL/size limits handle it."""
    with _lock:
        _cache.clear()
