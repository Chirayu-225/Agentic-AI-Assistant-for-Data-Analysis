"""
cache_layer.py — bounded, thread-safe, TTL + LRU caches for the expensive,
REPEATED work every endpoint was doing from scratch.

WHAT WAS BEING REPEATED
    Every endpoint did: fetch the file from Blob storage (network), then
    parse it (read_csv / read_excel — ~24s for the large sheet in the
    live bug that raised the client timeouts), then rule_based_clean it.
    A user who uploads one file and then profiles it, cleans it, runs the
    cleaning agent, auto-analyzes it and asks three follow-ups paid that
    full cost around EIGHT times, even though the file never changed.
    session_cache.py only covered /analyze. This module covers the rest.

THREE CACHES (all keyed so staleness is impossible, see below)
    blob_cache   file_url -> raw bytes. Skips the network fetch.
    frame_cache  (content digest, filename) -> parsed DataFrame, and
                 (content digest, "cleaned") -> (cleaned DataFrame, log).
                 Skips parsing and rule-based cleaning.
    result_cache (endpoint, content digest) -> JSON-able response, ONLY
                 for deterministic endpoints (/profile, /clean).

WHAT IS DELIBERATELY NOT CACHED
    LLM-backed results (auto-analyze, cleaning agent, follow-ups, RAG
    answers). They are non-deterministic and a user pressing "run again"
    expects a fresh attempt — returning the identical stale answer would
    be a bug, not a speed-up. The expensive part those endpoints shared
    with the others (fetch + parse + clean) IS cached, which is exactly
    what the timeouts were caused by.

WHY STALENESS CAN'T HAPPEN
    Frames and results are keyed on a digest of the file's BYTES, so a
    changed file is by definition a different key. blob_cache keys on the
    URL, which is safe because every upload gets a new unguessable Blob
    URL (same argument as session_cache.py), and entries expire after
    CACHE_TTL_SECONDS anyway.

SAFETY
    - Cached DataFrames are returned as COPIES, so one request mutating
      its frame can never corrupt another request's data.
    - Memory is bounded by BYTES (not just entry count) with LRU eviction;
      an entry bigger than half a cache's budget isn't stored at all.
    - Failures to cache never fail a request; a miss is always "just do
      the work".
    - In-process only (like session_cache.py): with several workers each
      keeps its own cache — it still helps, it just hits less often.

CONFIG (environment variables)
    CACHE_ENABLED=0       turn everything off
    CACHE_TTL_SECONDS     default 1800
    CACHE_BLOB_MB         default 64
    CACHE_FRAME_MB        default 96
    CACHE_RESULT_MB       default 16
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def enabled() -> bool:
    return os.environ.get("CACHE_ENABLED", "1") != "0"


class BoundedCache:
    """LRU cache bounded by total bytes, with per-entry TTL."""

    def __init__(self, name: str, max_bytes: int, ttl_seconds: float,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.max_bytes = int(max_bytes)
        self.ttl = float(ttl_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._data: "OrderedDict[Any, tuple[Any, int, float]]" = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = self.evictions = self.rejected = 0

    def get(self, key: Any) -> Optional[Any]:
        if not enabled():
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            value, size, stored_at = entry
            if self._clock() - stored_at > self.ttl:
                self._drop(key)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: Any, value: Any, size_bytes: int) -> bool:
        if not enabled():
            return False
        size_bytes = int(size_bytes)
        if size_bytes > self.max_bytes // 2:
            with self._lock:
                self.rejected += 1
            return False
        with self._lock:
            if key in self._data:
                self._drop(key)
            self._data[key] = (value, size_bytes, self._clock())
            self._bytes += size_bytes
            while self._bytes > self.max_bytes and self._data:
                oldest = next(iter(self._data))
                self._drop(oldest)
                self.evictions += 1
            return True

    def _drop(self, key: Any) -> None:
        _, size, _ = self._data.pop(key)
        self._bytes -= size

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0
            self.hits = self.misses = self.evictions = self.rejected = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._data),
                "bytes": self._bytes,
                "maxBytes": self.max_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "tooLargeToCache": self.rejected,
                "hitRate": round(self.hits / total, 3) if total else None,
            }


_MB = 1024 * 1024
_TTL = _env_float("CACHE_TTL_SECONDS", 1800)

blob_cache = BoundedCache("blob", _env_float("CACHE_BLOB_MB", 64) * _MB, _TTL)
frame_cache = BoundedCache("frame", _env_float("CACHE_FRAME_MB", 96) * _MB, _TTL)
result_cache = BoundedCache("result", _env_float("CACHE_RESULT_MB", 16) * _MB, _TTL)


def stats() -> dict:
    return {
        "enabled": enabled(),
        "blob": blob_cache.stats(),
        "frame": frame_cache.stats(),
        "result": result_cache.stats(),
    }


def clear_all() -> None:
    for c in (blob_cache, frame_cache, result_cache):
        c.clear()


# ─────────────────────────────────────────────────────────────────────────────
# digests and helpers used by the rest of the app
# ─────────────────────────────────────────────────────────────────────────────

def content_digest(data: bytes) -> str:
    """Fast, non-cryptographic-grade-needed content fingerprint (blake2b,
    ~1 GB/s — a 30 MB upload costs ~30 ms to fingerprint, against seconds
    to parse)."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def frame_nbytes(df) -> int:
    try:
        return int(df.memory_usage(index=True, deep=True).sum())
    except Exception:
        return int(df.memory_usage(index=True).sum())


def cached_parse(file_bytes: bytes, filename: str, parse: Callable[[bytes, str], Any]):
    """Parses `file_bytes` via `parse`, returning a cached COPY when the
    exact same bytes + filename were parsed recently. Used by
    file_reader.read_tabular_file."""
    if not enabled():
        return parse(file_bytes, filename)
    key = ("parsed", content_digest(file_bytes), filename.lower().rsplit(".", 1)[-1] if "." in filename else "")
    hit = frame_cache.get(key)
    if hit is not None:
        return hit.copy()
    df = parse(file_bytes, filename)
    try:
        frame_cache.put(key, df.copy(), frame_nbytes(df))
    except Exception:
        pass  # caching is best-effort
    return df


def cached_rule_clean(df_raw, file_bytes: bytes, clean: Callable[[Any], tuple]):
    """rule_based_clean() is a pure function of the parsed table, so its
    output (cleaned frame + log) is cached by the file's content digest.
    Returns (cleaned_df_copy, log_copy)."""
    if not enabled():
        return clean(df_raw)
    key = ("cleaned", content_digest(file_bytes))
    hit = frame_cache.get(key)
    if hit is not None:
        df, log = hit
        return df.copy(), list(log)
    df, log = clean(df_raw)
    try:
        frame_cache.put(key, (df.copy(), list(log)), frame_nbytes(df))
    except Exception:
        pass
    return df, log


def cached_result(endpoint: str, file_bytes: bytes, compute: Callable[[], Any], extra: str = ""):
    """For deterministic endpoints: returns a deep copy of the cached
    response, computing and storing it on a miss. Only successful
    responses (dicts with success=True) are stored."""
    if not enabled():
        return compute()
    key = (endpoint, content_digest(file_bytes), extra)  # extra: e.g. filename, echoed in the response
    hit = result_cache.get(key)
    if hit is not None:
        return copy.deepcopy(hit)
    result = compute()
    try:
        if isinstance(result, dict) and result.get("success"):
            import json
            size = len(json.dumps(result, default=str))
            result_cache.put(key, copy.deepcopy(result), size)
    except Exception:
        pass
    return result
