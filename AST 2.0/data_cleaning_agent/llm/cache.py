"""
llm/cache.py - persistent SQLite cache for validated LLM answers.

Key = sha256(task + prompt_version + canonical payload JSON). The payload is built from
the column-profile signature, issue types and evidence, so the same file (or the same
issue on a similar file) costs zero calls. Only answers that already passed validation
are stored. A broken cache never breaks the app: every error is treated as a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def make_key(task: str, prompt_version: str, payload: Any) -> str:
    raw = f"{task}\x00{prompt_version}\x00{canonical_json(payload)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        try:
            if path != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "key TEXT PRIMARY KEY, task TEXT NOT NULL, response TEXT NOT NULL, "
                "provider TEXT, model TEXT, created_at REAL NOT NULL)"
            )
            self._conn.commit()
        except (sqlite3.Error, OSError):
            self._conn = None  # run without a cache rather than fail

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def get(self, key: str) -> str | None:
        if not self._conn:
            return None
        try:
            with self._lock:
                row = self._conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None
        except sqlite3.Error:
            return None

    def put(self, key: str, task: str, response: str, provider: str, model: str) -> None:
        if not self._conn:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, task, response, provider, model, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, task, response, provider, model, time.time()),
                )
                self._conn.commit()
        except sqlite3.Error:
            pass

    def count(self) -> int:
        if not self._conn:
            return 0
        try:
            with self._lock:
                return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])
        except sqlite3.Error:
            return 0

    def clear(self) -> None:
        if not self._conn:
            return
        try:
            with self._lock:
                self._conn.execute("DELETE FROM llm_cache")
                self._conn.commit()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        if self._conn:
            with self._lock:
                self._conn.close()
            self._conn = None
