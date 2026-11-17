"""
rate_limiter.py — in-process sliding-window rate limiting for the API.

WHY THIS EXISTS
    Every LLM-backed endpoint costs real money and real seconds, and the
    sandboxed-exec endpoints cost CPU and memory. The shared-secret header
    stops strangers, but nothing stopped a single session (a script, a
    stuck retry loop, a curious user mashing "Run") from firing hundreds
    of expensive calls — the Groq quota or the server's RAM would be the
    thing that finally said no. This puts a ceiling in front of that.

HOW IT KEYS REQUESTS
    Behind the Next.js server, every request reaches this API from the
    SAME network address, so per-IP limiting alone would throttle all
    users together. Limits are therefore keyed on the caller's sessionId
    (docId for the document endpoints that carry no session), with a
    generous per-IP backstop on top (see api.py).

POLICY (requests per window; env var RATE_LIMIT_MULTIPLIER scales all)
    heavy  — auto-analyze, clean/agent: several LLM calls each
               4 / minute and 30 / hour
    medium — analyze, follow-up, multi-table, rag/query, rag/insights
               20 / minute and 200 / hour
    light  — profile, clean, rag/index: file parsing only
               30 / minute
    ip     — backstop per source address:  300 / minute

DESIGN NOTES
    - Sliding-window LOG (a deque of timestamps per key): exact, simple,
      no burst-at-the-boundary problem of fixed windows. Memory per key is
      bounded by the largest limit in force.
    - In-memory and per-process, like session_cache.py: with several
      uvicorn workers each worker counts separately, so the effective
      limit is limits x workers. Use Redis if this is ever scaled out.
    - Idle keys are pruned and the table is capped at MAX_KEYS so a flood
      of unique session ids can't grow memory without bound.
    - Rejections are returned with a Retry-After header (seconds).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Callable, Optional

MAX_KEYS = 50_000

# group -> list of (max_requests, window_seconds)
POLICIES: dict[str, list[tuple[int, int]]] = {
    "heavy":  [(4, 60), (30, 3600)],
    "medium": [(20, 60), (200, 3600)],
    "light":  [(30, 60)],
    "ip":     [(300, 60)],
}


def enabled() -> bool:
    return os.environ.get("RATE_LIMIT_ENABLED", "1") != "0"


def _multiplier() -> float:
    try:
        return max(float(os.environ.get("RATE_LIMIT_MULTIPLIER", "1")), 0.01)
    except ValueError:
        return 1.0


class SlidingWindowLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = {}
        self._max_window: dict[str, float] = {}
        self.allowed_total = 0
        self.rejected_total = 0

    def check(self, key: str, rules: list[tuple[int, float]]) -> tuple[bool, float]:
        """Records a hit for `key` if EVERY rule allows it.
        Returns (allowed, retry_after_seconds). A rejected request is not
        recorded, so hammering a limited endpoint doesn't extend the ban."""
        now = self._clock()
        longest = max(w for _, w in rules)
        with self._lock:
            dq = self._hits.get(key)
            if dq is None:
                if len(self._hits) >= MAX_KEYS:
                    self._prune(now, force=True)
                dq = self._hits[key] = deque()
            self._max_window[key] = max(self._max_window.get(key, 0), longest)

            while dq and now - dq[0] > longest:
                dq.popleft()

            retry_after = 0.0
            for limit, window in rules:
                in_window = [t for t in dq if now - t <= window]
                if len(in_window) >= limit:
                    # the oldest hit in this window must expire before one more fits
                    retry_after = max(retry_after, window - (now - in_window[0]))
            if retry_after > 0:
                self.rejected_total += 1
                return False, max(retry_after, 0.001)

            dq.append(now)
            self.allowed_total += 1
            if len(self._hits) > MAX_KEYS // 2 and self.allowed_total % 1000 == 0:
                self._prune(now)
            return True, 0.0

    def _prune(self, now: float, force: bool = False) -> None:
        dead = [k for k, dq in self._hits.items()
                if not dq or now - dq[-1] > self._max_window.get(k, 0)]
        for k in dead:
            self._hits.pop(k, None)
            self._max_window.pop(k, None)
        if force and len(self._hits) >= MAX_KEYS:
            # still full of live keys: drop the least recently active quarter
            by_age = sorted(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0)
            for k in by_age[: MAX_KEYS // 4]:
                self._hits.pop(k, None)
                self._max_window.pop(k, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "trackedKeys": len(self._hits),
                "allowed": self.allowed_total,
                "rejected": self.rejected_total,
            }

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._max_window.clear()
            self.allowed_total = self.rejected_total = 0


limiter = SlidingWindowLimiter()


def rules_for(group: str) -> list[tuple[int, float]]:
    base = POLICIES[group]
    m = _multiplier()
    return [(max(int(round(n * m)), 1), w) for n, w in base]


def check(group: str, key: str) -> tuple[bool, float]:
    """Convenience wrapper: applies the named policy to `key`.
    Always allows when RATE_LIMIT_ENABLED=0."""
    if not enabled():
        return True, 0.0
    return limiter.check(f"{group}:{key}", rules_for(group))
