"""
llm/limiter.py - token-bucket rate limiter that keeps us BELOW the free-tier cap.

One bucket guards every outgoing LLM request (all providers share it). Clock and
sleep are injectable so tests never really wait.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class TokenBucket:
    def __init__(
        self,
        rate_per_min: float = 20.0,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be positive")
        self.rate = rate_per_min / 60.0  # tokens per second
        # A small burst allowance keeps the long-run rate at or under rate_per_min.
        self.capacity = float(capacity if capacity is not None else max(1.0, rate_per_min // 4))
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def wait_time(self) -> float:
        """Seconds until one token is available (0 if one is available now)."""
        with self._lock:
            self._refill()
            return max((1.0 - self._tokens) / self.rate, 0.0)

    def acquire(self, max_wait: float | None = None) -> bool:
        """Take a token, sleeping if needed. Returns False (without taking one) when the
        total wait would exceed `max_wait` seconds."""
        waited = 0.0
        while True:
            if self.try_acquire():
                return True
            wait = self.wait_time()
            if max_wait is not None and waited + wait > max_wait:
                return False
            self._sleep(wait)
            waited += wait
