"""
resilience.py - keep answering when the model provider is slow or down.

THE HAZARD (H5, availability side)
    Every question depends on a remote model. Without limits, one slow or
    dead provider makes every request wait out a 90-second timeout, several
    times over (plan, repair, code, retry), and the service is down in all
    but name.

THREE MECHANISMS
    1. A SECOND PROVIDER. llm_provider retries a failed call on a fallback
       (Gemini by default when GEMINI_API_KEY is set). The request records
       that a fallback answered, and the response says so.

    2. A CIRCUIT BREAKER per provider. After FAILURE_THRESHOLD failures in a
       row the provider is skipped entirely for COOLDOWN_SECONDS (calls go
       straight to the fallback, or fail fast if there is none). Then ONE
       trial call is let through: success closes the circuit, failure opens
       it again. A daily-quota refusal opens it at once for a longer time.

            closed --(N failures in a row)--> open --(cooldown)--> half_open
               ^                                ^                     |
               +--------- trial succeeds -------+---- trial fails ----+

    3. ONE DEADLINE PER REQUEST. request_deadline(seconds) starts a budget;
       every model call inside it gets a timeout no longer than what is left,
       and no call starts once the budget is spent. A request therefore ends
       in bounded time however many model calls its route makes.

    Env: LLM_BREAKER_FAILURES (3), LLM_BREAKER_COOLDOWN_S (60), REQUEST_DEADLINE_S (120).

LIMITS
    State is per process and in memory: several server processes each keep
    their own breaker, and a restart resets it.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Callable, Optional


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


FAILURE_THRESHOLD = _int_env("LLM_BREAKER_FAILURES", 3)
COOLDOWN_SECONDS = _int_env("LLM_BREAKER_COOLDOWN_S", 60)
MAX_COOLDOWN_SECONDS = 900

_clock: Callable[[], float] = time.monotonic


def set_clock(fn: Callable[[], float]) -> None:
    """Tests replace the clock so no test has to wait."""
    global _clock
    _clock = fn


class DeadlineExceeded(RuntimeError):
    """The request's time budget is spent."""


class ProviderUnavailable(RuntimeError):
    """The provider's circuit is open and there is no fallback."""


# ── circuit breaker ────────────────────────────────────────────────────────
class CircuitBreaker:
    def __init__(self, name: str, threshold: Optional[int] = None, cooldown: Optional[float] = None):
        self.name = name
        self.threshold = threshold or FAILURE_THRESHOLD
        self.cooldown = cooldown or COOLDOWN_SECONDS
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until: Optional[float] = None
        self._trial_in_flight = False
        self.opened_total = 0

    def state(self) -> str:
        with self._lock:
            if self._open_until is None:
                return "closed"
            return "open" if _clock() < self._open_until else "half_open"

    def allow(self) -> bool:
        """May a call go to this provider now? In half_open, only one caller gets a yes."""
        with self._lock:
            if self._open_until is None:
                return True
            if _clock() < self._open_until or self._trial_in_flight:
                return False
            self._trial_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures, self._open_until, self._trial_in_flight = 0, None, False

    def record_failure(self, cooldown: Optional[float] = None) -> None:
        """`cooldown` set = open immediately for that long (a quota that will not clear soon)."""
        with self._lock:
            was_trial = self._trial_in_flight
            self._trial_in_flight = False
            self._failures += 1
            if cooldown is not None or was_trial or self._failures >= self.threshold:
                wait = min(float(cooldown if cooldown is not None else self.cooldown), MAX_COOLDOWN_SECONDS)
                if self._open_until is None or was_trial:
                    self.opened_total += 1
                self._open_until = _clock() + wait

    def release_trial(self) -> None:
        """A trial call ended without telling us anything (it was cut short by
        the request's own deadline): let the next caller try instead."""
        with self._lock:
            self._trial_in_flight = False

    def retry_in(self) -> float:
        with self._lock:
            return max(0.0, (self._open_until or 0) - _clock()) if self._open_until else 0.0

    def stats(self) -> dict:
        return {"state": self.state(), "consecutiveFailures": self._failures,
                "retryInSeconds": round(self.retry_in(), 1), "timesOpened": self.opened_total}


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def breaker(name: str) -> CircuitBreaker:
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(name)
        return _breakers[name]


def reset_breakers() -> None:
    with _breakers_lock:
        _breakers.clear()


def breaker_stats() -> dict:
    """For /admin/stats and /health."""
    with _breakers_lock:
        return {name: b.stats() for name, b in _breakers.items()}


# ── request deadline ───────────────────────────────────────────────────────
_tls = threading.local()


class RequestScope:
    def __init__(self, deadline: float):
        self.deadline = deadline
        self.fallback_used = False

    def remaining(self) -> float:
        return self.deadline - _clock()


def _stack() -> list:
    if not hasattr(_tls, "scopes"):
        _tls.scopes = []
    return _tls.scopes


@contextlib.contextmanager
def request_deadline(seconds: float):
    """Everything inside shares one time budget. Nesting can shorten the
    budget but never extend the one it is inside."""
    stack = _stack()
    deadline = _clock() + float(seconds)
    if stack:
        deadline = min(deadline, stack[-1].deadline)
    scope = RequestScope(deadline)
    stack.append(scope)
    try:
        yield scope
    finally:
        stack.pop()


def remaining() -> float:
    """Seconds left in the current request, or infinity outside any request."""
    stack = _stack()
    return stack[-1].remaining() if stack else float("inf")


def check_deadline() -> None:
    if remaining() <= 0:
        raise DeadlineExceeded("This request took too long and was stopped. Please try again, "
                               "or ask a simpler question.")


def call_timeout(default: float) -> float:
    """The timeout to give one model call: the default, or less if the request has less left."""
    return max(1.0, min(float(default), remaining()))


def note_fallback_used() -> None:
    for scope in _stack():
        scope.fallback_used = True


def default_request_seconds() -> float:
    try:
        return max(5.0, float(os.environ.get("REQUEST_DEADLINE_S", "120")))
    except ValueError:
        return 120.0
