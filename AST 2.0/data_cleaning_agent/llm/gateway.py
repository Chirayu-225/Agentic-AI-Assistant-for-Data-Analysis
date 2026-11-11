"""
llm/gateway.py - THE ONLY module in this project that calls an LLM API.

    call_llm(task, payload)  ->  a validated Pydantic object (see llm/tasks.py)

What it guarantees
  * cache first   identical requests are answered from SQLite at zero cost
  * bounded       a per-session call budget; retries/fallbacks do not count as extra calls
  * polite        one token bucket (default 20 req/min) in front of every request
  * resilient     retry with exponential backoff on 429/5xx/timeouts (max 3 attempts),
                  then fall back Groq -> Gemini -> Ollama (if configured)
  * never fatal   when nothing can answer it raises LLMUnavailable / LLMInvalidOutput, and the
                  caller falls back to rule-based proposals (low confidence, source=rule)
  * no raw data   payloads are checked: no row dumps, short strings, <= 8 samples
  * no PII        personal columns are withheld and PII patterns redacted first (llm/privacy.py)

Config (all optional env vars; see GatewayConfig):
  GROQ_API_KEY  GROQ_SMALL_MODEL  GROQ_LARGE_MODEL
  GEMINI_API_KEY  GEMINI_MODEL
  OLLAMA_MODEL  OLLAMA_URL
  LLM_PROVIDER_ORDER (default "groq,gemini,ollama")  LLM_MAX_RPM (20)
  LLM_SESSION_BUDGET (20)  LLM_CACHE_PATH
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import requests
from pydantic import BaseModel, ValidationError

from llm.cache import LLMCache, canonical_json, make_key
from llm.limiter import TokenBucket
from llm.privacy import Sanitizer
from llm.tasks import TASKS, TaskSpec


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Base class: callers catch this and fall back to rules."""


class LLMUnavailable(LLMError):
    """No provider could answer (none configured, all failed, budget or rate limit reached)."""


class LLMInvalidOutput(LLMError):
    """A provider answered, but not with valid JSON of the required shape (after one retry)."""


class PayloadRejected(ValueError):
    """The caller tried to send something that looks like raw data. A bug, not an outage."""


class ProviderError(Exception):
    """One failed provider request. `retryable` decides whether the gateway retries it."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None,
                 retry_after: float | None = None, daily: bool = False, permanent: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after
        self.daily = daily  # a per-day quota: retrying today is pointless
        self.permanent = permanent  # auth / unknown model: skip this provider for the session


# --------------------------------------------------------------------------- #
# Payload guard ("never send raw rows")
# --------------------------------------------------------------------------- #
FORBIDDEN_KEYS = frozenset({"rows", "raw_rows", "records", "data", "dataframe", "df", "csv", "cells", "table"})
SAMPLE_KEYS = frozenset({"samples", "sample_values"})
MAX_SAMPLES = 8
MAX_STRING_CHARS = 500
MAX_PAYLOAD_CHARS = 20_000


def check_payload(payload: Any) -> None:
    """Raise PayloadRejected unless `payload` is a compact summary rather than data."""

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if str(k).lower() in FORBIDDEN_KEYS:
                    raise PayloadRejected(f"payload key '{path}{k}' looks like raw data; send summaries only")
                if str(k).lower() in SAMPLE_KEYS and isinstance(v, (list, tuple)) and len(v) > MAX_SAMPLES:
                    raise PayloadRejected(f"'{path}{k}' has {len(v)} sample values; the limit is {MAX_SAMPLES}")
                walk(v, f"{path}{k}.")
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, path)
        elif isinstance(node, str) and len(node) > MAX_STRING_CHARS:
            raise PayloadRejected(f"a string under '{path}' is {len(node)} chars long; looks like raw content")

    walk(payload, "")
    if len(canonical_json(payload)) > MAX_PAYLOAD_CHARS:
        raise PayloadRejected(f"payload exceeds {MAX_PAYLOAD_CHARS} characters; summarise it")


# --------------------------------------------------------------------------- #
# Providers (plain HTTP via `requests`; `post` is injectable so tests never hit the network)
# --------------------------------------------------------------------------- #
@dataclass
class ProviderReply:
    text: str
    model: str
    truncated: bool = False


_DAILY_HINTS = ("per day", "tokens per day", "requests per day", "tpd", "rpd", "daily")


class Provider:
    name = "base"

    def __init__(self, *, post: Callable[..., Any] = requests.post, timeout: float = 30.0):
        self._post_fn = post
        self.timeout = timeout

    def is_configured(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def complete(self, system: str, user: str, slot: str, max_tokens: int) -> ProviderReply:  # pragma: no cover
        raise NotImplementedError

    # -- shared HTTP handling ------------------------------------------------------
    def _request(self, url: str, headers: dict, body: dict) -> dict:
        try:
            resp = self._post_fn(url, headers=headers, json=body, timeout=self.timeout)
        except requests.Timeout:
            raise ProviderError(f"{self.name}: request timed out", retryable=True) from None
        except requests.ConnectionError:
            raise ProviderError(f"{self.name}: connection failed", retryable=True) from None
        except requests.RequestException as exc:
            raise ProviderError(f"{self.name}: {type(exc).__name__}", retryable=True) from None

        status = resp.status_code
        if status == 200:
            try:
                return resp.json()
            except ValueError:
                raise ProviderError(f"{self.name}: response was not JSON", retryable=True, status=status) from None

        snippet = (getattr(resp, "text", "") or "")[:200]
        if status == 429:
            daily = any(h in snippet.lower() for h in _DAILY_HINTS)
            return self._raise(ProviderError(
                f"{self.name}: rate limited (429)", retryable=not daily, status=status,
                retry_after=_retry_after(resp), daily=daily))
        if status in (408, 409, 425) or status >= 500:
            raise ProviderError(f"{self.name}: server error ({status})", retryable=True, status=status,
                                retry_after=_retry_after(resp))
        if status in (401, 403, 404):
            raise ProviderError(f"{self.name}: rejected the request ({status}); check the key and model name",
                                status=status, permanent=True)
        raise ProviderError(f"{self.name}: request failed ({status})", status=status)

    @staticmethod
    def _raise(err: ProviderError) -> dict:
        raise err


def _retry_after(resp: Any) -> float | None:
    try:
        value = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


class GroqProvider(Provider):
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, small_model: str, large_model: str, **kw: Any):
        super().__init__(**kw)
        self.api_key, self.small_model, self.large_model = api_key, small_model, large_model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, slot: str, max_tokens: int) -> ProviderReply:
        model = self.large_model if slot == "large" else self.small_model
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if model.startswith("openai/gpt-oss"):
            # Reasoning models spend max_tokens on hidden reasoning first; keep it low.
            body["reasoning_effort"] = "low"
        data = self._request(self.URL, {"Authorization": f"Bearer {self.api_key}"}, body)
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("groq: unexpected response shape", retryable=True) from None
        if not text.strip():
            raise ProviderError("groq: empty answer", retryable=True)
        return ProviderReply(text, model, truncated=choice.get("finish_reason") == "length")


class GeminiProvider(Provider):
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, **kw: Any):
        super().__init__(**kw)
        self.api_key, self.model = api_key, model

    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    def complete(self, system: str, user: str, slot: str, max_tokens: int) -> ProviderReply:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"},
        }
        data = self._request(self.URL.format(model=self.model), {"x-goog-api-key": self.api_key}, body)
        try:
            cand = data["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("gemini: no usable answer (blocked or empty)", retryable=False) from None
        if not text.strip():
            raise ProviderError("gemini: empty answer", retryable=True)
        return ProviderReply(text, self.model, truncated=cand.get("finishReason") == "MAX_TOKENS")


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kw: Any):
        super().__init__(**kw)
        self.model, self.base_url = model, base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.model)

    def complete(self, system: str, user: str, slot: str, max_tokens: int) -> ProviderReply:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        data = self._request(f"{self.base_url}/api/chat", {}, body)
        try:
            text = data["message"]["content"] or ""
        except (KeyError, TypeError):
            raise ProviderError("ollama: unexpected response shape", retryable=True) from None
        if not text.strip():
            raise ProviderError("ollama: empty answer", retryable=True)
        return ProviderReply(text, self.model, truncated=data.get("done_reason") == "length")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class GatewayConfig:
    provider_order: list[str] = field(default_factory=lambda: ["groq", "gemini", "ollama"])
    groq_api_key: str = ""
    groq_small_model: str = "openai/gpt-oss-20b"
    groq_large_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    ollama_model: str = ""
    ollama_url: str = "http://localhost:11434"
    max_rpm: float = 20.0
    session_budget: int = 20
    cache_path: str = os.path.join(".cache", "llm_cache.sqlite")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GatewayConfig":
        e = os.environ if env is None else env
        d = cls()
        order = [p.strip().lower() for p in e.get("LLM_PROVIDER_ORDER", "").split(",") if p.strip()]
        return cls(
            provider_order=order or d.provider_order,
            groq_api_key=e.get("GROQ_API_KEY", ""),
            groq_small_model=e.get("GROQ_SMALL_MODEL", d.groq_small_model),
            groq_large_model=e.get("GROQ_LARGE_MODEL", d.groq_large_model),
            gemini_api_key=e.get("GEMINI_API_KEY", ""),
            gemini_model=e.get("GEMINI_MODEL", d.gemini_model),
            ollama_model=e.get("OLLAMA_MODEL", ""),
            ollama_url=e.get("OLLAMA_URL", d.ollama_url),
            max_rpm=float(e.get("LLM_MAX_RPM", d.max_rpm)),
            session_budget=int(e.get("LLM_SESSION_BUDGET", d.session_budget)),
            cache_path=e.get("LLM_CACHE_PATH", d.cache_path),
        )

    def build_providers(self, **kw: Any) -> list[Provider]:
        made: dict[str, Provider] = {
            "groq": GroqProvider(self.groq_api_key, self.groq_small_model, self.groq_large_model, **kw),
            "gemini": GeminiProvider(self.gemini_api_key, self.gemini_model, **kw),
            "ollama": OllamaProvider(self.ollama_model, self.ollama_url, **kw),
        }
        return [made[n] for n in self.provider_order if n in made and made[n].is_configured()]


# --------------------------------------------------------------------------- #
# Gateway
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_RETRY_NUDGE = ("\n\nYour previous reply was not valid JSON of the required shape. "
                "Reply with ONLY the JSON object, no prose, no code fences.")


class Gateway:
    """One instance per user session (it owns that session's call counter and budget)."""

    def __init__(
        self,
        providers: list[Provider] | None = None,
        *,
        cache: LLMCache | None = None,
        limiter: TokenBucket | None = None,
        session_budget: int = 20,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        max_retry_wait: float = 20.0,
        max_limiter_wait: float = 30.0,
        cooldown_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        sanitizer: Sanitizer | None = None,
    ):
        self.providers = list(providers or [])
        self.cache = cache if cache is not None else LLMCache(":memory:")
        self.limiter = limiter if limiter is not None else TokenBucket(20.0, sleep=sleep, clock=clock)
        self.session_budget = session_budget
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.max_retry_wait = max_retry_wait
        self.max_limiter_wait = max_limiter_wait
        self.cooldown_s = cooldown_s
        self._sleep = sleep
        self._clock = clock
        self.sanitizer = sanitizer if sanitizer is not None else Sanitizer()
        self._skip_until: dict[str, float] = {}
        # counters
        self.calls_used = 0  # logical calls that reached a provider (cache hits excluded)
        self.cache_hits = 0
        self.provider_requests = 0  # HTTP requests, including retries and validation retries
        self.invalid_retries = 0
        self.by_provider: dict[str, int] = {}

    @classmethod
    def from_config(cls, cfg: GatewayConfig | None = None, **kw: Any) -> "Gateway":
        cfg = cfg or GatewayConfig.from_env()
        return cls(
            cfg.build_providers(),
            cache=LLMCache(cfg.cache_path),
            limiter=TokenBucket(cfg.max_rpm),
            session_budget=cfg.session_budget,
            **kw,
        )

    # -- public ---------------------------------------------------------------------
    def call(self, task: str, payload: dict) -> BaseModel:
        spec = TASKS.get(task)
        if spec is None:
            raise ValueError(f"Unknown LLM task '{task}'. Known: {', '.join(TASKS)}")
        payload = self.sanitizer.clean(payload)  # personal values never reach a prompt or a cache key
        check_payload(payload)

        key = make_key(spec.name, spec.prompt_version, payload)
        cached = self.cache.get(key)
        if cached is not None:
            try:
                model = spec.response_model.model_validate_json(cached)
            except ValidationError:
                model = None  # written under an older schema: recompute
            if model is not None:
                self.cache_hits += 1
                return model

        usable = self._usable_providers()
        if not usable:
            raise LLMUnavailable(self._why_unavailable())
        if self.calls_used >= self.session_budget:
            raise LLMUnavailable(f"The session limit of {self.session_budget} AI calls has been reached.")
        self.calls_used += 1

        user = canonical_json(payload)
        failures: list[str] = []
        for prov in usable:
            try:
                return self._answer_with(prov, spec, user, key)
            except ProviderError as exc:
                failures.append(str(exc))
        raise LLMUnavailable("No AI provider could answer: " + "; ".join(failures))

    @property
    def usage_label(self) -> str:
        return f"LLM calls used: {self.calls_used}"

    def stats(self) -> dict[str, Any]:
        return {
            "calls_used": self.calls_used, "session_budget": self.session_budget,
            "cache_hits": self.cache_hits, "provider_requests": self.provider_requests,
            "invalid_retries": self.invalid_retries, "by_provider": dict(self.by_provider),
            "providers": [p.name for p in self.providers],
            "privacy_backend": self.sanitizer.backend,
            "personal_columns_withheld": sorted(self.sanitizer.columns_withheld),
            "values_redacted": self.sanitizer.redactions,
        }

    # -- internals --------------------------------------------------------------------
    def _usable_providers(self) -> list[Provider]:
        now = self._clock()
        return [p for p in self.providers if self._skip_until.get(p.name, 0.0) <= now]

    def _why_unavailable(self) -> str:
        if not self.providers:
            return "No AI provider is configured (set GROQ_API_KEY, GEMINI_API_KEY or OLLAMA_MODEL)."
        return "All AI providers are temporarily unavailable; using rule-based proposals."

    def _bench(self, prov: Provider, seconds: float) -> None:
        self._skip_until[prov.name] = self._clock() + seconds

    def _answer_with(self, prov: Provider, spec: TaskSpec, user: str, key: str) -> BaseModel:
        reply = self._request_with_retry(prov, spec, user, spec.max_tokens)
        for attempt in (1, 2):
            try:
                model = spec.response_model.model_validate_json(_FENCE.sub("", reply.text.strip()))
            except ValidationError as exc:
                if attempt == 2:
                    raise LLMInvalidOutput(
                        f"{prov.name} did not return valid JSON for '{spec.name}' after one retry.") from None
                self.invalid_retries += 1
                budget = spec.max_tokens * 2 if reply.truncated else spec.max_tokens
                reply = self._request_with_retry(prov, spec, user + _RETRY_NUDGE, budget)
                continue
            self.cache.put(key, spec.name, model.model_dump_json(), prov.name, reply.model)
            self.by_provider[prov.name] = self.by_provider.get(prov.name, 0) + 1
            return model
        raise AssertionError("unreachable")  # pragma: no cover

    def _request_with_retry(self, prov: Provider, spec: TaskSpec, user: str, max_tokens: int) -> ProviderReply:
        delay = self.backoff_base
        for attempt in range(1, self.max_attempts + 1):
            if not self.limiter.acquire(max_wait=self.max_limiter_wait):
                raise LLMUnavailable("Local rate limit reached; try again in a moment.")
            self.provider_requests += 1
            try:
                return prov.complete(spec.system, user, spec.slot, max_tokens)
            except ProviderError as exc:
                if exc.permanent:
                    self._bench(prov, 10 ** 9)  # bad key / unknown model: skip for the whole session
                    raise
                if exc.daily:
                    self._bench(prov, 6 * 3600)  # per-day quota: do not retry or hammer
                    raise
                if not exc.retryable or attempt == self.max_attempts:
                    self._bench(prov, self.cooldown_s)
                    raise
                wait = exc.retry_after if exc.retry_after is not None else delay
                if wait > self.max_retry_wait:  # provider wants a long pause: use someone else
                    self._bench(prov, wait)
                    raise
                self._sleep(wait)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Module-level entry point used by the whole app
# --------------------------------------------------------------------------- #
_default: Gateway | None = None


def get_default_gateway() -> Gateway:
    global _default
    if _default is None:
        _default = Gateway.from_config()
    return _default


def reset_default_gateway() -> None:
    global _default
    _default = None


def call_llm(task: str, payload: dict, *, gateway: Gateway | None = None) -> BaseModel:
    """The single function the app uses. Pass the session's own `gateway` so that session's
    counter and budget apply (the Streamlit app keeps one in st.session_state)."""
    return (gateway or get_default_gateway()).call(task, payload)
