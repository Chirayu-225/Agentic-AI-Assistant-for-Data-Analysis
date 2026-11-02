"""
llm_provider.py — Centralised LLM call layer.

Single source of truth for all model inference. Every other module calls
llm_code_call / llm_chat_call / llm_uncached_code_call from here.

PRIMARY: Groq (GROQ_API_KEY)
    CODE slot  → openai/gpt-oss-20b   (fast pandas/SQL generation)
    CHAT slot  → openai/gpt-oss-120b  (insights, RAG answers, follow-ups)
    Override with GROQ_CODE_MODEL / GROQ_CHAT_MODEL without a code change.

FALLBACK (optional): any OpenAI-compatible /chat/completions endpoint.
    Groq has retired this app's models twice (llama-3.1-8b-instant and
    llama-3.3-70b-versatile were shut down 2026-08-16; every request then
    404'd). A single provider is a single point of failure, so if the
    primary call fails and these are set, the same prompt is retried on
    the fallback:
        FALLBACK_LLM_BASE_URL   e.g. https://api.openai.com/v1
        FALLBACK_LLM_API_KEY
        FALLBACK_CODE_MODEL     model for the CODE slot
        FALLBACK_CHAT_MODEL     model for the CHAT slot (defaults to CODE)
    Uses plain HTTP via `requests`, so no extra SDK is needed.

DAILY CAP: LLM_DAILY_CALL_LIMIT (default 400) caps real model calls per
UTC day across ALL users, so a public demo can pause but can never drain
the Groq quota. Cached answers don't count. The count is kept in
logs/llm_usage.json so a restart doesn't reset it. Set the limit to 0 to
disable the cap (not recommended for anything public).

No Streamlit dependency: this module used to import streamlit for
st.cache_data and st.error, which made a UI library load-bearing inside
the FastAPI service. The code-generation cache is now a small in-process
TTL + LRU cache below, keyed on (model, prompt, temperature, session_id)
so one session's generated code is never returned to another session.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

_CODE_MODEL = os.environ.get("GROQ_CODE_MODEL", "openai/gpt-oss-20b")
_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")

_client = None

# Whether the most recent model call on this thread stopped because it hit
# its token budget (finish_reason == "length"). A cut-off answer must never
# be cached: code generation that ran out of room is missing its closing
# lines, and caching it made "try again" return the same broken answer.
_tls = threading.local()


def last_call_truncated() -> bool:
    return bool(getattr(_tls, "truncated", False))


def _get_client():
    """Return a cached Groq client. Raises a clear error if the key is missing."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the backend's .env "
            "(see .env.example). Get a key at https://console.groq.com"
        )
    from groq import Groq
    _client = Groq(api_key=api_key)
    return _client


def _call_groq(prompt: str, temperature: float, model: str, max_tokens: int) -> str:
    client = _get_client()
    kwargs = {}
    if model.startswith("openai/gpt-oss"):
        # gpt-oss are reasoning models: hidden reasoning tokens count against
        # max_tokens, so a small budget can be fully spent on reasoning and
        # leave an EMPTY answer. Ask for low effort (via extra_body so older
        # groq SDKs don't reject the argument).
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    choice = resp.choices[0]
    text = choice.message.content or ""
    _tls.truncated = choice.finish_reason == "length"
    if _tls.truncated:
        print(f"[llm] answer from {model} hit max_tokens={max_tokens} and was cut off", flush=True)
    if not text.strip():
        print(f"[llm] empty answer from {model} (finish_reason={choice.finish_reason}, "
              f"max_tokens={max_tokens})", flush=True)
        raise RuntimeError(
            f"Model returned an empty answer (finish_reason={choice.finish_reason}) — try again."
        )
    return text


def _fallback_model(slot: str) -> str | None:
    code = os.environ.get("FALLBACK_CODE_MODEL", "")
    if slot == "chat":
        return os.environ.get("FALLBACK_CHAT_MODEL", "") or code or None
    return code or None


def _call_fallback(prompt: str, temperature: float, slot: str, max_tokens: int) -> str:
    import requests
    base = os.environ.get("FALLBACK_LLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("FALLBACK_LLM_API_KEY", "")
    model = _fallback_model(slot)
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=90,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    _tls.truncated = choice.get("finish_reason") == "length"
    text = (choice["message"].get("content") or "")
    if not text.strip():
        raise RuntimeError(f"Fallback model {model} returned an empty answer.")
    return text


def fallback_configured() -> bool:
    return bool(os.environ.get("FALLBACK_LLM_BASE_URL") and os.environ.get("FALLBACK_LLM_API_KEY")
                and _fallback_model("code"))


# ── Daily usage cap ───────────────────────────────────────────────────────────

class DailyLimitReached(RuntimeError):
    """Raised when today's LLM call budget is spent."""


_usage_lock = threading.Lock()
_usage_path = os.environ.get(
    "LLM_USAGE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "llm_usage.json"),
)
_usage = {"day": "", "calls": 0}


def _daily_limit() -> int:
    try:
        return max(0, int(os.environ.get("LLM_DAILY_CALL_LIMIT", "400")))
    except ValueError:
        return 400


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_usage() -> None:
    try:
        with open(_usage_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("calls"), int):
            _usage.update(day=str(data.get("day", "")), calls=data["calls"])
    except (OSError, ValueError):
        pass  # missing or unreadable file = start counting from zero


def _save_usage() -> None:
    try:
        os.makedirs(os.path.dirname(_usage_path), exist_ok=True)
        tmp = _usage_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_usage, f)
        os.replace(tmp, _usage_path)
    except OSError:
        pass  # counting still works in memory if the disk is read-only


def _reserve_call() -> None:
    """Counts one model call against today's budget, or refuses it."""
    limit = _daily_limit()
    with _usage_lock:
        if not _usage["day"]:
            _load_usage()
        today = _today()
        if _usage["day"] != today:
            _usage.update(day=today, calls=0)
        if limit and _usage["calls"] >= limit:
            raise DailyLimitReached(
                "The daily AI usage limit for this demo has been reached. "
                "It resets at 00:00 UTC (05:30 IST)."
            )
        _usage["calls"] += 1
        _save_usage()


def usage_stats() -> dict:
    """For /admin/stats."""
    with _usage_lock:
        if not _usage["day"]:
            _load_usage()
        calls = _usage["calls"] if _usage["day"] == _today() else 0
    limit = _daily_limit()
    return {"day": _today(), "calls": calls, "limit": limit or None,
            "remaining": (limit - calls) if limit else None}


def _call(prompt: str, temperature: float, model: str, max_tokens: int = 2048) -> str:
    """Primary call, then the optional fallback provider. Raises only if
    every configured provider failed — callers handle that."""
    _reserve_call()
    slot = "chat" if model == _CHAT_MODEL else "code"
    try:
        return _call_groq(prompt, temperature, model, max_tokens)
    except Exception as primary_error:
        if not fallback_configured():
            raise
        print(f"[llm] primary ({model}) failed: {primary_error} — trying fallback provider", flush=True)
        try:
            return _call_fallback(prompt, temperature, slot, max_tokens)
        except Exception as fallback_error:
            raise RuntimeError(
                f"primary failed ({primary_error}); fallback failed ({fallback_error})"
            ) from fallback_error


# ── Code-generation cache (replaces st.cache_data) ────────────────────────────

_CACHE_TTL_SECONDS = 3600
_CACHE_MAX_ENTRIES = 512
_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(model: str, prompt: str, temperature: float, session_id: str, max_tokens: int = 2048) -> str:
    h = hashlib.sha256()
    for part in (model, f"{temperature:.3f}", session_id, str(max_tokens), prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def cached_llm_call(model: str, prompt_text: str, temperature: float, session_id: str = "",
                    max_tokens: int = 2048) -> str:
    """Identical (model, prompt, temperature, session_id) returns the cached
    answer for up to an hour. session_id is part of the key so two users
    asking the same question on identically-shaped data never receive each
    other's generated code. Set CACHE_ENABLED=0 to bypass."""
    if os.environ.get("CACHE_ENABLED", "1") == "0":
        return _call(prompt_text, temperature, model, max_tokens)
    key = _cache_key(model, prompt_text, temperature, session_id, max_tokens)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_SECONDS:
            _cache.move_to_end(key)
            return hit[1]
    _tls.truncated = False
    text = _call(prompt_text, temperature, model, max_tokens)
    if last_call_truncated():
        return text  # never cache an answer that was cut off
    with _cache_lock:
        _cache[key] = (now, text)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)
    return text


def llm_code_call(prompt_text: str, temperature: float = 0.1, session_id: str = "",
                  max_tokens: int = 2048) -> str:
    """Code generation (analysis, cleaning, retry). Cached per session."""
    try:
        return cached_llm_call(_CODE_MODEL, prompt_text, temperature, session_id, max_tokens)
    except Exception as e:
        raise RuntimeError(f"LLM code call failed: {e}") from e


def llm_chat_call(prompt_text: str, temperature: float = 0.3) -> str:
    """Natural-language work (insights, explanations, RAG answers). Never cached."""
    try:
        return _call(prompt_text, temperature, _CHAT_MODEL, max_tokens=4096)
    except Exception as e:
        raise RuntimeError(f"LLM chat call failed: {e}") from e


def llm_uncached_code_call(prompt_text: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """One-shot code calls that must NOT be cached (cleaning retry, error retry)."""
    try:
        return _call(prompt_text, temperature, _CODE_MODEL, max_tokens)
    except Exception as e:
        raise RuntimeError(f"LLM uncached code call failed: {e}") from e


def _clear_code_cache() -> None:
    with _cache_lock:
        _cache.clear()


# Same call shape st.cache_data exposed, so existing callers/tests that do
# cached_llm_call.clear() keep working.
cached_llm_call.clear = _clear_code_cache  # type: ignore[attr-defined]
