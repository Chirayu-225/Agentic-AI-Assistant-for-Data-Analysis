"""
llm_provider.py — Centralised LLM call layer.

Single source of truth for all model inference in Analyst Assistant.
Backed by Groq API — fast, free-tier-generous, production models only.

All other modules import from here instead of calling ollama directly.
Swapping providers in future = change this file only.

Environment variable required:
    GROQ_API_KEY  — get a free key at https://console.groq.com

Models used:
    CODE  → openai/gpt-oss-20b   (fast pandas/code gen)
    CHAT  → openai/gpt-oss-120b  (best reasoning for insights/RAG)

MODEL MIGRATION NOTE (2026-08-26)
    Originally llama-3.1-8b-instant / llama-3.3-70b-versatile. Groq
    announced deprecation of both on 2026-06-17 and shut them down
    2026-08-16 — after that date, every request returned a 404
    model_not_found error, which is what surfaced as "LLM code call
    failed: Error code: 404 ... model_not_found" in the app. Not a bug
    in this codebase — an upstream model retirement. Migrated to Groq's
    own recommended replacements (openai/gpt-oss-20b and
    openai/gpt-oss-120b respectively). If you ever see a 404
    model_not_found error again, check
    https://console.groq.com/docs/deprecations first — this is the
    second time Groq has retired a model this app depended on, so it's
    likely to happen again eventually.
"""

import os
import streamlit as st
from groq import Groq

# ── Model names ───────────────────────────────────────────────────────────────
# Split into two roles so each is optimised for its task:
#   CODE — speed matters most (code gen, cleaning, retries)
#   CHAT — quality matters most (business insights, RAG answers)
_CODE_MODEL = "openai/gpt-oss-20b"
_CHAT_MODEL = "openai/gpt-oss-120b"

# ── Client (lazy singleton) ───────────────────────────────────────────────────
_client: Groq | None = None


def _get_client() -> Groq:
    """Return a cached Groq client. Raises a clear error if key is missing."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error(
            "⚠ **GROQ_API_KEY not set.**  \n"
            "Add it to your `.env` file locally, or to Streamlit Cloud → "
            "App Settings → Secrets as `GROQ_API_KEY = \"gsk_...\"`.  \n"
            "Get a free key at https://console.groq.com",
            icon="🔑",
        )
        raise RuntimeError("GROQ_API_KEY environment variable not set.")

    _client = Groq(api_key=api_key)
    return _client


# ── Core call ─────────────────────────────────────────────────────────────────

def _call(
    prompt: str,
    temperature: float,
    model: str,
    max_tokens: int = 2048,
) -> str:
    """
    Single raw call to Groq. Returns the assistant message text.
    Raises on network/auth errors — callers handle fallback logic.
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ── Public interface — drop-in replacements for ollama.chat patterns ──────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_llm_call(model: str, prompt_text: str, temperature: float, session_id: str = "") -> str:
    """
    Cached wrapper — identical (model, prompt, temp, session_id) returns
    instantly. Mirrors the signature used in query_engine.py.

    session_id is folded into the cache key purely for isolation, not for
    lookup — st.cache_data's cache is process-wide (shared across every
    browser session hitting this server), not per-session. Without
    session_id in the key, two different users asking the same question
    on identically-shaped data could get back each other's cached
    generated code. Including it means each session gets its own cache
    entries, at the cost of losing cross-user cache hits — an acceptable
    trade once this runs on shared infrastructure instead of localhost.
    """
    return _call(prompt_text, temperature, model)


def llm_code_call(prompt_text: str, temperature: float = 0.1, session_id: str = "") -> str:
    """
    For code generation queries (analysis, cleaning, retry).
    Uses the code-optimised model slot.
    Falls back gracefully and surfaces error to the caller.

    Pass the caller's session_id (or eventual user_id) so the cache below
    can't return one session's generated code to a different session.
    """
    try:
        return cached_llm_call(_CODE_MODEL, prompt_text, temperature, session_id)
    except Exception as e:
        raise RuntimeError(f"LLM code call failed: {e}") from e


def llm_chat_call(prompt_text: str, temperature: float = 0.3) -> str:
    """
    For natural language queries (insights, explanations, RAG answers).
    Uses the chat/reasoning model slot.
    """
    try:
        return _call(prompt_text, temperature, _CHAT_MODEL, max_tokens=1024)
    except Exception as e:
        raise RuntimeError(f"LLM chat call failed: {e}") from e


def llm_uncached_code_call(prompt_text: str, temperature: float = 0.0) -> str:
    """
    For one-shot calls that must NOT be cached (cleaning retry, error retry).
    Always hits the API — avoids stale cached code for a different error context.
    """
    try:
        return _call(prompt_text, temperature, _CODE_MODEL)
    except Exception as e:
        raise RuntimeError(f"LLM uncached code call failed: {e}") from e