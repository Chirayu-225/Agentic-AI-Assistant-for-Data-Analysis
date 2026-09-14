"""
analysis_service.py — the actual analysis pipeline, with no HTTP/FastAPI
dependency at all.

WHY SPLIT FROM api.py
    Keeping this logic free of FastAPI/pydantic means it can be unit
    tested directly (see test_analysis_service.py) without needing a
    running server or any HTTP mocking — you feed it CSV bytes and a
    query, and get back the same dict api.py would return as JSON.
    api.py's job is reduced to: check the secret header, fetch the file
    bytes from the Blob URL, call run_analysis(), return the result.

WHAT THIS DOES NOT DO (yet)
    - No LLM-based secondary cleaning pass — only the deterministic
      rule_based_clean() runs automatically. The original Streamlit app's
      optional LLM cleaning refinement isn't wired in here yet; worth
      adding once the basic contract is confirmed working end-to-end.
    - matplotlib figures aren't converted to something the frontend can
      render — only Plotly figures are, since Next.js's ResultDisplay.tsx
      only speaks Plotly JSON right now. A matplotlib result comes back
      as a clear, explicit error rather than something broken silently.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd

from cleaning import rule_based_clean
from file_reader import read_tabular_file
from sandbox_limits import run_sandboxed


def _is_plotly_fig(obj: Any) -> bool:
    t = str(type(obj))
    return "plotly" in t and "Figure" in t


def _is_matplotlib_fig(obj: Any) -> bool:
    return hasattr(obj, "savefig") and hasattr(obj, "gca")


def _json_safe_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalar types to plain Python so json.dumps
    (or FastAPI's default JSON encoder) doesn't choke on them."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    return [{k: _json_safe_scalar(v) for k, v in row.items()} for row in records]


def _shape_result(value: Any) -> dict:
    """Turn whatever SandboxResult.value is into the resultType/resultData
    shape the Next.js frontend expects (see the API contract in the
    Next.js repo's README)."""
    if isinstance(value, pd.DataFrame):
        return {"resultType": "table", "resultData": _dataframe_to_records(value)}

    if isinstance(value, pd.Series):
        return {
            "resultType": "table",
            "resultData": _dataframe_to_records(value.reset_index()),
        }

    if _is_plotly_fig(value):
        return {"resultType": "chart", "resultData": json.loads(value.to_json())}

    if _is_matplotlib_fig(value):
        return {
            "resultType": "error",
            "resultData": None,
            "error": (
                "This result is a matplotlib chart, which isn't converted "
                "for the web frontend yet (only Plotly charts are). Try "
                "rephrasing the query, or ask for a table instead."
            ),
        }

    # scalar (int, float, str, bool, numpy scalar, None)
    return {"resultType": "scalar", "resultData": _json_safe_scalar(value)}


def _compute_confidence(value: Any, success: bool, shortcut_used: bool) -> str:
    """Same scoring approach as ui.py's compute_confidence(), reimplemented
    here without a streamlit import (this module intentionally has none)."""
    factors = 0
    if value is not None:
        factors += 1
    if success:
        factors += 1
    if shortcut_used:
        factors += 1
    if isinstance(value, pd.DataFrame) and len(value) > 0:
        factors += 1
    elif isinstance(value, pd.Series) and len(value) > 0:
        factors += 1
    elif isinstance(value, (int, float)) and value == value:  # NaN check
        factors += 1

    if factors >= 4:
        return "HIGH"
    if factors >= 2:
        return "MEDIUM"
    return "LOW"


def run_analysis(csv_bytes: bytes, filename: str, query: str, session_id: str) -> dict:
    """
    The full pipeline: parse CSV -> clean -> route query (shortcut or LLM)
    -> sandboxed exec -> shape result.

    Returns a dict matching the /analyze response contract exactly:
        {success, resultType, resultData, error, confidence}

    Imports query_engine/llm_provider lazily, inside the function, rather
    than at module load time. Both transitively require `streamlit`
    (llm_provider.py uses st.cache_data for the code-gen cache) even
    though this is a pure API service with no UI — lazy import means a
    missing/broken streamlit install fails loudly on first real request
    instead of crashing api.py at startup before you even get to see
    what's wrong.
    """
    try:
        df = read_tabular_file(csv_bytes, filename)
    except Exception as e:
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": f"Could not parse file: {e}", "confidence": "LOW",
        }

    df, _cleaning_log = rule_based_clean(df)

    try:
        from query_engine import (
            detect_shortcut,
            build_analysis_prompt,
            llm_with_fallback,
            extract_and_clean_code,
        )

        shortcut_code = detect_shortcut(query, df)
        shortcut_used = shortcut_code is not None

        if shortcut_used:
            code = shortcut_code
        else:
            schema_text = ", ".join(df.columns.tolist())
            prompt = build_analysis_prompt(schema_text, query)
            raw = llm_with_fallback(prompt, temperature=0.1, session_id=session_id)
            code = extract_and_clean_code(raw)

        # 25s, not sandbox_limits.py's 10s default: on Windows (spawn-only,
        # no fork), each sandboxed run cold-imports pandas/numpy/matplotlib/
        # plotly fresh in a new process — this overhead alone can approach
        # 10s locally. Render (Linux, fork) doesn't pay this cost, so
        # production should rarely need anywhere near this ceiling — this
        # is a local-dev accommodation, not evidence the pipeline is slow.
        sbr = run_sandboxed(code, df, timeout_seconds=25)
    except Exception as e:
        # Anything unexpected here (a bad LLM response shape, an import
        # failure, a query_engine bug) becomes a clean error response
        # instead of an unhandled 500 — this is a public API endpoint now,
        # not a Streamlit script with its own error boundary.
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": f"Analysis pipeline failed: {e}", "confidence": "LOW",
        }

    if not sbr.success:
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": sbr.error, "confidence": "LOW",
        }

    try:
        shaped = _shape_result(sbr.value)
        confidence = _compute_confidence(sbr.value, sbr.success, shortcut_used)
    except Exception as e:
        # sbr.value came out of the sandbox successfully, but shaping it
        # into JSON (e.g. fig.to_json() on a malformed Plotly figure)
        # failed. Same principle as the guard above — clean error, not
        # a raw 500, and now we know exactly which stage failed if this
        # message shows up.
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": f"Result shaping failed ({type(sbr.value).__name__}): {e}",
            "confidence": "LOW",
        }

    return {
        "success": shaped["resultType"] != "error",
        "confidence": confidence,
        "error": shaped.get("error"),
        **{k: v for k, v in shaped.items() if k != "error"},
    }
