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
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from cleaning import rule_based_clean
from file_reader import read_tabular_file, should_use_large_csv_path, MAX_RESULT_ROWS
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


def _dataframe_to_records(df: pd.DataFrame, max_rows: int = MAX_RESULT_ROWS) -> tuple[list[dict], int, bool]:
    """Returns (records, total_row_count, truncated). Caps serialization
    at max_rows — see file_reader.MAX_RESULT_ROWS for why this exists:
    the sandbox already returned a real, correctly-computed DataFrame by
    the time this runs, so capping here only affects how much of an
    already-correct answer gets shipped over the wire, never the
    correctness of the computation itself."""
    total = len(df)
    truncated = total > max_rows
    if truncated:
        df = df.head(max_rows)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    return [{k: _json_safe_scalar(v) for k, v in row.items()} for row in records], total, truncated


def _shape_result(value: Any) -> dict:
    """Turn whatever SandboxResult.value is into the resultType/resultData
    shape the Next.js frontend expects (see the API contract in the
    Next.js repo's README). For a table result, also returns rowCount
    (rows actually returned) and totalRowCount/truncated so the frontend
    can show "showing 5,000 of 214,003 rows" instead of silently handing
    back a partial table with no indication anything was cut."""
    if isinstance(value, pd.DataFrame):
        records, total, truncated = _dataframe_to_records(value)
        return {
            "resultType": "table", "resultData": records,
            "rowCount": len(records), "totalRowCount": total, "truncated": truncated,
        }

    if isinstance(value, pd.Series):
        records, total, truncated = _dataframe_to_records(value.reset_index())
        return {
            "resultType": "table", "resultData": records,
            "rowCount": len(records), "totalRowCount": total, "truncated": truncated,
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


def _run_large_csv_analysis(csv_bytes: bytes, filename: str, query: str, session_id: str) -> dict:
    """Large-CSV counterpart to the pandas path below run_analysis().
    Never runs pd.read_csv() on the full file — a small sample (nrows)
    is read just to build the schema/sample text for the prompt, which
    is negligible regardless of the file's real size. The actual query
    runs against the whole file through sandbox_limits.
    run_large_csv_sql_sandboxed(), which uses DuckDB's own CSV engine
    inside the worker subprocess — see that function's docstring for
    why this is the real fix, not just a bigger memory cap."""
    from sandbox_limits import run_large_csv_sql_sandboxed
    from multi_table_service import _parse_sql_response  # same REASONING + ```python parser, no need to duplicate it
    from query_engine import build_large_csv_sql_prompt, llm_with_fallback

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "csv"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name

        # Schema-only sample — deliberately small and pandas-based; this
        # is the one place this path still touches pandas, and it's
        # bounded (nrows) regardless of the real file's size.
        try:
            sample_df = pd.read_csv(io.BytesIO(csv_bytes), nrows=20)
        except UnicodeDecodeError:
            sample_df = pd.read_csv(io.BytesIO(csv_bytes), nrows=20, encoding="latin1")

        cols = ", ".join(f"{c} ({sample_df[c].dtype})" for c in sample_df.columns)
        # AI SECURITY: these sample rows are REAL values from the user's
        # file and go straight into the prompt. Withhold columns named like
        # personal data, redact PII patterns in the rest, and fence the
        # block as untrusted data — this path previously skipped all three.
        from pii_redaction import mask_personal_columns, redact_pii_logged
        from prompt_injection_guard import wrap_untrusted
        sample_rows = mask_personal_columns(sample_df.head(3)).to_dict(orient="records")
        schema_text = f"  columns: {cols}\n  sample rows: {sample_rows}"
        schema_text = wrap_untrusted(
            redact_pii_logged(schema_text, context="large CSV sample rows"),
            source_label="large CSV schema and sample rows",
        )

        alias = "data"
        prompt = build_large_csv_sql_prompt(schema_text, query, alias=alias)
        raw = llm_with_fallback(prompt, temperature=0.1, session_id=session_id)
        _reasoning, code = _parse_sql_response(raw)

        if not code:
            return {
                "success": False, "resultType": "error", "resultData": None,
                "error": "Could not generate a SQL query for this question — try again.",
                "confidence": "LOW",
            }

        sbr = run_large_csv_sql_sandboxed(code, tmp_path, alias=alias, timeout_seconds=60)
    except Exception as e:
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": f"Large-file analysis pipeline failed: {e}", "confidence": "LOW",
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not sbr.success:
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": sbr.error, "confidence": "LOW",
        }

    try:
        shaped = _shape_result(sbr.value)
        confidence = _compute_confidence(sbr.value, sbr.success, shortcut_used=False)
    except Exception as e:
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


def _execute_query_against_df(df: pd.DataFrame, query: str, session_id: str) -> dict:
    """The second half of the pipeline — route query (shortcut or LLM) ->
    sandboxed exec -> shape result — split out from run_analysis() so a
    session_cache hit can jump straight here with an already-cleaned
    DataFrame, skipping the fetch + parse + clean that produced it the
    first time. This is the ONLY part of the pipeline that genuinely
    differs between two questions about the same upload; everything
    before it (fetch, parse, clean) is identical every time and is
    exactly what session_cache exists to avoid redoing.

    Imports query_engine/llm_provider lazily, inside the function, rather
    than at module load time. Both transitively require `streamlit`
    (llm_provider.py uses st.cache_data for the code-gen cache) even
    though this is a pure API service with no UI — lazy import means a
    missing/broken streamlit install fails loudly on first real request
    instead of crashing api.py at startup before you even get to see
    what's wrong.
    """
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


def run_analysis_from_cleaned_df(cleaned_df: pd.DataFrame, query: str, session_id: str) -> dict:
    """The cache-hit fast path: session_cache already has this exact
    (session, file) pair's cleaned DataFrame from a previous request, so
    there's nothing to fetch or parse or clean this time — just route
    and run the new question against it. api.py calls this instead of
    run_analysis() whenever session_cache.get() returns a hit, skipping
    the Blob fetch entirely on its side too."""
    return _execute_query_against_df(cleaned_df, query, session_id)


def run_analysis(csv_bytes: bytes, filename: str, query: str, session_id: str, file_url: str | None = None) -> dict:
    """
    The full pipeline: parse CSV -> clean -> route query (shortcut or LLM)
    -> sandboxed exec -> shape result.

    Returns a dict matching the /analyze response contract exactly:
        {success, resultType, resultData, error, confidence}

    Files above file_reader.LARGE_FILE_THRESHOLD_BYTES (CSV/TXT only —
    see should_use_large_csv_path) never reach pd.read_csv() at all;
    they're routed to _run_large_csv_analysis() before any parsing
    happens, which queries the file through DuckDB's own CSV engine
    instead. This decision is made on the raw bytes' size, so the
    "expensive" case (a huge file) is exactly the case that skips
    pandas — not a fallback for when pandas already failed. The large-
    file path is also deliberately NOT cached here — see session_cache.py's
    module docstring (SCOPE) for why.

    `file_url` is optional and used ONLY as a session_cache key — this
    function has no other use for it and doesn't fetch anything itself
    (api.py already did that). When provided, and the file takes the
    small/medium path, the cleaned DataFrame is cached so the NEXT
    request for this exact (session_id, file_url) can skip straight to
    run_analysis_from_cleaned_df() instead of landing here again. Callers
    that don't pass it (e.g. existing tests, or any call site that
    doesn't have a stable file_url to key on) simply don't get caching —
    not an error, just no caching for that call.
    """
    if should_use_large_csv_path(csv_bytes, filename):
        return _run_large_csv_analysis(csv_bytes, filename, query, session_id)

    try:
        df = read_tabular_file(csv_bytes, filename)
    except Exception as e:
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": f"Could not parse file: {e}", "confidence": "LOW",
        }

    import cache_layer
    df, _cleaning_log = cache_layer.cached_rule_clean(df, csv_bytes, rule_based_clean)

    if file_url:
        import session_cache
        session_cache.put(session_id, file_url, filename, csv_bytes, df)

    return _execute_query_against_df(df, query, session_id)
