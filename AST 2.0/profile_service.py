"""
profile_service.py — pure logic behind two things the frontend needs:

    get_profile(csv_bytes, filename)         — raw (pre-cleaning) stats +
                                                 schema manifest + preview,
                                                 shown immediately after upload
    get_cleaning_report(csv_bytes, filename)  — runs rule_based_clean(),
                                                 returns a before/after
                                                 report so cleaning is a
                                                 visible, auditable step
                                                 instead of a silent one

No FastAPI dependency, mirrors analysis_service.py / rag_service.py's
split from api.py. Unlike those two, this module has NO streamlit
dependency at all — profiler.py's only use of streamlit is a
@st.cache_data decorator we don't even call here (we compute stats
directly instead, since we also need things it doesn't return: friendly
per-column dtype labels and a row preview). Worth knowing if you're
tracing why this file behaves differently from the other two under
testing — this one can be fully exercised without streamlit installed.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def _friendly_dtype(dtype) -> str:
    """Map a pandas dtype to a short display label matching the
    original Streamlit app's schema manifest convention — 'object'
    reads as 'str' to anyone not already fluent in pandas internals."""
    s = str(dtype)
    return "str" if s == "object" else s


def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.where(pd.notnull(df), None)
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _read_csv(csv_bytes: bytes, filename: str) -> pd.DataFrame:
    # Was a standalone utf-8/latin1-only CSV parser duplicated across 4
    # files (this being one), which is exactly how "no Excel support"
    # went unnoticed for so long: fixing one copy didn't fix the others.
    # Now delegates to file_reader.py, the single shared reader.
    from file_reader import read_tabular_file
    return read_tabular_file(csv_bytes, filename)


def _schema(df: pd.DataFrame) -> list[dict]:
    null_counts = df.isnull().sum()
    return [
        {"column": col, "dtype": _friendly_dtype(df[col].dtype), "nulls": int(null_counts[col])}
        for col in df.columns
    ]


def get_profile(csv_bytes: bytes, filename: str, preview_rows: int = 10) -> dict:
    """Raw, pre-cleaning profile — matches what a user should see the
    instant a file finishes uploading, before anything has been altered."""
    try:
        df = _read_csv(csv_bytes, filename)
    except Exception as e:
        return {"success": False, "error": f"Could not parse CSV: {e}"}

    try:
        return {
            "success": True,
            "filename": filename,
            "totalRows": len(df),
            "totalColumns": len(df.columns),
            "numericCols": len(df.select_dtypes(include="number").columns),
            "nullValues": int(df.isnull().sum().sum()),
            "schema": _schema(df),
            "preview": _records(df.head(preview_rows)),
            "previewCount": min(preview_rows, len(df)),
        }
    except Exception as e:
        return {"success": False, "error": f"Profiling failed: {e}"}


def get_cleaning_report(csv_bytes: bytes, filename: str, preview_rows: int = 10) -> dict:
    """
    Runs rule_based_clean() once and returns a before/after report: the
    human-readable log line cleaning.py already generates for each action
    it took, plus before/after row/column/null counts and a preview of
    the cleaned data. This is a REPORT for user visibility — it does not
    persist the cleaned dataframe anywhere. Queries in analysis_service.py
    still run rule_based_clean() again themselves when a query executes
    (cheap, deterministic, idempotent) — this endpoint exists so the user
    can see and trust what cleaning will do before/while querying, not to
    change how querying itself works.
    """
    try:
        df_raw = _read_csv(csv_bytes, filename)
    except Exception as e:
        return {"success": False, "error": f"Could not parse CSV: {e}"}

    try:
        from cleaning import rule_based_clean

        df_clean, log = rule_based_clean(df_raw)

        return {
            "success": True,
            "filename": filename,
            "log": log,
            "before": {
                "rows": len(df_raw),
                "columns": len(df_raw.columns),
                "nullValues": int(df_raw.isnull().sum().sum()),
            },
            "after": {
                "rows": len(df_clean),
                "columns": len(df_clean.columns),
                "nullValues": int(df_clean.isnull().sum().sum()),
            },
            "schema": _schema(df_clean),
            "preview": _records(df_clean.head(preview_rows)),
            "previewCount": min(preview_rows, len(df_clean)),
        }
    except Exception as e:
        return {"success": False, "error": f"Cleaning failed: {e}"}
