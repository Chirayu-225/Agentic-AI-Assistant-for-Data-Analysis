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
import re

import numpy as np
import pandas as pd

UNREADABLE_FILE_ERROR = "This file could not be read as a table. Check that it is a valid CSV or Excel file."


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


def _pii_report(df) -> dict | None:
    """Never lets a scanning problem break profiling — the report is an
    extra, not part of the contract the rest of the profile depends on."""
    try:
        from pii_redaction import scan_dataframe
        return scan_dataframe(df)
    except Exception as e:
        print(f"[profile] PII scan failed (continuing without it): {e}", flush=True)
        return None


def get_profile(csv_bytes: bytes, filename: str, preview_rows: int = 10) -> dict:
    """Raw, pre-cleaning profile — matches what a user should see the
    instant a file finishes uploading, before anything has been altered."""
    try:
        df = _read_csv(csv_bytes, filename)
    except Exception as e:
        print(f"[profile-service] could not read file: {type(e).__name__}: {e}", flush=True)
        return {"success": False, "error": UNREADABLE_FILE_ERROR}

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
            # What the user sees BEFORE anything is sent to an LLM: which
            # columns look like personal data and how they will be treated.
            "piiReport": _pii_report(df),
        }
    except Exception as e:
        return {"success": False, "error": f"Profiling failed: {e}"}


_SIDE_COLUMN = re.compile(r".+_original(_\d+)?")


def _null_count_without_side_columns(df_clean: pd.DataFrame, df_raw: pd.DataFrame) -> int:
    """Nulls in the cleaned table, ignoring the '<col>_original' side columns the cleaner ADDS to keep the text of
    values it could not convert. Those are blank by design, not missing data. Same rule as cleaning.py: the name
    matches and its base column exists; a column that was already in the user's file is never ignored."""
    raw_cols = set(df_raw.columns)
    side = [c for c in df_clean.columns
            if c not in raw_cols and _SIDE_COLUMN.fullmatch(str(c))
            and str(c).rsplit("_original", 1)[0] in df_clean.columns]
    return int(df_clean.drop(columns=side).isnull().sum().sum()) if side else int(df_clean.isnull().sum().sum())


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
        print(f"[profile-service] could not read file: {type(e).__name__}: {e}", flush=True)
        return {"success": False, "error": UNREADABLE_FILE_ERROR}

    try:
        from cleaning import rule_based_clean

        from cache_layer import cached_rule_clean
        df_clean, log = cached_rule_clean(df_raw, csv_bytes, rule_based_clean)

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
                "nullValues": _null_count_without_side_columns(df_clean, df_raw),
            },
            "schema": _schema(df_clean),
            "preview": _records(df_clean.head(preview_rows)),
            "previewCount": min(preview_rows, len(df_clean)),
        }
    except Exception as e:
        return {"success": False, "error": f"Cleaning failed: {e}"}
