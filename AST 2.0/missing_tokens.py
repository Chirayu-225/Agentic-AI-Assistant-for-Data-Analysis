"""
missing_tokens.py — when is the text "NA" / "None" a missing value, and when is it a real value?

THE PROBLEM
  pandas' read_csv turns the literal text "NA" and "None" into missing values, and the cleaner's sentinel set blanks
  "na" / "none" in any case. In a TERRITORY column (EMEA / NA / APAC / Japan) "NA" is North America, a real value, and
  it was erased silently (1,074 of 2,823 rows reported as "missing").

THE RULE (one shared function, ambiguous_is_marker)
  The AMBIGUOUS tokens are "na" and "none", compared case-insensitively. In a column such an ambiguous token is a
  missing-value MARKER only when, among the values that are not missing and not themselves ambiguous tokens (trimmed),
  at least 95% parse as a number (pd.to_numeric) or as a datetime where the value contains at least one digit (at most
  2,000 values are sampled, with a fixed seed). An empty remainder keeps the token. Otherwise the token is a real VALUE.
  Every other pandas default token ("", "N/A", "n/a", "#N/A", "NULL", "null", "NaN", "nan", "<NA>", "-NaN", ...) and
  every other cleaner sentinel stays ALWAYS missing; this module never touches them.

THE RESTORE (restore_ambiguous_tokens) — the reader's second pass
  The caller reads the CSV exactly as before. Then ONLY for candidate columns (not numeric / bool / datetime /
  timedelta, and with missing values) it re-reads those columns as raw strings (the caller passes a function that does
  that with the SAME kwargs as the first read) and puts the text back where the first read gave missing AND the raw
  token is exactly "NA" or "None" AND the column is not marker-context. A column that pandas read as entirely missing
  (all "NA") is a candidate too, since an all-NA column has no numbers to be numeric about. No dtype is converted.
  FAIL CLOSED: before restoring, the second read must have the same row count and column names, every cell that was
  missing in the first read must hold a token of pandas' default NA list (or the empty string), and every cell that was
  NOT missing must be equal to the first read. Any mismatch or exception returns the first-read frame unchanged, with
  the reason in RestoreResult.fallback_reason.

POLICIES
  "legacy"  (the default for library code): nothing in this module is used; behaviour is byte-for-byte what it was.
  "safe"    (APP_NA_POLICY, requested at ONE choke point: file_reader.read_tabular_file and
            cache_layer.cached_rule_clean).
"""

from __future__ import annotations

import warnings
from typing import Callable, NamedTuple, Optional

import numpy as np
import pandas as pd

LEGACY_NA_POLICY = "legacy"
APP_NA_POLICY = "safe"
NA_POLICIES = (LEGACY_NA_POLICY, APP_NA_POLICY)

# pandas' documented default NA list for read_csv (test_missing_tokens.py asserts this equals the installed pandas').
PANDAS_NA_TOKENS = frozenset({
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
})

# The cleaner's sentinel strings (cleaning._SENTINELS), lower case.
CLEANER_SENTINELS = frozenset({"nan", "none", "null", "n/a", "na", "--", "-", "", "nat"})

AMBIGUOUS_TOKENS = frozenset({"na", "none"})        # compared case-insensitively
RESTORABLE_TOKENS = ("NA", "None")                  # the only exact tokens the reader puts back

MARKER_SHARE_PCT = 95
MARKER_SAMPLE_SIZE = 2000
MARKER_SAMPLE_SEED = 0

_MISSING_LIKE = frozenset({t.lower() for t in PANDAS_NA_TOKENS}) | CLEANER_SENTINELS
_ALLOWED_AT_MISSING = frozenset(PANDAS_NA_TOKENS) | {""}


def check_policy(na_policy: str) -> str:
    if na_policy not in NA_POLICIES:
        raise ValueError(f"na_policy must be one of {NA_POLICIES}, not {na_policy!r}")
    return na_policy


# ── the shared rule ───────────────────────────────────────────────────────────────────────────────────────────────

def ambiguous_is_marker(values) -> bool:
    """True if an ambiguous token ("NA", "None", any case) in this column is a missing-value MARKER; False if it is a
    real VALUE. `values` is the column as it is (missing values, tokens and all); it is not modified."""
    s = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype=object)
    s = s[s.notna()]
    if s.empty:
        return False
    text = s.astype(str).str.strip()
    remainder = text[~text.str.lower().isin(_MISSING_LIKE)]
    if remainder.empty:
        return False
    if len(remainder) > MARKER_SAMPLE_SIZE:
        remainder = remainder.sample(MARKER_SAMPLE_SIZE, random_state=MARKER_SAMPLE_SEED)
    n = len(remainder)
    is_number = pd.to_numeric(remainder, errors="coerce").notna()
    parsed = int(is_number.sum())
    if 100 * parsed < MARKER_SHARE_PCT * n:
        rest = remainder[~is_number.to_numpy()]
        rest = rest[rest.str.contains(r"\d", regex=True)]            # a date must contain a digit ("May" is not a date)
        if len(rest):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    stamps = pd.to_datetime(rest, errors="coerce", format="mixed")
                except (TypeError, ValueError):                       # pandas < 2.0 has no format="mixed"
                    stamps = pd.to_datetime(rest, errors="coerce")
            parsed += int(stamps.notna().sum())
    return 100 * parsed >= MARKER_SHARE_PCT * n


# ── log text (one place, so the reader, the cleaner and the tests agree) ───────────────────────────────────────────

def marker_note(column, token: str, count: int) -> str:
    return f"✓ '{column}': {count:,} '{token}' treated as missing."


def kept_note(column, token: str, count: int) -> str:
    return f"✓ '{column}': kept {count:,} '{token}' values as real values."


# ── the reader's second pass ──────────────────────────────────────────────────────────────────────────────────────

class RestoreResult(NamedTuple):
    df: pd.DataFrame
    markers: dict                 # (column, token) -> cells left missing because the column is marker-context
    kept: dict                    # (column, token) -> cells put back as real values
    fallback_reason: Optional[str]


def candidate_columns(df: pd.DataFrame) -> list:
    """Columns that could hold a real "NA" / "None": not numeric / bool / datetime / timedelta, and with missing
    values. (A numeric column that is missing everywhere counts: it has no numbers to be numeric about.)"""
    out = []
    for col in df.columns:
        s = df[col]
        if not s.isna().any():
            continue
        typed = (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                 or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s))
        if typed and not s.isna().all():
            continue
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            continue
        out.append(col)
    return out


def restore_ambiguous_tokens(df: pd.DataFrame, reread: Callable[[list], pd.DataFrame]) -> RestoreResult:
    """See the module docstring. `reread(columns)` must return those columns of the same file as raw strings."""
    def fallback(reason: str) -> RestoreResult:
        return RestoreResult(df, {}, {}, reason)

    cands = candidate_columns(df)
    if not cands:
        return RestoreResult(df, {}, {}, None)
    if not all(isinstance(c, str) for c in df.columns) or len(set(df.columns)) != len(df.columns):
        return fallback("column names are not unique text")
    try:
        raw = reread(cands)
    except Exception as e:                                   # noqa: BLE001 — fail closed, whatever went wrong
        return fallback(f"the second read failed ({type(e).__name__})")
    if not isinstance(raw, pd.DataFrame):
        return fallback("the second read did not return a table")
    if len(raw) != len(df):
        return fallback(f"row count differs ({len(raw)} vs {len(df)})")
    if list(raw.columns) != list(cands):
        return fallback("column names differ")

    plan = []
    for col in cands:
        missing = df[col].isna().to_numpy()
        raw_vals = raw[col].to_numpy(dtype=object)
        if not pd.Series(raw_vals[missing], dtype=object).isin(_ALLOWED_AT_MISSING).all():
            return fallback(f"a missing cell in '{col}' does not hold a NA token in the second read")
        present = ~missing
        if present.any():
            first = pd.Series(df[col].to_numpy(dtype=object)[present]).astype(str).to_numpy()
            second = pd.Series(raw_vals[present], dtype=object).astype(str).to_numpy()
            if not np.array_equal(first, second):
                return fallback(f"values in '{col}' differ between the two reads")
        restorable = missing & pd.Series(raw_vals, dtype=object).isin(RESTORABLE_TOKENS).to_numpy()
        if restorable.any():
            plan.append((col, raw_vals, restorable))
    if not plan:
        return RestoreResult(df, {}, {}, None)

    out = df.copy(deep=False)
    markers: dict = {}
    kept: dict = {}
    for col, raw_vals, restorable in plan:
        counts = pd.Series(raw_vals[restorable], dtype=object).value_counts()
        if ambiguous_is_marker(df[col]):
            for token, n in counts.items():
                markers[(col, token)] = int(n)
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):                   # entirely missing: becomes text
            series = series.astype(raw[col].dtype)
        else:
            series = series.copy()
        series[restorable] = raw_vals[restorable]
        out[col] = series
        for token, n in counts.items():
            kept[(col, token)] = int(n)
    return RestoreResult(out, markers, kept, None)
