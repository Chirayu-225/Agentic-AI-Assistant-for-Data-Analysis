"""
agent_verify.py — check, in the backend, what the data-quality agent's fix script did to the table.

The agent asks a model for ONE pandas script and runs it in the sandbox. A model can blank real values, fill missing
ones with "Unknown", rename "Planes" to "Aircraft", drop rows. This module decides, from the two tables themselves, what
is allowed to survive. It never trusts anything the fix script reports.

    verify_fix(before_df, after_df) -> {
        "ok":               bool   the fix may be used (False: structural failure or verifier fault; table unchanged)
        "structural":       bool   the fix changed the table's structure (rows, order, index, or dropped/renamed a column)
        "violations":       [{"kind", "column", "count", "example"}]   what was found; kind is one of
                            "structure", "filled", "blanked", "changed", "name_clash"
        "reverted_columns": [name]  columns put back to the original because they violated a rule
        "kept_changes":     [{"column", "count", "kind"}]  changes that survived: "tidied" (the same value written
                            more cleanly; count = cells) or "added" (a new column; count = rows)
        "new_columns":      [name]  columns the fix added
        "repaired":         DataFrame  what to use: the original with the compliant changes applied (the original,
                            unchanged, when ok is False)
        "fault":            bool   the global tripwire fired: a verifier bug, not a model violation
    }

STRUCTURE FIRST (any failure discards the whole fix): same row count, same index, same row order, every original column
present under the same name (a rename is a drop plus an add). A new column whose name equals an existing column is a
violation of that column ("name_clash"): the extra column is dropped.

PER COLUMN, by row position, nulls equal nulls: before null and after not null = "filled"; before not null and after
null = "blanked"; both non-null but not equivalent = "changed". A violating column is reverted to the original;
compliant columns and new columns stay.

EQUIVALENT means the same value written differently:
  * equal after trimming, collapsing internal whitespace and case-folding. A change of CASE counts only in a
    low-cardinality text column (at most max(20, 5% of the rows) distinct values after trimming); elsewhere it is a
    change (product codes and ids are never lowercased);
  * numerically equal within 1e-9 ("1" and 1.0), only when the before value has no leading zero (a value matching
    ^[+-]?0\\d is an identifier: "007" -> 7 is a change);
  * the same datetime, only when the before text contains a digit and is not just a number.
A dtype coercion that changes any value is a change.

GLOBAL TRIPWIRE: after the per-column reverts, the null count over the ORIGINAL columns must not rise. If it does, the
verifier itself is wrong: the whole fix is discarded and fault is True.

Vectorised, fast on 100,000 rows, inputs never mutated. Pure: pandas and numpy only.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

EXAMPLE_MAX = 60
LOW_CARDINALITY_FLOOR = 20
LOW_CARDINALITY_PCT = 5
NUMERIC_TOLERANCE = 1e-9

KIND_STRUCTURE, KIND_FILLED, KIND_BLANKED, KIND_CHANGED, KIND_NAME_CLASH = (
    "structure", "filled", "blanked", "changed", "name_clash")

_LEADING_ZERO = re.compile(r"^[+-]?0\d")
_WHITESPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_NAMED_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape(value) -> str:
    """Printable, single-line, at most EXAMPLE_MAX characters."""
    text = _CONTROL.sub(lambda m: _NAMED_ESCAPES.get(m.group(0), f"\\x{ord(m.group(0)):02x}"), str(value))
    return text if len(text) <= EXAMPLE_MAX else text[: EXAMPLE_MAX - 3] + "..."


def _violation(kind: str, column, count: int, example) -> dict:
    return {"kind": kind, "column": column, "count": int(count), "example": _escape(example)}


def _norm_text(values: np.ndarray) -> pd.Series:
    """Trimmed text with internal whitespace collapsed."""
    s = pd.Series(values, dtype=object).astype(str).str.strip()
    return s.str.replace(_WHITESPACE, " ", regex=True)


def _is_low_cardinality_text(before: pd.Series, n_rows: int) -> bool:
    if (pd.api.types.is_numeric_dtype(before) or pd.api.types.is_bool_dtype(before)
            or pd.api.types.is_datetime64_any_dtype(before)):
        return False
    distinct = before.dropna().astype(str).str.strip().nunique()
    return distinct <= max(LOW_CARDINALITY_FLOOR, LOW_CARDINALITY_PCT * n_rows // 100)


def _exact_equal(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Element-wise equality of two object arrays of non-null values."""
    if len(b) == 0:
        return np.zeros(0, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = np.asarray(b == a)
            if out.dtype == bool and out.shape == b.shape:
                return out
        except Exception:                                     # noqa: BLE001 — fall back to the slow, safe path
            pass
    return np.fromiter((bool(x == y) for x, y in zip(b, a)), dtype=bool, count=len(b))


def _parse_datetimes(text: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(text, errors="coerce", format="mixed")
        except (TypeError, ValueError):                       # pandas < 2.0 has no format="mixed"
            return pd.to_datetime(text, errors="coerce")


def _equivalent(b_vals: np.ndarray, a_vals: np.ndarray, allow_case: bool) -> np.ndarray:
    """For cells that are not exactly equal (both non-null): True where `a` is the same value as `b` written
    differently."""
    b_text = _norm_text(b_vals)
    a_text = _norm_text(a_vals)
    ok = np.array((b_text == a_text).to_numpy(), dtype=bool)  # whitespace only (a writable copy: pandas 3 arrays are read-only)
    if allow_case:
        ok |= (b_text.str.casefold() == a_text.str.casefold()).to_numpy()
    todo = np.flatnonzero(~ok)
    if todo.size == 0:
        return ok
    bs = b_text.iloc[todo].reset_index(drop=True)
    as_ = a_text.iloc[todo].reset_index(drop=True)

    b_num = pd.to_numeric(bs, errors="coerce")
    a_num = pd.to_numeric(as_, errors="coerce")
    b_is_number = b_num.notna().to_numpy()
    identifier = bs.str.contains(_LEADING_ZERO, regex=True).to_numpy()          # "007": an id, not a quantity
    diff = np.abs(b_num.to_numpy(dtype=float) - a_num.to_numpy(dtype=float))
    num_ok = b_is_number & a_num.notna().to_numpy() & ~identifier & (diff <= NUMERIC_TOLERANCE)

    date_ok = np.zeros(len(bs), dtype=bool)
    maybe_date = ~b_is_number & bs.str.contains(r"\d", regex=True).to_numpy()   # a digit, and not just a number
    if maybe_date.any():
        where = np.flatnonzero(maybe_date)
        before_dt = _parse_datetimes(bs.iloc[where])
        after_dt = _parse_datetimes(as_.iloc[where])
        date_ok[where] = (before_dt.notna() & after_dt.notna() & (before_dt == after_dt)).to_numpy()

    ok[todo] = num_ok | date_ok
    return ok


def _compare_column(before: pd.Series, after: pd.Series, n_rows: int) -> dict:
    """-> {"filled", "blanked", "changed", "tidied": cell counts, "examples": {kind: text}}. The one function that
    decides, per column, what a fix did (tests replace it to simulate a verifier bug)."""
    b_null = before.isna().to_numpy()
    a_null = after.isna().to_numpy()
    b_vals = before.to_numpy(dtype=object)
    a_vals = after.to_numpy(dtype=object)
    filled = b_null & ~a_null
    blanked = ~b_null & a_null
    both = ~b_null & ~a_null
    exact = np.zeros(n_rows, dtype=bool)
    exact[both] = _exact_equal(b_vals[both], a_vals[both])
    differ = both & ~exact
    changed = np.zeros(n_rows, dtype=bool)
    tidied = np.zeros(n_rows, dtype=bool)
    if differ.any():
        idx = np.flatnonzero(differ)
        same = _equivalent(b_vals[idx], a_vals[idx], _is_low_cardinality_text(before, n_rows))
        tidied[idx[same]] = True
        changed[idx[~same]] = True
    examples = {}
    if filled.any():
        examples["filled"] = a_vals[int(np.flatnonzero(filled)[0])]
    if blanked.any():
        examples["blanked"] = b_vals[int(np.flatnonzero(blanked)[0])]
    if changed.any():
        i = int(np.flatnonzero(changed)[0])
        examples["changed"] = f"{b_vals[i]} -> {a_vals[i]}"
    return {"filled": int(filled.sum()), "blanked": int(blanked.sum()), "changed": int(changed.sum()),
            "tidied": int(tidied.sum()), "examples": examples}


def _empty_result(before_df: pd.DataFrame, **overrides) -> dict:
    result = {"ok": False, "structural": False, "violations": [], "reverted_columns": [], "kept_changes": [],
              "new_columns": [], "repaired": before_df.copy(), "fault": False}
    result.update(overrides)
    return result


def _structure_violations(before_df: pd.DataFrame, after_df: pd.DataFrame) -> list:
    if len(after_df) != len(before_df):
        return [_violation(KIND_STRUCTURE, None, abs(len(after_df) - len(before_df)),
                           f"{len(before_df)} rows -> {len(after_df)} rows")]
    if not after_df.index.equals(before_df.index):
        return [_violation(KIND_STRUCTURE, None, len(before_df), "the row index changed")]
    out = []
    remaining = list(after_df.columns)
    for name in before_df.columns:
        if name in remaining:
            remaining.remove(name)
        else:
            out.append(_violation(KIND_STRUCTURE, name, len(before_df), f"column {name!r} is missing or renamed"))
    return out


def _row_order_changed(before_df: pd.DataFrame, positions: list, after_df: pd.DataFrame) -> bool:
    """A column that holds the same values in a different sequence: rows were sorted or shuffled (or values swapped).
    Constant columns say nothing about order and are skipped."""
    for j, k in positions:
        b = before_df.iloc[:, j].to_numpy(dtype=object)
        a = after_df.iloc[:, k].to_numpy(dtype=object)
        b_null, a_null = np.asarray(pd.isna(b)), np.asarray(pd.isna(a))
        if b_null.sum() != a_null.sum():
            continue
        if (b_null == a_null).all() and _exact_equal(b[~b_null], a[~a_null]).all():
            continue                                                       # this column is identical
        b_sorted = np.sort(pd.Series(b[~b_null], dtype=object).astype(str).to_numpy())
        a_sorted = np.sort(pd.Series(a[~a_null], dtype=object).astype(str).to_numpy())
        if len(b_sorted) > 1 and b_sorted[0] != b_sorted[-1] and np.array_equal(b_sorted, a_sorted):
            return True
    return False


def verify_fix(before_df: pd.DataFrame, after_df: pd.DataFrame) -> dict:
    """See the module docstring."""
    structure = _structure_violations(before_df, after_df)
    if structure:
        return _empty_result(before_df, structural=True, violations=structure)

    n_rows = len(before_df)
    before_names = list(before_df.columns)
    # pair each original column with its place in the fix's table (the k-th occurrence of a name with the k-th)
    seen: dict = {}
    positions = []
    for j, name in enumerate(before_names):
        occurrences = [k for k, c in enumerate(after_df.columns) if c == name]
        n = seen.get(name, 0)
        seen[name] = n + 1
        positions.append((j, occurrences[n]))
    paired = {k for _, k in positions}

    if _row_order_changed(before_df, positions, after_df):
        return _empty_result(before_df, structural=True,
                             violations=[_violation(KIND_STRUCTURE, None, n_rows, "the row order changed")])

    violations: list = []
    reverted: list = []
    kept: list = []
    columns_out: list = []
    for j, k in positions:
        name = before_names[j]
        b, a = before_df.iloc[:, j], after_df.iloc[:, k]
        found = _compare_column(b, a, n_rows)
        bad = False
        for kind in (KIND_FILLED, KIND_BLANKED, KIND_CHANGED):
            if found[kind]:
                violations.append(_violation(kind, name, found[kind], found["examples"][kind]))
                bad = True
        if bad:
            reverted.append(name)
            columns_out.append(b)
        else:
            if found["tidied"]:
                kept.append({"column": name, "count": found["tidied"], "kind": "tidied"})
            columns_out.append(a)

    new_columns: list = []
    for k, name in enumerate(after_df.columns):
        if k in paired:
            continue
        if name in before_names:                                    # a new column that reuses an existing name
            violations.append(_violation(KIND_NAME_CLASH, name, n_rows, f"a new column reuses the name {name!r}"))
            if name not in reverted:
                reverted.append(name)
            continue
        new_columns.append(name)
        kept.append({"column": name, "count": n_rows, "kind": "added"})
        columns_out.append(after_df.iloc[:, k])

    repaired = pd.concat(columns_out, axis=1)
    repaired.columns = before_names + new_columns
    repaired.index = before_df.index

    # GLOBAL TRIPWIRE — over the ORIGINAL columns only (new flag columns may hold nulls)
    if int(repaired.iloc[:, : len(before_names)].isna().sum().sum()) > int(before_df.isna().sum().sum()):
        return _empty_result(before_df, fault=True, violations=violations)

    return {"ok": True, "structural": False, "violations": violations, "reverted_columns": reverted,
            "kept_changes": kept, "new_columns": new_columns, "repaired": repaired, "fault": False}


# ── plain-English text for the response (no exception text, no paths) ──────────────────────────────────────────

def describe_violation(v: dict) -> str:
    col, n = v["column"], v["count"]
    if v["kind"] == KIND_FILLED:
        return f"{col}: {n:,} missing values would have been filled in"
    if v["kind"] == KIND_BLANKED:
        return f"{col}: {n:,} existing values would have been blanked"
    if v["kind"] == KIND_CHANGED:
        return f"{col}: {n:,} existing values would have been changed"
    if v["kind"] == KIND_NAME_CLASH:
        return f"{col}: a new column reused this column's name"
    return "The suggested fix changed the table's rows or columns"


def build_message(result: dict) -> str:
    """The one-line summary shown to the user, generated from the verifier result."""
    n = len(result["kept_changes"])
    m = len(result["reverted_columns"])
    s = lambda count, word: f"{count} {word}{'s' if count != 1 else ''}"
    if result["fault"]:
        return "No data was changed. The safety check could not confirm the suggested fix, so it was not applied."
    if result["structural"]:
        return "No data was changed. The suggested fix would have altered the table's rows or columns, so it was discarded."
    if n and not m:
        return f"Applied {s(n, 'safe change')}, used for your questions"
    if n and m:
        return f"Applied {s(n, 'safe change')}; reverted {s(m, 'column')} that would have changed your values"
    if m:
        return f"No data was changed. {s(m, 'column')} would have changed your values, so the suggestions were not applied."
    return "No data was changed. The suggested fix made no safe changes."
