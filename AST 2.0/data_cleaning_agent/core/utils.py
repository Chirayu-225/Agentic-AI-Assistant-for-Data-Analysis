"""
core/utils.py - small, dependency-free helpers shared by the deterministic core.

Everything here is pure pandas/numpy. No LLM imports, no I/O.

The single most important definition lives here: what counts as a *null*.
A cell is null when it is NaN/None/NaT **or** a text cell that is empty or
whitespace-only. Profiler, detectors and operations all use null_mask() so
they can never disagree about how many nulls a column has.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api import types as pdt

_WS_RUN = re.compile(r"\s+")
_NA = object()  # sentinel used for NA-aware comparisons


def is_text(s: pd.Series) -> bool:
    """True for string-like columns (object, string/str dtype, categorical of strings)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = s.cat.categories
        return pdt.is_string_dtype(cats.dtype) or cats.dtype == object
    return s.dtype == object or pdt.is_string_dtype(s.dtype)


def is_numeric(s: pd.Series) -> bool:
    """Real numbers only - booleans are deliberately excluded."""
    return pdt.is_numeric_dtype(s.dtype) and not pdt.is_bool_dtype(s.dtype)


def is_datetime(s: pd.Series) -> bool:
    return pdt.is_datetime64_any_dtype(s.dtype)


def null_mask(s: pd.Series) -> pd.Series:
    """Boolean mask of null cells, counting blank / whitespace-only strings as null."""
    mask = s.isna()
    if is_text(s):
        txt = s.astype("string")
        blank = txt.str.strip().eq("").fillna(False).astype(bool)
        mask = mask | blank
    return mask.astype(bool)


def string_values(s: pd.Series) -> pd.Series:
    """The non-null cells of `s` that are actual Python strings (original index kept)."""
    nn = s.dropna()
    if nn.dtype == object or isinstance(nn.dtype, pd.CategoricalDtype):
        nn = nn.astype(object)
        nn = nn[nn.map(lambda v: isinstance(v, str))]
    return nn.astype("string")


def clean_ws(v: str) -> str:
    """Trim and collapse internal whitespace runs to a single space."""
    return _WS_RUN.sub(" ", v.strip())


def str_map(s: pd.Series, fn: Callable[[str], object]) -> pd.Series:
    """Apply `fn` to the *string* cells of `s` only; everything else (NaN, numbers) is untouched.

    `fn` may return None to turn a cell into a null. The column keeps its dtype family
    (object stays object, string stays string, categorical stays categorical).
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        as_obj = s.astype(object).map(lambda v: fn(v) if isinstance(v, str) else v)
        return as_obj.astype("category")
    out = s.map(lambda v: fn(v) if isinstance(v, str) else v)
    if s.dtype != object:
        try:
            out = out.astype(s.dtype)
        except (TypeError, ValueError):
            pass
    return out


def changed_mask(before: pd.Series, after: pd.Series) -> pd.Series:
    """Cell-wise 'did this value change' mask. NA == NA, so null -> null is not a change."""
    if len(before) != len(after):
        raise ValueError("changed_mask needs equally long series")
    b = before.astype(object).to_numpy()
    a = after.astype(object).to_numpy()
    b = np.where(pd.isna(b), _NA, b)
    a = np.where(pd.isna(a), _NA, a)
    neq = np.fromiter((x != y for x, y in zip(b, a)), dtype=bool, count=len(b))
    return pd.Series(neq, index=before.index)


def _all_integral(x: pd.Series) -> bool:
    x = x.dropna()
    if x.empty:
        return True
    arr = x.astype("float64").to_numpy()
    return bool(np.isfinite(arr).all() and (np.mod(arr, 1) == 0).all())


def restore_integral(before: pd.Series, after: pd.Series) -> pd.Series:
    """If `before` held only whole numbers and `after` (a float column) still does,
    return `after` as nullable Int64 so a CSV shows 3 instead of 3.0."""
    if not pdt.is_float_dtype(after.dtype):
        return after
    if before.dropna().empty or not (is_numeric(before) and _all_integral(before)):
        return after
    if not _all_integral(after):
        return after
    return after.astype("Int64")


def short(v: object, width: int = 40) -> str:
    """Compact, JSON-safe rendering of a cell value (strings keep their quotes so
    leading/trailing spaces stay visible)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "<null>"
    text = repr(v) if isinstance(v, str) else str(v)
    return text if len(text) <= width else text[: width - 1] + "…"
