"""
core/profiler.py - per-column statistics. Deterministic, no LLM.

The profile is the only view of the data that ever leaves the machine (as compact
summaries + <= 8 sample values per column), so it is deliberately small.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.utils import is_datetime, is_numeric, is_text, null_mask, short

MAX_SAMPLES = 8
Kind = Literal["numeric", "bool", "datetime", "categorical", "text", "empty"]


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    kind: Kind
    n_rows: int
    null_count: int  # NaN/None/NaT plus blank or whitespace-only strings
    null_pct: float  # 0-100
    n_unique: int  # distinct non-null values
    unique_ratio: float  # n_unique / non-null count (0 when empty)
    looks_like_id: bool = False
    skew: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    top_values: list[tuple[str, int]] = Field(default_factory=list)  # up to 5
    sample_values: list[str] = Field(default_factory=list, max_length=MAX_SAMPLES)

    def signature(self) -> str:
        """Stable fingerprint of this column's profile, used as part of LLM cache keys."""
        payload = [
            self.name, self.dtype, self.kind, self.n_rows, self.null_count,
            self.n_unique, self.sample_values,
        ]
        return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()[:16]


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_cols: int
    duplicate_rows: int  # extra copies beyond the first occurrence
    columns: list[ColumnProfile]

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def signature(self) -> str:
        joined = "|".join(c.signature() for c in self.columns) + f"|{self.n_rows}|{self.duplicate_rows}"
        return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _num(x: object) -> float | None:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 6)


def _infer_kind(s: pd.Series, n_unique: int, non_null: int) -> Kind:
    if non_null == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(s.dtype):
        return "bool"
    if is_numeric(s):
        return "numeric"
    if is_datetime(s):
        return "datetime"
    ratio = n_unique / non_null
    if (n_unique <= 50 and ratio <= 0.5) or (n_unique <= 500 and ratio <= 0.05):
        return "categorical"
    return "text"


def _looks_like_id(s: pd.Series, kind: Kind, n_unique: int, non_null: int) -> bool:
    if non_null < 20 or n_unique / non_null < 0.95:
        return False
    if kind == "text":
        return True
    if kind == "numeric":
        arr = s.dropna().astype("float64")
        return bool((arr % 1 == 0).all())
    return False


def _profile_column(df: pd.DataFrame, name: str) -> ColumnProfile:
    s = df[name]
    mask = null_mask(s)
    nn = s[~mask]
    non_null = len(nn)
    try:
        counts = nn.value_counts(sort=False) if non_null else pd.Series(dtype="int64")
    except TypeError:  # unhashable cells (lists/dicts)
        counts = nn.astype(str).value_counts(sort=False)
    n_unique = int(len(counts))
    kind = _infer_kind(s, n_unique, non_null)

    ranked = counts.sort_values(ascending=False, kind="stable") if n_unique else counts
    top = [(short(v), int(c)) for v, c in ranked.head(5).items()]
    samples = [short(v) for v in ranked.head(MAX_SAMPLES).index]

    stats: dict[str, float | None] = dict.fromkeys(("skew", "mean", "median", "std", "min", "max"))
    if kind == "numeric" and non_null:
        x = nn.astype("float64")
        stats = {
            "skew": _num(x.skew()) if non_null >= 3 else None,
            "mean": _num(x.mean()),
            "median": _num(x.median()),
            "std": _num(x.std()) if non_null >= 2 else None,
            "min": _num(x.min()),
            "max": _num(x.max()),
        }

    return ColumnProfile(
        name=str(name),
        dtype=str(s.dtype),
        kind=kind,
        n_rows=len(df),
        null_count=int(mask.sum()),
        null_pct=round(100 * mask.sum() / len(df), 2) if len(df) else 0.0,
        n_unique=n_unique,
        unique_ratio=round(n_unique / non_null, 4) if non_null else 0.0,
        looks_like_id=_looks_like_id(s, kind, n_unique, non_null),
        top_values=top,
        sample_values=samples,
        **stats,
    )


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """Profile every column. Column names must be unique strings (read_csv guarantees this)."""
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported; rename them first.")
    if not all(isinstance(c, str) for c in df.columns):
        raise ValueError("All column names must be strings (pd.read_csv guarantees this).")
    try:
        dups = int(df.duplicated().sum())
    except TypeError:
        dups = 0
    return DatasetProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        duplicate_rows=dups,
        columns=[_profile_column(df, c) for c in df.columns],
    )
