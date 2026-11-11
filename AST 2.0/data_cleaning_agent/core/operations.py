"""
core/operations.py - the CLOSED operation library.

* validate_params()   per-op parameter schemas (unknown params / ops are rejected)
* validate_against()  checks an Operation against a concrete DataFrame
                      (columns exist, dtypes make sense) BEFORE anything runs
* apply()             pure function: returns a NEW DataFrame, never mutates its input
* rows_affected()     how many rows an applied operation touched

No LLM imports. Every behaviour here is deterministic, so a recipe of operations
replays to the same bytes on the same input.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.schemas import OP_NAMES, Operation
from core.utils import (
    changed_mask,
    clean_ws,
    is_datetime,
    is_numeric,
    is_text,
    null_mask,
    restore_integral,
    str_map,
    string_values,
)

# Dropping more than this share of rows needs an extra explicit confirmation (executor).
ROW_DROP_CONFIRM_THRESHOLD = 0.05
MAX_MAPPING_ENTRIES = 1000

ROW_DROPPING_OPS = frozenset({"drop_duplicates", "drop_rows_where_null"})
DESTRUCTIVE_OPS = ROW_DROPPING_OPS | {"drop_column"}
# Ops that need `column`; the rest must NOT have one (they use params.subset instead).
COLUMN_OPS = frozenset(OP_NAMES) - ROW_DROPPING_OPS


class OperationError(ValueError):
    """Raised for anything that is not a valid, runnable operation. Messages are user-facing."""


# --------------------------------------------------------------------------- #
# Parameter schemas (one per op, all extra="forbid")
# --------------------------------------------------------------------------- #
class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DropDuplicatesParams(_Params):
    subset: list[str] | None = None
    keep: Literal["first", "last"] = "first"


class ImputeParams(_Params):
    strategy: Literal["mean", "median", "mode", "constant", "ffill"]
    value: str | int | float | bool | None = None
    group_by: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "ImputeParams":
        if self.strategy == "constant" and self.value is None:
            raise ValueError("strategy 'constant' needs a 'value'")
        if self.strategy != "constant" and self.value is not None:
            raise ValueError("'value' is only allowed with strategy 'constant'")
        return self


class CastTypeParams(_Params):
    dtype: Literal["int", "float", "str", "bool", "category"]
    errors: Literal["coerce", "raise"] = "coerce"


class ParseDatesParams(_Params):
    format: str | None = Field(default=None, max_length=40)
    dayfirst: bool = False
    errors: Literal["coerce", "raise"] = "coerce"

    @model_validator(mode="after")
    def _check(self) -> "ParseDatesParams":
        if self.format is not None and "%" not in self.format:
            raise ValueError("'format' must be a strftime pattern such as '%Y-%m-%d'")
        return self


class StripWhitespaceParams(_Params):
    collapse: bool = True  # also squeeze internal runs of whitespace to one space


class NormalizeCaseParams(_Params):
    mode: Literal["lower", "upper", "title"]


class ClipOutliersParams(_Params):
    method: Literal["iqr", "percentile", "manual"] = "iqr"
    k: float = Field(default=1.5, gt=0, le=10)  # IQR multiplier
    bounds: tuple[float | None, float | None] | None = None

    @model_validator(mode="after")
    def _check(self) -> "ClipOutliersParams":
        lo, hi = self.bounds if self.bounds else (None, None)
        if self.method == "iqr":
            if self.bounds is not None:
                raise ValueError("'bounds' is not used with method 'iqr' (use 'k')")
        elif self.method == "percentile":
            if self.bounds is None or lo is None or hi is None:
                raise ValueError("method 'percentile' needs bounds=[low_pct, high_pct]")
            if not (0 <= lo < hi <= 100):
                raise ValueError("percentile bounds must satisfy 0 <= low < high <= 100")
        else:  # manual
            if lo is None and hi is None:
                raise ValueError("method 'manual' needs at least one of the two bounds")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError("manual bounds must satisfy low <= high")
        return self


class MapValuesParams(_Params):
    mapping: dict[str, str]

    @model_validator(mode="after")
    def _check(self) -> "MapValuesParams":
        if not self.mapping:
            raise ValueError("'mapping' must not be empty")
        if len(self.mapping) > MAX_MAPPING_ENTRIES:
            raise ValueError(f"'mapping' is limited to {MAX_MAPPING_ENTRIES} entries")
        return self


class DropColumnParams(_Params):
    pass


class DropRowsWhereNullParams(_Params):
    subset: list[str] = Field(min_length=1)
    how: Literal["any", "all"] = "any"


PARAM_MODELS: dict[str, type[_Params]] = {
    "drop_duplicates": DropDuplicatesParams,
    "impute": ImputeParams,
    "cast_type": CastTypeParams,
    "parse_dates": ParseDatesParams,
    "strip_whitespace": StripWhitespaceParams,
    "normalize_case": NormalizeCaseParams,
    "clip_outliers": ClipOutliersParams,
    "map_values": MapValuesParams,
    "drop_column": DropColumnParams,
    "drop_rows_where_null": DropRowsWhereNullParams,
}
assert set(PARAM_MODELS) == set(OP_NAMES), "every op needs a param schema"


def validate_params(op: str, column: str | None, params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise `params` for `op`. Returns a JSON-safe dict with defaults filled in."""
    model = PARAM_MODELS.get(op)
    if model is None:
        raise OperationError(f"Unknown operation '{op}'. Allowed: {', '.join(sorted(OP_NAMES))}.")
    if op in COLUMN_OPS and not column:
        raise OperationError(f"'{op}' needs a column.")
    if op not in COLUMN_OPS and column:
        raise OperationError(f"'{op}' works on rows, not a single column; use params.subset instead of column.")
    try:
        return model.model_validate(params or {}).model_dump(mode="json")
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'params'}: {e['msg']}" for e in exc.errors())
        raise OperationError(f"Invalid parameters for '{op}': {details}") from None


def is_row_dropping(op: Operation) -> bool:
    return op.op in ROW_DROPPING_OPS


def is_destructive(op: Operation) -> bool:
    return op.op in DESTRUCTIVE_OPS


# --------------------------------------------------------------------------- #
# Validation against a concrete frame
# --------------------------------------------------------------------------- #
def validate_against(df: pd.DataFrame, op: Operation, *, strict: bool = True) -> None:
    """Raise OperationError if `op` cannot run on `df` (hallucinated column, wrong dtype...).

    strict=True also applies the provenance rule: a model-proposed map_values may only map TO
    values already present in the column (merging spellings), never invent new ones. Values the
    user typed themselves (source="user") and replays of an already-approved recipe
    (strict=False) are exempt."""
    cols = set(map(str, df.columns))
    named: list[str] = []
    if op.column:
        named.append(op.column)
    named += list(op.params.get("subset") or [])
    if op.params.get("group_by"):
        named.append(op.params["group_by"])
    missing = [c for c in named if c not in cols]
    if missing:
        raise OperationError(
            f"Column {', '.join(repr(c) for c in missing)} does not exist in the current data."
        )
    if op.column:
        s = df[op.column]
        p = op.params
        if op.op in ("strip_whitespace", "normalize_case", "map_values") and not is_text(s):
            raise OperationError(f"'{op.op}' needs a text column; '{op.column}' is {s.dtype}.")
        if op.op == "clip_outliers" and not is_numeric(s):
            raise OperationError(f"'clip_outliers' needs a numeric column; '{op.column}' is {s.dtype}.")
        if op.op == "impute" and p["strategy"] in ("mean", "median") and not is_numeric(s):
            raise OperationError(f"'{p['strategy']}' needs a numeric column; '{op.column}' is {s.dtype}.")
        if op.op == "impute" and p.get("group_by") == op.column:
            raise OperationError("Cannot impute a column grouped by itself.")
        if strict and op.op == "map_values" and op.source == "llm":
            present = set(string_values(s).astype(str))
            invented = sorted({t for t in p["mapping"].values() if t not in present})
            if invented:
                shown = ", ".join(repr(t) for t in invented[:3])
                raise OperationError(
                    f"The suggested mapping introduces value(s) not found in '{op.column}' ({shown}). "
                    "Only mappings onto existing values are allowed unless you type the value yourself.")


# --------------------------------------------------------------------------- #
# apply()
# --------------------------------------------------------------------------- #
def apply(df: pd.DataFrame, op: Operation, *, strict: bool = True) -> pd.DataFrame:
    """Run `op` on a copy of `df` and return the result. `df` itself is never modified."""
    validate_against(df, op, strict=strict)
    out = df.copy()
    handler = _HANDLERS[op.op]
    return handler(out, op.column, op.params)


def _h_drop_duplicates(df: pd.DataFrame, col: str | None, p: dict) -> pd.DataFrame:
    return df.drop_duplicates(subset=p["subset"], keep=p["keep"])


def _fill_value(series: pd.Series, strategy: str) -> Any:
    nn = series.dropna()
    if nn.empty:
        return None
    if strategy == "mean":
        return nn.mean()
    if strategy == "median":
        return nn.median()
    return nn.mode().iloc[0]  # mode(): sorted, so ties resolve deterministically


def _coerce_constant(s: pd.Series, value: Any, col: str) -> Any:
    if is_numeric(s):
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise OperationError(f"'{value}' is not a number, so it cannot fill numeric column '{col}'.") from None
        return int(num) if num.is_integer() and not pd.api.types.is_float_dtype(s.dtype) else num
    if is_datetime(s):
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError):
            raise OperationError(f"'{value}' is not a valid date for column '{col}'.") from None
    return str(value)


def _h_impute(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    s = df[col]
    mask = null_mask(s)
    if not mask.any():
        return df
    was_categorical = isinstance(s.dtype, pd.CategoricalDtype)
    work = s.astype(object) if was_categorical else s
    work = work.mask(mask)  # blank strings become real nulls
    if pd.api.types.is_integer_dtype(work.dtype):
        work = work.astype("float64")  # a mean/median may not be whole; restore_integral undoes this if it is
    strategy = p["strategy"]

    if strategy == "constant":
        filled = work.fillna(_coerce_constant(s, p["value"], col))
    elif strategy == "ffill":
        filled = work.ffill()
    else:
        fills = None
        if p.get("group_by"):
            fills = work.groupby(df[p["group_by"]]).transform(lambda g: _fill_value(g, strategy))
        filled = work if fills is None else work.fillna(fills)
        global_fill = _fill_value(work, strategy)
        if global_fill is not None:
            filled = filled.fillna(global_fill)

    if was_categorical:
        filled = filled.astype("category")
    elif is_numeric(s):
        filled = restore_integral(s, filled)
    df[col] = filled
    return df


_BOOL_TRUE = {"true", "t", "yes", "y", "1"}
_BOOL_FALSE = {"false", "f", "no", "n", "0"}


def _h_cast_type(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    s, dtype, errors = df[col], p["dtype"], p["errors"]
    blank = null_mask(s)

    if dtype in ("int", "float"):
        src = s.mask(blank)
        if is_text(s):
            src = src.astype("string").str.strip()
        num = pd.to_numeric(src, errors="coerce").astype("float64")
        inf = np.isinf(num.to_numpy())
        num = num.mask(inf)
        bad = src.notna() & num.isna()
        if errors == "raise" and bad.any():
            ex = ", ".join(repr(v) for v in src[bad].head(3))
            raise OperationError(f"{int(bad.sum())} value(s) in '{col}' are not numbers (e.g. {ex}).")
        if dtype == "int":
            frac = num.notna() & (np.mod(num.fillna(0).to_numpy(), 1) != 0)
            if frac.any():
                ex = ", ".join(str(v) for v in num[frac].head(3))
                raise OperationError(f"'{col}' has non-whole numbers (e.g. {ex}); cast to float instead.")
            df[col] = num.astype("Int64")
        else:
            df[col] = num
    elif dtype == "str":
        df[col] = s.mask(blank).astype("string")
    elif dtype == "category":
        df[col] = s.mask(blank).astype("category")
    else:  # bool
        txt = s.mask(blank).astype("string").str.strip().str.lower()
        mapped = txt.map(lambda v: True if v in _BOOL_TRUE else (False if v in _BOOL_FALSE else pd.NA))
        bad = txt.notna() & mapped.isna()
        if errors == "raise" and bad.any():
            ex = ", ".join(repr(v) for v in txt[bad].head(3))
            raise OperationError(f"{int(bad.sum())} value(s) in '{col}' are not yes/no style booleans (e.g. {ex}).")
        df[col] = mapped.astype("boolean")
    return df


def _h_parse_dates(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    s = df[col]
    if is_datetime(s):
        return df
    text = s.mask(null_mask(s)).astype("string").str.strip()
    if p["format"]:
        parsed = pd.to_datetime(text, format=p["format"], errors="coerce")
    else:
        parsed = pd.to_datetime(text, format="mixed", dayfirst=p["dayfirst"], errors="coerce")
    bad = text.notna() & parsed.isna()
    if text.notna().any() and bad.sum() == text.notna().sum():
        hint = f" with format '{p['format']}'" if p["format"] else ""
        raise OperationError(f"None of the values in '{col}' could be read as dates{hint}; nothing was changed.")
    if p["errors"] == "raise" and bad.any():
        ex = ", ".join(repr(v) for v in text[bad].head(3))
        raise OperationError(f"{int(bad.sum())} value(s) in '{col}' are not dates (e.g. {ex}).")
    df[col] = parsed
    return df


def _h_strip_whitespace(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    def fn(v: str) -> str | None:
        t = clean_ws(v) if p["collapse"] else v.strip()
        return t or None  # whitespace-only becomes a real null

    df[col] = str_map(df[col], fn)
    return df


def _title(v: str) -> str:
    return " ".join(w[:1].upper() + w[1:].lower() for w in v.split(" "))


def _h_normalize_case(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    fn: Callable[[str], str] = {"lower": str.lower, "upper": str.upper, "title": _title}[p["mode"]]
    df[col] = str_map(df[col], fn)
    return df


def _clip_bounds(nn: pd.Series, p: dict) -> tuple[float | None, float | None]:
    if p["method"] == "iqr":
        q1, q3 = nn.quantile(0.25), nn.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            raise OperationError("The IQR is 0 for this column, so IQR clipping would flatten it; use manual bounds.")
        return float(q1 - p["k"] * iqr), float(q3 + p["k"] * iqr)
    lo, hi = p["bounds"]
    if p["method"] == "percentile":
        return float(nn.quantile(lo / 100)), float(nn.quantile(hi / 100))
    return lo, hi


def _h_clip_outliers(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    s = df[col]
    as_float = s.astype("float64")
    nn = as_float.dropna()
    if nn.empty:
        return df
    lo, hi = _clip_bounds(nn, p)
    df[col] = restore_integral(s, as_float.clip(lower=lo, upper=hi))
    return df


def _h_map_values(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    mapping = p["mapping"]
    df[col] = str_map(df[col], lambda v: mapping.get(v, v))
    return df


def _h_drop_column(df: pd.DataFrame, col: str, p: dict) -> pd.DataFrame:
    return df.drop(columns=[col])


def _h_drop_rows_where_null(df: pd.DataFrame, col: None, p: dict) -> pd.DataFrame:
    masks = pd.concat([null_mask(df[c]) for c in p["subset"]], axis=1)
    drop = masks.any(axis=1) if p["how"] == "any" else masks.all(axis=1)
    return df.loc[~drop]


_HANDLERS: dict[str, Callable[..., pd.DataFrame]] = {
    "drop_duplicates": _h_drop_duplicates,
    "impute": _h_impute,
    "cast_type": _h_cast_type,
    "parse_dates": _h_parse_dates,
    "strip_whitespace": _h_strip_whitespace,
    "normalize_case": _h_normalize_case,
    "clip_outliers": _h_clip_outliers,
    "map_values": _h_map_values,
    "drop_column": _h_drop_column,
    "drop_rows_where_null": _h_drop_rows_where_null,
}
assert set(_HANDLERS) == set(OP_NAMES)


# --------------------------------------------------------------------------- #
# Effect measurement
# --------------------------------------------------------------------------- #
def rows_dropped(before: pd.DataFrame, after: pd.DataFrame) -> int:
    return max(len(before) - len(after), 0)


def rows_affected(before: pd.DataFrame, after: pd.DataFrame, op: Operation) -> int:
    """Rows touched by `op`: removed rows for row-dropping ops, every row for drop_column,
    otherwise the number of cells whose value (or, for a cast, representation) changed."""
    if op.op in ROW_DROPPING_OPS:
        return rows_dropped(before, after)
    if op.op == "drop_column":
        return len(before)
    col = op.column
    assert col is not None
    b, a = before[col], after[col]
    if op.op in ("cast_type", "parse_dates") and str(b.dtype) != str(a.dtype):
        return int((~null_mask(b)).sum())
    return int(changed_mask(b, a).sum())
