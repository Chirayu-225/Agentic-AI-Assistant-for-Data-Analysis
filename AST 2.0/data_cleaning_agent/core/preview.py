"""
core/preview.py - dry-run an Operation on a copy and report what WOULD change.

Never raises for a bad operation: failures come back as Preview(ok=False, error=...)
so the UI can show them on the proposal card instead of crashing.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core import operations
from core.operations import OperationError, ROW_DROP_CONFIRM_THRESHOLD
from core.schemas import Operation
from core.utils import changed_mask, short


class SampleDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: str  # original row label
    column: str | None
    before: str
    after: str


class Preview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Operation
    ok: bool = True
    error: str | None = None
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    rows_affected: int = 0
    pct_affected: float = 0.0  # 0-100
    rows_dropped: int = 0
    pct_dropped: float = 0.0  # 0-100
    requires_extra_confirmation: bool = False
    dtype_before: str | None = None
    dtype_after: str | None = None
    samples: list[SampleDiff] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _row_text(row: pd.Series, limit: int = 6) -> str:
    cells = [f"{k}={short(v, 18)}" for k, v in list(row.items())[:limit]]
    more = " …" if len(row) > limit else ""
    return " | ".join(cells) + more


def _samples(before: pd.DataFrame, after: pd.DataFrame, op: Operation, n: int) -> list[SampleDiff]:
    if op.op in operations.ROW_DROPPING_OPS:
        gone = before.index.difference(after.index, sort=False)
        gone = [i for i in before.index if i in set(gone)][:n]
        return [SampleDiff(row=str(i), column=None, before=_row_text(before.loc[i]), after="(row removed)")
                for i in gone]
    if op.op == "drop_column":
        col = op.column or ""
        return [SampleDiff(row=str(i), column=col, before=short(v), after="(column removed)")
                for i, v in before[col].head(n).items()]
    col = op.column or ""
    b, a = before[col], after[col]
    mask = changed_mask(b, a)
    out = []
    for i in b.index[mask.to_numpy()][:n]:
        out.append(SampleDiff(row=str(i), column=col, before=short(b.loc[i]), after=short(a.loc[i])))
    return out


def preview(
    df: pd.DataFrame,
    op: Operation,
    *,
    n_samples: int = 5,
    row_drop_threshold: float = ROW_DROP_CONFIRM_THRESHOLD,
    strict: bool = True,
) -> Preview:
    """Dry-run `op` on `df` (which is not modified)."""
    if not df.index.is_unique:  # row labels must identify rows for the before/after samples
        df = df.reset_index(drop=True)
    base = Preview(operation=op, rows_before=len(df), cols_before=df.shape[1])
    try:
        after = operations.apply(df, op, strict=strict)
    except OperationError as exc:
        return base.model_copy(update={"ok": False, "error": str(exc), "rows_after": len(df),
                                       "cols_after": df.shape[1]})

    affected = operations.rows_affected(df, after, op)
    dropped = operations.rows_dropped(df, after)
    total = len(df)
    warnings: list[str] = []
    if affected == 0:
        warnings.append("This would not change any rows.")
    if op.op == "drop_column":
        warnings.append(f"Removes the whole '{op.column}' column.")
    if op.op == "parse_dates" and op.column and op.params.get("errors") == "coerce":
        before_nulls = int(operations.null_mask(df[op.column]).sum())
        lost = int(after[op.column].isna().sum()) - before_nulls
        if lost > 0:
            warnings.append(f"{lost} value(s) are not valid dates and would become empty.")
    if op.op == "cast_type" and op.column and op.params.get("errors") == "coerce":
        before_nulls = int(operations.null_mask(df[op.column]).sum())
        lost = int(after[op.column].isna().sum()) - before_nulls
        if lost > 0:
            warnings.append(f"{lost} value(s) could not be converted and would become empty.")

    pct_dropped = round(100 * dropped / total, 2) if total else 0.0
    return base.model_copy(update={
        "rows_after": len(after),
        "cols_after": after.shape[1],
        "rows_affected": affected,
        "pct_affected": round(100 * affected / total, 2) if total else 0.0,
        "rows_dropped": dropped,
        "pct_dropped": pct_dropped,
        "requires_extra_confirmation": bool(total and dropped / total > row_drop_threshold),
        "dtype_before": str(df[op.column].dtype) if op.column else None,
        "dtype_after": str(after[op.column].dtype) if op.column and op.column in after.columns else None,
        "samples": _samples(df, after, op, n_samples),
        "warnings": warnings,
    })
