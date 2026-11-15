"""
Double computation for query-plan answers (hazard H1: a wrong number shown as
if it were verified).

A validated plan (analysis/query_plan.py) is executed by DuckDB as SQL. This
module computes the SAME plan a second time in pandas, compares the two
results, and runs invariant checks that need neither engine to be right.

    verified      both computations agree and every invariant holds
    disagreement  they differ, or an invariant is broken -> the caller must
                  WITHHOLD the answer ("I can't verify this")
    unverified    the cross-check itself could not run -> the caller may show
                  the answer but must label it as not cross-checked

INDEPENDENCE
    This file deliberately imports nothing from the SQL side except the Plan
    data classes. It does not call compile_plan, and it re-implements filters,
    grouping, aggregation, ordering and limits from the written semantics, so
    a bug in the SQL compiler is not copied here.

WHAT THIS DOES NOT PROVE (be honest about it)
    Both computations start from the same plan. If the model produced a plan
    that answers a DIFFERENT question from the one the user asked, both
    engines will agree on the wrong answer. That failure is measured by the
    evaluation suite, not caught here.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from analysis.query_plan import Plan

REL_TOL = 1e-9
ABS_TOL = 1e-9


@dataclass
class Verification:
    status: str                      # "verified" | "disagreement" | "unverified"
    reasons: list = field(default_factory=list)
    checks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "reasons": list(self.reasons)[:5], "checks": list(self.checks)}


# ── the reference implementation (pandas) ──────────────────────────────────
def _plain(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.astype(s.cat.categories.dtype)
    return s


def _filter_mask(f, s: pd.Series) -> np.ndarray:
    """SQL semantics: a comparison with a missing value is never true."""
    s = _plain(s)
    present = s.notna().to_numpy(dtype=bool)
    if f.op == "is_null":
        return ~present
    if f.op == "not_null":
        return present
    v = f.value
    if f.op == "=":
        m = s == v
    elif f.op == "!=":
        m = s != v
    elif f.op == "<":
        m = s < v
    elif f.op == "<=":
        m = s <= v
    elif f.op == ">":
        m = s > v
    elif f.op == ">=":
        m = s >= v
    elif f.op == "between":
        m = (s >= v[0]) & (s <= v[1])
    elif f.op == "in":
        m = s.isin(list(v))
    elif f.op == "not_in":
        m = ~s.isin(list(v))
    elif f.op == "contains":
        needle = v.lower()
        m = s.astype(object).map(lambda x: isinstance(x, str) and needle in x.lower())
    else:
        raise ValueError(f"unknown operator {f.op!r}")
    hit = np.array([bool(x) if x is not pd.NA and x == x else False for x in m.tolist()], dtype=bool)
    return hit & present


def _sorted_positions(keys: list, ascending: list, n: int) -> np.ndarray:
    """Row positions ordered by the given key columns, missing values last."""
    if not keys:
        return np.arange(n)
    sf = pd.DataFrame({f"s{i}": pd.Series(k).reset_index(drop=True) for i, k in enumerate(keys)})
    return sf.sort_values(by=list(sf.columns), ascending=ascending, kind="stable",
                          na_position="last").index.to_numpy()


def reference_execute(plan: Plan, df: pd.DataFrame) -> pd.DataFrame:
    work = df.reset_index(drop=True)
    keep = np.ones(len(work), dtype=bool)
    for f in plan.filters:
        keep &= _filter_mask(f, work[f.column])
    work = work.loc[keep]
    has_agg = any(i.agg for i in plan.select)

    if not has_agg:
        out = pd.DataFrame({i.out: _plain(work[i.column]).reset_index(drop=True) for i in plan.select})
        by_out = {i.out: i for i in plan.select}
        keys = [out[b] for b, _ in plan.order_by] + [pd.Series(np.arange(len(out)))]
        asc = [d == "asc" for _, d in plan.order_by] + [True]
        order = _sorted_positions(keys, asc, len(out))
        return out.iloc[order].head(plan.limit).reset_index(drop=True)

    if plan.group_by:
        gcols = [_plain(work[g]).reset_index(drop=True) for g in plan.group_by]
        gframe = pd.DataFrame({f"g{k}": c for k, c in enumerate(gcols)})
        ids = gframe.groupby(list(gframe.columns), dropna=False, sort=False).ngroup().to_numpy()
        n_groups = int(ids.max()) + 1 if len(ids) else 0
        first = pd.Series(np.arange(len(ids))).groupby(ids).first().reindex(range(n_groups)).to_numpy()
        key_values = {g: gcols[k].iloc[first].reset_index(drop=True) for k, g in enumerate(plan.group_by)}
    else:
        ids = np.zeros(len(work), dtype=int)
        n_groups = 1
        key_values = {}

    cols: dict = {}
    for i in plan.select:
        if i.agg is None:
            cols[i.out] = key_values[i.column]
            continue
        if i.column is None:
            vals = pd.Series(ids).groupby(ids).size()
            cols[i.out] = vals.reindex(range(n_groups), fill_value=0).reset_index(drop=True)
            continue
        s = _plain(work[i.column]).reset_index(drop=True)
        g = s.groupby(ids)
        if i.agg == "count":
            vals = g.count().reindex(range(n_groups), fill_value=0)
        elif i.agg == "count_distinct":
            vals = g.nunique().reindex(range(n_groups), fill_value=0)
        elif i.agg == "sum":
            vals = g.sum(min_count=1).reindex(range(n_groups))
        elif i.agg == "avg":
            vals = g.mean().reindex(range(n_groups))
        elif i.agg == "min":
            vals = g.min().reindex(range(n_groups))
        elif i.agg == "max":
            vals = g.max().reindex(range(n_groups))
        elif i.agg == "median":
            vals = g.median().reindex(range(n_groups))
        else:
            raise ValueError(f"unknown aggregate {i.agg!r}")
        cols[i.out] = vals.reset_index(drop=True)
    out = pd.DataFrame(cols, columns=[i.out for i in plan.select])

    keys = [out[b] for b, _ in plan.order_by] + [key_values[g] for g in plan.group_by]
    asc = [d == "asc" for _, d in plan.order_by] + [True] * len(plan.group_by)
    order = _sorted_positions(keys, asc, len(out))
    return out.iloc[order].head(plan.limit).reset_index(drop=True)


# ── comparison ─────────────────────────────────────────────────────────────
def _norm(v: Any) -> Any:
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if f != f else f
    if isinstance(v, (pd.Timestamp, _dt.datetime, np.datetime64)):
        t = pd.Timestamp(v)
        return None if pd.isna(t) else t
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _rows(frame: pd.DataFrame) -> list:
    cols = [[_norm(x) for x in frame.iloc[:, j].tolist()] for j in range(frame.shape[1])]
    return [tuple(c[r] for c in cols) for r in range(len(frame))]


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return type(a) is type(b) and a == b


def _sort_key(row: tuple) -> tuple:
    def cell(v):
        if v is None:
            return (1, "", 0)
        if isinstance(v, bool):
            return (0, "b", int(v))
        if isinstance(v, float):
            return (0, "f", float(f"{v:.9g}"))
        if isinstance(v, pd.Timestamp):
            return (0, "t", v.value)
        return (0, "s", v)
    return tuple(cell(v) for v in row)


def _first_difference(a_rows: list, b_rows: list, names: list) -> str | None:
    for r, (a, b) in enumerate(zip(a_rows, b_rows)):
        for j, (x, y) in enumerate(zip(a, b)):
            if not _close(x, y):
                return f"row {r + 1}, column {names[j]!r}: engine gave {x!r}, the second computation gave {y!r}"
    return None


def _order_problem(rows: list, plan: Plan, names: list) -> str | None:
    idx = [(names.index(b), d) for b, d in plan.order_by]
    for r in range(len(rows) - 1):
        a, b = rows[r], rows[r + 1]
        for j, d in idx:
            x, y = a[j], b[j]
            if x is None and y is None:
                continue
            if x is None:
                return f"rows {r + 1} and {r + 2} are out of order on {names[j]!r} (missing values must come last)"
            if y is None or _close(x, y):
                if y is None:
                    break
                continue
            if (x < y) == (d == "asc"):
                break
            return f"rows {r + 1} and {r + 2} are out of order on {names[j]!r}"
    return None


def compare_frames(plan: Plan, engine: pd.DataFrame, reference: pd.DataFrame) -> list:
    names = [i.out for i in plan.select]
    if list(map(str, engine.columns)) != names:
        return [f"output columns are {list(map(str, engine.columns))}, expected {names}"]
    if len(engine) != len(reference):
        return [f"the engine returned {len(engine)} row(s), the second computation {len(reference)}"]
    a, b = _rows(engine), _rows(reference)
    aligned = _first_difference(a, b, names)
    if aligned is None:
        return []
    # Sums added in a different order differ in the last digits, which can swap two
    # near-tied rows. Accept that only if the SAME rows are present and the engine's
    # order is still correct within tolerance.
    if plan.order_by and any(i.agg for i in plan.select):
        if _first_difference(sorted(a, key=_sort_key), sorted(b, key=_sort_key), names) is None:
            problem = _order_problem(a, plan, names)
            return [problem] if problem else []
    return [aligned]


# ── invariants (need neither engine to be right) ───────────────────────────
def check_invariants(plan: Plan, df: pd.DataFrame, frame: pd.DataFrame) -> list:
    problems: list = []
    names = [i.out for i in plan.select]
    if list(map(str, frame.columns)) != names:
        return [f"output columns are {list(map(str, frame.columns))}, expected {names}"]
    n, total = len(frame), len(df)
    has_agg = any(i.agg for i in plan.select)
    if n > plan.limit:
        problems.append(f"{n} rows returned, more than the limit of {plan.limit}")
    values = {i.out: [_norm(x) for x in frame[i.out].tolist()] for i in plan.select}

    if not has_agg:
        if n > total:
            problems.append(f"{n} rows returned from a table of {total}")
        if not plan.filters and n != min(total, plan.limit):
            problems.append(f"{n} rows returned, expected {min(total, plan.limit)} with no filters")
        return problems

    if not plan.group_by and n != 1:
        problems.append(f"an ungrouped aggregate must return exactly 1 row, got {n}")

    for i in plan.select:
        if i.agg in ("count", "count_distinct"):
            for v in values[i.out]:
                if v is None or v < 0 or v > total or v != int(v):
                    problems.append(f"{i.out!r} = {v!r} is not a possible count for a table of {total} rows")
                    break
        elif i.agg in ("min", "max", "avg", "median"):
            col = _plain(df[i.column]).dropna()
            if len(col):
                lo, hi = _norm(col.min()), _norm(col.max())
                for v in values[i.out]:
                    if v is None:
                        continue
                    below = v < lo and not _close(v, lo)
                    above = v > hi and not _close(v, hi)
                    if below or above:
                        problems.append(f"{i.out!r} = {v!r} lies outside the column's range [{lo!r}, {hi!r}]")
                        break

    by_col: dict = {}
    for i in plan.select:
        if i.agg in ("min", "avg", "median", "max") and i.column is not None:
            by_col.setdefault(i.column, {})[i.agg] = values[i.out]
    for col, aggs in by_col.items():
        for low, high in (("min", "avg"), ("min", "median"), ("avg", "max"), ("median", "max"), ("min", "max")):
            if low in aggs and high in aggs:
                for x, y in zip(aggs[low], aggs[high]):
                    if x is not None and y is not None and x > y and not _close(x, y):
                        problems.append(f"{low} of {col!r} ({x!r}) is greater than its {high} ({y!r})")
                        break

    complete = n < plan.limit  # nothing was cut off by the limit
    if plan.group_by and complete:
        distinct = len(df.drop_duplicates(subset=list(plan.group_by)))
        if n > distinct:
            problems.append(f"{n} groups returned, but the table has only {distinct} distinct group(s)")
        for i in plan.select:
            if i.agg == "count" and i.column is None:
                s = sum(v or 0 for v in values[i.out])
                if s > total or (not plan.filters and s != total):
                    problems.append(f"group counts add up to {s:g}, but the table has {total} rows")
            if i.agg == "sum" and not plan.filters:
                s = sum(v or 0.0 for v in values[i.out])
                whole = float(_plain(df[i.column]).sum())
                if not math.isclose(s, whole, rel_tol=1e-9, abs_tol=1e-6):
                    problems.append(f"group sums of {i.column!r} add up to {s!r}, but the column total is {whole!r}")
    return problems


# ── entry point ────────────────────────────────────────────────────────────
def verify(plan: Plan, df: pd.DataFrame, frame: pd.DataFrame) -> Verification:
    checks: list = []
    try:
        problems = check_invariants(plan, df, frame)
        checks.append("invariants")
    except Exception as e:
        return Verification("unverified", [f"the invariant checks could not run ({type(e).__name__})"], checks)
    if problems:
        return Verification("disagreement", problems, checks)
    try:
        reference = reference_execute(plan, df)
        problems = compare_frames(plan, frame, reference)
        checks.append("independent pandas recomputation")
    except Exception as e:
        return Verification("unverified", [f"the second computation could not run ({type(e).__name__})"], checks)
    return Verification("disagreement" if problems else "verified", problems, checks)
