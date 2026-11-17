"""
Typed query plan: a small, closed language for the questions that make up most
data analysis (select / filter / group / aggregate / order / limit).

WHY THIS EXISTS (hazards H1 "wrong answer shown as verified" and H3 "generated
code reaches the host")
    Letting a model write free-form Python for every question means the system
    can only check the code AFTER the fact, with blocklists. Here the model
    proposes DATA (a JSON plan), and this module:
      1. validates it strictly against the real schema - anything outside the
         language is rejected with a reason the model can be shown for repair;
      2. compiles it to SQL in which every VALUE is a bound parameter and every
         IDENTIFIER is quoted by us - model text never becomes SQL syntax;
      3. executes it in DuckDB with fixed tie-breaking, so the same question
         over the same data returns the same table, and
      4. returns the plan, SQL and parameters for the audit record.

    Questions the language cannot express (charts, joins, custom logic) are not
    forced into it; the caller falls back to the sandboxed-Python path.

SEMANTICS WORTH KNOWING (SQL, not pandas)
    * != and not_in exclude rows where the column is NULL.
    * sum/avg/min/max/median ignore NULLs; SUM of no non-null values is NULL.
    * count(*) counts every row; count(column) counts non-null values.
    * contains is a case-insensitive substring match; % and _ are literal.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PLAN_VERSION = 1
AGGS = ("count", "count_distinct", "sum", "avg", "min", "max", "median")
OPS = ("=", "!=", "<", "<=", ">", ">=", "between", "in", "not_in", "contains", "is_null", "not_null")
DEFAULT_LIMIT = 1000
MAX_LIMIT = 10_000
MAX_ITEMS = 50
MAX_IN_VALUES = 100
MAX_PLAN_CHARS = 20_000
ROW_COL = "__ast_row__"
_ALIAS = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_TOP_KEYS = {"version", "select", "filters", "group_by", "order_by", "limit"}


class PlanError(ValueError):
    """The plan is outside the language or does not fit the data. The message
    is written to be shown back to the model for one repair attempt."""


@dataclass(frozen=True)
class SelectItem:
    column: str | None
    agg: str | None
    out: str


@dataclass(frozen=True)
class Filter:
    column: str
    op: str
    value: Any


@dataclass(frozen=True)
class Plan:
    select: tuple
    filters: tuple
    group_by: tuple
    order_by: tuple  # of (output name, "asc"|"desc")
    limit: int

    def to_dict(self) -> dict:
        return {
            "version": PLAN_VERSION,
            "select": [{"agg": i.agg, "column": i.column, "as": i.out} for i in self.select],
            "filters": [{"column": f.column, "op": f.op, "value": _jsonable(f.value)} for f in self.filters],
            "group_by": list(self.group_by),
            "order_by": [{"by": b, "dir": d} for b, d in self.order_by],
            "limit": self.limit,
        }


@dataclass(frozen=True)
class PlanResult:
    frame: pd.DataFrame
    sql: str
    params: list
    plan: Plan


def _jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


# ── schema ─────────────────────────────────────────────────────────────────
def _kind(s: pd.Series) -> str:
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return _kind(pd.Series(s.cat.categories))
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_complex_dtype(dt):
        return "num"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime" if getattr(dt, "tz", None) is None else "other"
    if dt == object:
        # Text columns usually hold some None/NaN; judge by the non-missing values.
        return "str" if pd.api.types.infer_dtype(s, skipna=True) in ("string", "empty") else "other"
    if pd.api.types.is_string_dtype(dt):
        return "str"
    return "other"


def _schema(df: pd.DataFrame) -> dict[str, str]:
    cols = list(df.columns)
    if not all(isinstance(c, str) for c in cols):
        raise PlanError("this table has non-text column names, which query plans do not support")
    if len(set(cols)) != len(cols):
        raise PlanError("this table has duplicate column names, which query plans do not support")
    if ROW_COL in cols:
        raise PlanError(f"column name {ROW_COL!r} is reserved")
    return {c: _kind(df[c]) for c in cols}


# ── validation ─────────────────────────────────────────────────────────────
def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _scalar_for(kind: str, v: Any, where: str) -> Any:
    """Check one filter value against the column kind; return it normalised."""
    if kind == "num":
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise PlanError(f"{where}: needs a number, got {v!r}")
        if not math.isfinite(float(v)):
            raise PlanError(f"{where}: the number must be finite, got {v!r}")
        return v.item() if isinstance(v, np.generic) else v
    if kind == "str":
        if not isinstance(v, str):
            raise PlanError(f"{where}: needs text, got {v!r}")
        return v
    if kind == "bool":
        if not isinstance(v, (bool, np.bool_)):
            raise PlanError(f"{where}: needs true or false, got {v!r}")
        return bool(v)
    if kind == "datetime":
        if not isinstance(v, str):
            raise PlanError(f"{where}: needs a date as text such as '2024-03-01', got {v!r}")
        try:
            ts = pd.Timestamp(v)
        except (ValueError, TypeError):
            raise PlanError(f"{where}: {v!r} is not a date") from None
        if pd.isna(ts) or ts.tzinfo is not None:
            raise PlanError(f"{where}: {v!r} must be a plain date or datetime without a timezone")
        return ts.to_pydatetime()
    raise PlanError(f"{where}: this column's type cannot be filtered")


def validate_plan(plan: Any, df: pd.DataFrame) -> Plan:
    if not isinstance(plan, dict):
        raise PlanError("a plan must be a JSON object")
    extra = set(plan) - _TOP_KEYS
    if extra:
        raise PlanError(f"unknown key {sorted(extra)[0]!r} in plan; allowed keys are {sorted(_TOP_KEYS)}")
    if "version" in plan and plan["version"] != PLAN_VERSION:
        raise PlanError(f"unsupported plan version {plan['version']!r}; use version {PLAN_VERSION}")
    kinds = _schema(df)

    def need_col(name, where):
        if not isinstance(name, str) or name not in kinds:
            raise PlanError(f"{where}: unknown column {name!r}; columns are {list(kinds)}")
        return name

    # select -----------------------------------------------------------------
    raw_sel = plan.get("select")
    if not isinstance(raw_sel, list) or not raw_sel:
        raise PlanError("select must be a non-empty list")
    if len(raw_sel) > MAX_ITEMS:
        raise PlanError(f"select has more than {MAX_ITEMS} items")
    items: list[SelectItem] = []
    for k, it in enumerate(raw_sel):
        where = f"select[{k}]"
        if not isinstance(it, dict):
            raise PlanError(f"{where} must be an object")
        extra = set(it) - {"column", "agg", "as"}
        if extra:
            raise PlanError(f"{where}: unknown key {sorted(extra)[0]!r}")
        agg, col, alias = it.get("agg"), it.get("column"), it.get("as")
        if agg == "count" and col == "*":
            col = None  # SQL spelling of "count all rows"; models write it constantly
        if agg is not None and agg not in AGGS:
            raise PlanError(f"{where}: unknown aggregate {agg!r}; allowed: {list(AGGS)}")
        if agg is None:
            need_col(col, where)
        elif col is None:
            if agg != "count":
                raise PlanError(f"{where}: aggregate {agg!r} needs a column")
        else:
            need_col(col, where)
            kind = kinds[col]
            if agg in ("sum", "avg", "median") and kind != "num":
                raise PlanError(f"{where}: {agg} needs a numeric column, but {col!r} is {kind}")
            if agg in ("min", "max") and kind not in ("num", "datetime"):
                raise PlanError(f"{where}: {agg} needs a numeric or date column, but {col!r} is {kind}")
            if agg in ("count", "count_distinct") and kind == "other":
                raise PlanError(f"{where}: {col!r} has a mixed or unsupported type")
        if alias is not None and (not isinstance(alias, str) or not _ALIAS.match(alias)):
            raise PlanError(f"{where}: invalid alias {alias!r}; use letters, digits and underscores only")
        if alias is None:
            out = col if agg is None else ("count" if col is None else f"{agg}_{col}")
        else:
            out = alias
        if out in kinds and not (agg is None and out == col):
            raise PlanError(f"{where}: alias {out!r} collides with an existing column; choose another name with \"as\"")
        items.append(SelectItem(column=col, agg=agg, out=out))
    outs = [i.out for i in items]
    if len(set(outs)) != len(outs):
        raise PlanError(f"duplicate output names in select: {outs}")
    has_agg = any(i.agg for i in items)

    # group_by ---------------------------------------------------------------
    gb = plan.get("group_by", [])
    if not isinstance(gb, list) or len(gb) > MAX_ITEMS:
        raise PlanError("group_by must be a list of column names")
    for g in gb:
        need_col(g, "group_by")
        if kinds[g] == "other":
            raise PlanError(f"group_by: {g!r} has a mixed or unsupported type")
    if gb and not has_agg:
        raise PlanError("group_by needs at least one aggregate in select")
    if has_agg:
        for i in items:
            if i.agg is None and i.column not in gb:
                raise PlanError(f"column {i.column!r} must appear in group_by when select contains aggregates")

    # filters ----------------------------------------------------------------
    raw_f = plan.get("filters", [])
    if not isinstance(raw_f, list) or len(raw_f) > MAX_ITEMS:
        raise PlanError("filters must be a list")
    filters: list[Filter] = []
    for k, f in enumerate(raw_f):
        where = f"filters[{k}]"
        if not isinstance(f, dict):
            raise PlanError(f"{where} must be an object")
        extra = set(f) - {"column", "op", "value"}
        if extra:
            raise PlanError(f"{where}: unknown key {sorted(extra)[0]!r}")
        op = f.get("op")
        if op not in OPS:
            raise PlanError(f"{where}: unknown operator {op!r}; allowed: {list(OPS)}")
        col = need_col(f.get("column"), where)
        kind = kinds[col]
        label = f"{where} on {col!r} ({op})"
        if op in ("is_null", "not_null"):
            if f.get("value") is not None:
                raise PlanError(f"{label}: this operator takes no value")
            filters.append(Filter(col, op, None))
            continue
        if "value" not in f:
            raise PlanError(f"{label}: a value is required")
        v = f["value"]
        if op in ("=", "!="):
            filters.append(Filter(col, op, _scalar_for(kind, v, label)))
        elif op in ("<", "<=", ">", ">="):
            if kind not in ("num", "datetime"):
                raise PlanError(f"{label}: ordering comparisons work on numeric or date columns, but {col!r} is {kind}")
            filters.append(Filter(col, op, _scalar_for(kind, v, label)))
        elif op == "between":
            if kind not in ("num", "datetime"):
                raise PlanError(f"{label}: between works on numeric or date columns, but {col!r} is {kind}")
            if not isinstance(v, list) or len(v) != 2:
                raise PlanError(f"{label}: between needs a list of exactly two values")
            filters.append(Filter(col, op, tuple(_scalar_for(kind, x, label) for x in v)))
        elif op in ("in", "not_in"):
            if not isinstance(v, list) or not (1 <= len(v) <= MAX_IN_VALUES):
                raise PlanError(f"{label}: needs a list of 1 to {MAX_IN_VALUES} values")
            filters.append(Filter(col, op, tuple(_scalar_for(kind, x, label) for x in v)))
        else:  # contains
            if kind != "str":
                raise PlanError(f"{label}: contains works on text columns, but {col!r} is {kind}")
            if not isinstance(v, str) or not v or len(v) > 200:
                raise PlanError(f"{label}: needs non-empty text up to 200 characters")
            filters.append(Filter(col, op, v))

    # order_by / limit -------------------------------------------------------
    raw_o = plan.get("order_by", [])
    if not isinstance(raw_o, list) or len(raw_o) > 10:
        raise PlanError("order_by must be a list of up to 10 items")
    order: list[tuple[str, str]] = []
    for k, o in enumerate(raw_o):
        where = f"order_by[{k}]"
        if not isinstance(o, dict) or set(o) - {"by", "dir"}:
            raise PlanError(f"{where} must be an object with keys by and dir")
        d = o.get("dir", "asc")
        if d not in ("asc", "desc"):
            raise PlanError(f"{where}: dir must be 'asc' or 'desc', got {d!r}")
        if o.get("by") not in outs:
            raise PlanError(f"{where}: {o.get('by')!r} is not an output column; order by one of {outs}")
        order.append((o["by"], d))

    limit = plan.get("limit", DEFAULT_LIMIT)
    if not _is_int(limit) or not (1 <= limit <= MAX_LIMIT):
        raise PlanError(f"limit must be an integer between 1 and {MAX_LIMIT}")

    return Plan(select=tuple(items), filters=tuple(filters), group_by=tuple(gb),
                order_by=tuple(order), limit=int(limit))


def parse_plan(text: str, df: pd.DataFrame) -> Plan:
    """Extract and validate a JSON plan from raw model output."""
    if not isinstance(text, str) or not text.strip():
        raise PlanError("the model returned no plan")
    if len(text) > MAX_PLAN_CHARS:
        raise PlanError(f"the plan is too long (over {MAX_PLAN_CHARS} characters)")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise PlanError("no JSON object found in the model output")
    try:
        obj = json.loads(text[start:end + 1])
    except (ValueError, RecursionError):
        raise PlanError("the plan is not valid JSON") from None
    return validate_plan(obj, df)


# ── compilation ────────────────────────────────────────────────────────────
def _q(name: str) -> str:
    """Quote an identifier. Only ever applied to names validated against the schema."""
    if "\x00" in name:
        raise PlanError("invalid identifier")
    return '"' + name.replace('"', '""') + '"'


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def compile_plan(plan: Plan, df: pd.DataFrame) -> tuple[str, list]:
    """Plan -> (SQL text, parameter list). The SQL text contains identifiers we
    quoted and fixed keywords only; every value is a '?' with a bound parameter."""
    params: list = []

    def agg_sql(i: SelectItem) -> str:
        if i.agg is None:
            return _q(i.column)
        if i.agg == "count":
            return "COUNT(*)" if i.column is None else f"COUNT({_q(i.column)})"
        c = _q(i.column)
        if i.agg == "count_distinct":
            return f"COUNT(DISTINCT {c})"
        if i.agg == "sum":
            return f"CAST(SUM({c}) AS BIGINT)" if pd.api.types.is_integer_dtype(df[i.column]) else f"SUM({c})"
        return {"avg": "AVG", "min": "MIN", "max": "MAX", "median": "MEDIAN"}[i.agg] + f"({c})"

    sel = ", ".join(f"{agg_sql(i)} AS {_q(i.out)}" for i in plan.select)

    where = []
    for f in plan.filters:
        c = _q(f.column)
        if f.op in ("=", "!=", "<", "<=", ">", ">="):
            where.append(f"{c} {'<>' if f.op == '!=' else f.op} ?")
            params.append(f.value)
        elif f.op == "between":
            where.append(f"{c} BETWEEN ? AND ?")
            params.extend(f.value)
        elif f.op in ("in", "not_in"):
            marks = ", ".join("?" for _ in f.value)
            where.append(f"{c} {'NOT IN' if f.op == 'not_in' else 'IN'} ({marks})")
            params.extend(f.value)
        elif f.op == "contains":
            where.append(f"{c} ILIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(f.value)}%")
        elif f.op == "is_null":
            where.append(f"{c} IS NULL")
        else:
            where.append(f"{c} IS NOT NULL")

    sql = f"SELECT {sel} FROM \"t\""
    if where:
        sql += " WHERE " + " AND ".join(where)
    if plan.group_by:
        sql += " GROUP BY " + ", ".join(_q(g) for g in plan.group_by)

    order = [f"{_q(b)} {d.upper()} NULLS LAST" for b, d in plan.order_by]
    if plan.group_by:
        order += [f"{_q(g)} ASC NULLS LAST" for g in plan.group_by]   # fixed tie-break
    elif not any(i.agg for i in plan.select):
        order.append(f'"{ROW_COL}" ASC')                               # keep original row order
    if order:
        sql += " ORDER BY " + ", ".join(order)
    sql += f" LIMIT {int(plan.limit)}"
    return sql, params


# ── execution ──────────────────────────────────────────────────────────────
class PlanExecutionError(RuntimeError):
    """A validated plan failed while running (should be rare; reported, not hidden)."""


def execute_plan(plan: Plan, df: pd.DataFrame) -> PlanResult:
    import duckdb

    sql, params = compile_plan(plan, df)
    tbl = df.reset_index(drop=True).copy(deep=False)
    tbl[ROW_COL] = np.arange(len(tbl))
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET threads TO 2")
        con.execute("SET memory_limit='1GB'")
        con.register("t", tbl)
        con.execute("SET enable_external_access=false")
        frame = con.execute(sql, params).df()
    except duckdb.Error as e:
        raise PlanExecutionError(f"{type(e).__name__}: {e}") from e
    finally:
        con.close()
    return PlanResult(frame=frame, sql=sql, params=params, plan=plan)
