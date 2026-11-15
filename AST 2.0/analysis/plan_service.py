"""
The model-facing half of the typed query plan (see analysis/query_plan.py).

Asks the model for a JSON plan, validates it, gives it ONE repair attempt that
shows the validator's message, and executes it. Reports why it gave up when it
does, so the caller can fall back to the sandboxed-Python route and the
response can say which route produced the answer.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from analysis import query_plan as qp

PLAN_PROMPT_MARKER = "JSON QUERY PLAN"

_CHART_WORDS = ("plot", "chart", "graph", "visuali", "scatter", "histogram", "heatmap", "pie ")


def plan_path_enabled() -> bool:
    return os.environ.get("AST_QUERY_PLAN", "1").strip().lower() not in ("0", "false", "no", "off")


def wants_chart(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _CHART_WORDS)


def build_plan_prompt(schema_text: str, question: str, error: str | None = None,
                      previous: str | None = None) -> str:
    repair = ""
    if error:
        repair = (
            "\nYour previous reply was rejected by the validator.\n"
            f"Previous reply:\n{(previous or '')[:1500]}\n"
            f"Validator message: {error}\n"
            "Reply again with a corrected plan (or {\"unsupported\": true}).\n"
        )
    return f"""You translate a data question into a {PLAN_PROMPT_MARKER}. You do NOT write code.

Table columns and value spellings (use ONLY these column names, exactly):
{schema_text}

Question:
{question}
{repair}
Reply with ONE JSON object and nothing else. Keys (all optional except select):
  "select":   list of items. Each item is either
                 {{"column": "<column>", "as": "<optional name>"}}                 plain column
                 {{"agg": "<aggregate>", "column": "<column or null>", "as": "<name>"}}  aggregate
             aggregates: count, count_distinct, sum, avg, min, max, median
             (count with "column": null or "*" counts all rows; sum/avg/median need numeric columns)
  "filters":  list of {{"column": "<column>", "op": "<operator>", "value": <value>}}, all combined with AND
             operators: =, !=, <, <=, >, >=, between (value [low, high]), in, not_in (value is a list),
             contains (text, case-insensitive), is_null, not_null (no "value")
             dates are text like "2024-03-01"; text values must match the listed spellings exactly
  "group_by": list of columns. Required when select mixes plain columns with aggregates.
  "order_by": list of {{"by": "<output name>", "dir": "asc" or "desc"}}
  "limit":    integer 1-10000

Rules:
- Output names given with "as" use letters, digits and underscores only and must not equal an existing column name.
- If the question needs something this language cannot express (a chart, a join, a regression, a custom
  formula, anything else), reply exactly: {{"unsupported": true}}
- Never invent columns or values."""


@dataclass
class PlanAttempt:
    result: Optional[qp.PlanResult]
    llm_calls: int
    fallback_reason: Optional[str]
    retried: bool = False


def _unsupported(raw: str) -> bool:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("unsupported") is True


def answer_with_plan(df: pd.DataFrame, question: str, schema_text: str,
                     llm: Callable, session_id: str = "") -> PlanAttempt:
    if not plan_path_enabled():
        return PlanAttempt(None, 0, "plan route disabled")
    if wants_chart(question):
        return PlanAttempt(None, 0, "chart request")

    raw = llm(build_plan_prompt(schema_text, question), temperature=0.0, session_id=session_id)
    calls, retried = 1, False
    if _unsupported(raw):
        return PlanAttempt(None, calls, "model reported the question is outside the plan language")
    try:
        plan = qp.parse_plan(raw, df)
    except qp.PlanError as first_error:
        raw2 = llm(build_plan_prompt(schema_text, question, error=str(first_error), previous=raw),
                   temperature=0.0, session_id=session_id)
        calls, retried = 2, True
        if _unsupported(raw2):
            return PlanAttempt(None, calls, "model reported the question is outside the plan language", True)
        try:
            plan = qp.parse_plan(raw2, df)
        except qp.PlanError as second_error:
            return PlanAttempt(None, calls, f"invalid plan after repair: {second_error}", True)

    try:
        result = qp.execute_plan(plan, df)
    except qp.PlanExecutionError as e:
        return PlanAttempt(None, calls, f"plan failed to execute: {e}", retried)
    return PlanAttempt(result, calls, None, retried)
