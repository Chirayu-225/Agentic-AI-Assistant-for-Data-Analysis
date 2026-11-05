"""
exp_questions.py — the question set for the experiment "does cleaning change analysis answers?".

Each question is tied to ONE table built by clean_bench's `exp` profile and to the defect that table
carries. Ground truth is computed here, in the standard library, from truth.csv (the clean table), by
code that shares nothing with the cleaning pipeline.

THE "CLEAN" ARM IS THE ATTAINABLE TABLE, NOT THE FULL TRUTH
    Some defects destroy information: once a value is replaced by "N/A", a future date, a negative quantity, a swapped
    end date or an unknown customer id, no cleaner, reviewer or model can recover the original. Scoring against the full
    clean table would count that unrecoverable gap as headroom and push towards validators that can only flag. So the
    ceiling is what a perfect cleaner could deliver FROM THE DIRTY FILE: every recoverable defect repaired, every
    unrecoverable cell blanked. `expected` is computed on that table; `expected_true` (the full clean table) is kept
    so the report can show where information was lost (`info_lost`).

Two tiers:
  single    one defect type per table, so a wrong answer has one cause. Easiest case for a model that
            compensates for dirty data, so a small gap here is a LOWER BOUND on the headroom.
  compound  2-3 defects stacked on one table, like real files.

"kind" separates the two kinds of gain the experiment must not blur:
  structural  defects Stage 0 is built to fix (title lines, footers, ragged rows, BOM, trailing delimiter)
  value       everything else (the old reader still returns an answer, and the answer can be silently wrong)

Answers are checked deterministically by eval_harness.check_answer (no LLM judge): counts and years must be
exact, sums/averages/maxima/shares within 0.1% (relative), tenure within 1%.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from decimal import Decimal

import clean_bench as cb

# defects whose injected value cannot be turned back into the original by any cleaner
UNRECOVERABLE = {"sentinel_null", "future_date", "impossible_value", "end_before_start", "orphan_fk"}

ENTITY = {"orders": "orders", "staff": "employees", "inventory": "inventory items"}
COLS = {"amount": "order amount", "unit_price": "unit price", "quantity": "quantity ordered", "salary": "salary",
        "on_hand": "quantity on hand", "weight": "weight in kilograms"}
TOL_EXACT, TOL_NUM, TOL_TENURE = 0.0, 0.001, 0.01

C = lambda **k: k
PLAN = {
    # tag: (tier, kind, [ops])
    "footer": ("single", "structural", [C(op="count_rows"), C(op="sum", col="amount")]),
    "total": ("single", "structural", [C(op="sum", col="amount"), C(op="count_rows")]),
    "header": ("single", "structural", [C(op="count_rows"), C(op="avg", col="salary")]),
    "ragged": ("single", "structural", [C(op="count_rows"), C(op="sum", col="on_hand")]),
    "bomtrail": ("single", "structural", [C(op="count_rows"), C(op="avg", col="salary")]),
    "dupexact": ("single", "value", [C(op="count_rows"), C(op="count_distinct", col="order_id")]),
    "dupnear": ("single", "value", [C(op="count_rows"), C(op="sum", col="salary")]),
    "sentinel": ("single", "value", [C(op="avg", col="amount"), C(op="sum", col="amount")]),
    "numtext": ("single", "value", [C(op="sum", col="amount"), C(op="avg", col="amount")]),
    "outlier": ("single", "value", [C(op="avg", col="unit_price"), C(op="max", col="unit_price")]),
    "impossible": ("single", "value", [C(op="sum", col="quantity"), C(op="avg", col="quantity")]),
    "future": ("single", "value", [C(op="count_year", col="order_date", year=2025), C(op="max_year", col="order_date")]),
    "datemix": ("single", "value", [C(op="count_year", col="order_date", year=2024), C(op="count_year", col="order_date", year=2023)]),
    "catvar": ("single", "value", [C(op="count_value", col="category", value="Electronics"), C(op="share_value", col="category", value="Electronics")]),
    "whitespace": ("single", "value", [C(op="count_value", col="status", value="Completed"), C(op="share_value", col="status", value="Completed")]),
    "swap": ("single", "value", [C(op="count_value", col="region", value="East"), C(op="count_value", col="region", value="South")]),
    "unit": ("single", "value", [C(op="avg", col="weight"), C(op="max", col="weight")]),
    "pct": ("single", "value", [C(op="avg_pct", col="discount")]),
    "zeros": ("single", "value", [C(op="count_distinct", col="customer_id")]),
    "endstart": ("single", "value", [C(op="avg_tenure_years"), C(op="count_rows")]),
    "orphan": ("single", "value", [C(op="join_count_tier", value="Gold"), C(op="join_sum_tier", value="Gold")]),
    "c1": ("compound", "value", [C(op="count_value", col="category", value="Electronics"), C(op="avg", col="amount")]),
    "c2": ("compound", "value", [C(op="sum", col="amount"), C(op="count_rows")]),
    "c3": ("compound", "value", [C(op="count_year", col="order_date", year=2025), C(op="sum", col="quantity")]),
    "c4": ("compound", "value", [C(op="avg_tenure_years"), C(op="avg", col="salary")]),
    "c5": ("compound", "value", [C(op="avg", col="weight"), C(op="count_rows")]),
}


def question_text(op: dict, family: str) -> str:
    e, o, col = ENTITY[family], op["op"], op.get("col")
    if o == "count_rows":
        return f"How many {e} are there in total?"
    if o == "count_distinct":
        return {"order_id": "How many distinct order IDs are there?",
                "customer_id": "How many distinct customer IDs appear in the data?"}[col]
    if o == "sum":
        return f"What is the total {COLS[col]} across all {e}?"
    if o == "avg":
        return f"What is the average {COLS[col]} across all {e}?"
    if o == "max":
        return f"What is the highest {COLS[col]}?"
    if o == "count_value":
        return {"category": f"How many orders are in the {op['value']} category?",
                "status": f"How many orders have the status {op['value']}?",
                "region": f"How many orders are from the {op['value']} region?"}[col]
    if o == "share_value":
        return {"category": f"What percentage of orders are in the {op['value']} category?",
                "status": f"What percentage of orders have the status {op['value']}?"}[col]
    if o == "count_year":
        return f"How many orders were placed in {op['year']}?"
    if o == "max_year":
        return "In which year was the most recent order placed?"
    if o == "avg_pct":
        return "What is the average discount percentage across all orders?"
    if o == "avg_tenure_years":
        return "What is the average length of service, in years, of employees who have left (from start date to end date)?"
    if o == "join_count_tier":
        return f"How many orders were placed by customers in the {op['value']} loyalty tier?"
    if o == "join_sum_tier":
        return f"What is the total order amount from customers in the {op['value']} loyalty tier?"
    raise ValueError(o)


def _num(s):
    return Decimal(s)


def truth(op: dict, rows: list, customers: dict) -> tuple:
    """-> (expected string, answer_type, tolerance). `rows` = truth.csv rows as dicts of canonical text."""
    o, col = op["op"], op.get("col")
    vals = lambda: [r[col] for r in rows if r[col] != ""]
    if o == "count_rows":
        return str(len(rows)), "number", TOL_EXACT
    if o == "count_distinct":
        return str(len({r[col] for r in rows if r[col] != ""})), "number", TOL_EXACT
    if o in ("sum", "avg", "max"):
        v = [_num(x) for x in vals()]
        r = sum(v) if o == "sum" else (sum(v) / len(v) if o == "avg" else max(v))
        return f"{float(r):.6f}", "number", TOL_NUM
    if o == "count_value":
        return str(sum(1 for r in rows if r[col] == op["value"])), "number", TOL_EXACT
    if o == "share_value":
        return f"{100 * sum(1 for r in rows if r[col] == op['value']) / len(rows):.6f}", "number", TOL_NUM
    if o == "count_year":
        return str(sum(1 for r in rows if r[col] and int(r[col][:4]) == op["year"])), "number", TOL_EXACT
    if o == "max_year":
        return str(max(int(r[col][:4]) for r in rows if r[col])), "number", TOL_EXACT
    if o == "avg_pct":
        v = [_num(x) for x in vals()]
        return f"{float(sum(v) / len(v) * 100):.6f}", "number", TOL_NUM
    if o == "avg_tenure_years":
        import datetime as dt
        d = [(dt.date.fromisoformat(r["end_date"]) - dt.date.fromisoformat(r["start_date"])).days
             for r in rows if r["end_date"]]
        return f"{sum(d) / len(d) / 365.25:.6f}", "number", TOL_TENURE
    if o in ("join_count_tier", "join_sum_tier"):
        sel = [r for r in rows if customers.get(r["customer_id"]) == op["value"]]
        return (str(len(sel)), "number", TOL_EXACT) if o == "join_count_tier" else \
               (f"{float(sum(_num(r['amount']) for r in sel)):.6f}", "number", TOL_NUM)
    raise ValueError(o)


def attainable_rows(files: dict, doc: dict) -> list:
    """truth.csv rows with the cells a perfect cleaner could not recover blanked (see the module docstring)."""
    rows = list(csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8"))))
    by_line = {int(r["__line"]): r for r in rows}
    names = [c["name"] for c in doc["meta"]["columns"]]
    for L in doc["labels"]:
        if L["kind"] != "defect" or L["line"] not in by_line:
            continue
        row = by_line[L["line"]]
        if L["scope"] == "cell" and L["defect"] in UNRECOVERABLE:
            for c in L["cells"]:
                row[c["column"]] = ""
        if L["defect"] == "ragged_row" and L["extra"]["fields_found"] < L["extra"]["fields_expected"]:
            for name in names[L["extra"]["fields_found"]:]:  # the trailing fields a short row never had
                row[name] = ""
    return rows


def attainable_csv(files: dict, doc: dict) -> bytes:
    rows = attainable_rows(files, doc)
    names = [c["name"] for c in doc["meta"]["columns"]]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(names)
    for r in rows:
        w.writerow([r[n] for n in names])
    return out.getvalue().encode("utf-8")


def build_exp():
    """-> (cases {case_id: (spec, files, doc)}, questions [dict]). Deterministic."""
    cases, questions = {}, []
    for spec in cb.specs_for("exp", "dev"):
        files, doc = cb.build_case(spec)
        cases[spec.case_id] = (spec, files, doc)
        tag = spec.flags["tag"]
        tier, kind, ops = PLAN[tag]
        full = list(csv.DictReader(io.StringIO(files["truth.csv"].decode("utf-8"))))
        rows = attainable_rows(files, doc)
        customers = {}
        if "customers.csv" in files:
            customers = {r["customer_id"]: r.get("tier") for r in csv.DictReader(io.StringIO(files["customers.csv"].decode("utf-8")))}
        for i, op in enumerate(ops, start=1):
            expected, atype, tol = truth(op, rows, customers)
            expected_true, _, _ = truth(op, full, customers)
            questions.append({"id": f"{tag}-{i}", "tag": tag, "tier": tier, "kind": kind, "case_id": spec.case_id,
                              "family": spec.family, "question": question_text(op, spec.family), "expected": expected,
                              "expected_true": expected_true, "info_lost": expected != expected_true,
                              "answer_type": atype, "tolerance": tol, "op": op,
                              "join": op["op"].startswith("join_")})
    return cases, questions


def questions_sha256(questions: list) -> str:
    return hashlib.sha256(json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def data_sha256() -> str:
    return cb.build("exp", splits=("dev",))["splits"]["dev"]["sha256"]
