"""
eval_joins.py — scores the Join Datasets feature (AI-written SQL over
several tables, run by DuckDB in the sandbox) against the held-out set in
eval_data/joins_heldout/ (built by build_joins.py).

Every question gets all four tables, exactly as a user would upload them.
Reports accuracy with a Wilson interval, a breakdown for the TRAP
questions (fan-out, orphans, anti-joins — the classic join errors), and,
with --repeat, how consistently the same question gets the same answer.

How an answer is scored:
  number  — a single-row result containing the expected value (within
            tolerance), OR, for whole-number answers, a result with exactly
            that many rows (e.g. listing the 8 customers instead of "8").
  text    — the expected value appears in the first row.

Usage:
    python eval_joins.py               # needs GROQ_API_KEY
    python eval_joins.py --repeat 3    # + consistency
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "eval_data", "joins_heldout")
TABLES = [("customers", "customers.csv"), ("orders", "orders.csv"),
          ("order_lines", "order_lines.csv"), ("products", "products.csv")]


def _numbers(row: dict) -> list[float]:
    out = []
    for v in row.values():
        try:
            out.append(float(str(v).replace(",", "")))
        except (TypeError, ValueError):
            pass
    return out


def _signature(result: dict) -> str:
    """The answer's VALUES only — column names and column order ignored, so
    {"revenue": 140950} and {"total_revenue": 140950} count as the same
    answer for the consistency measure. (The first version compared column
    names too and reported 64% consistency for answers that were 93%
    identical in value.) Numbers are rounded to 2 decimals."""
    if not result.get("success"):
        return "ERROR"
    rows = (result.get("resultData") or [])[:5]
    def norm(v):
        try:
            return f"{round(float(str(v).replace(',', '')), 2):.2f}"
        except (TypeError, ValueError):
            return str(v).strip().lower()
    return "|".join(",".join(sorted(norm(v) for v in r.values())) for r in rows) + f"#{result.get('totalRowCount', len(rows))}"


def score(result: dict, expected: str, kind: str, tol: float) -> tuple[bool, str]:
    if not result.get("success"):
        return False, f"error: {str(result.get('error'))[:120]}"
    rows = result.get("resultData") or []
    total = result.get("totalRowCount", len(rows))
    if kind == "text":
        first = rows[0] if rows else {}
        ok = any(str(v).strip().lower() == expected.strip().lower() for v in first.values())
        return ok, "" if ok else f"first row: {json.dumps(first)[:120]}"
    exp = float(expected)
    if len(rows) == 1:
        for n in _numbers(rows[0]):
            if (tol == 0 and n == exp) or (tol and abs(n - exp) <= tol * max(abs(exp), 1)):
                return True, ""
    if exp.is_integer() and total == int(exp) and tol == 0:
        return True, "(matched by row count)"
    shown = _numbers(rows[0]) if rows else []
    return False, f"expected {exp:g}, got {shown[:4]} ({total} rows)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 2

    from eval_harness import wilson_interval
    from multi_table_service import run_multi_table_query

    datasets = []
    for alias, fname in TABLES:
        with open(os.path.join(DATA, fname), "rb") as f:
            datasets.append({"alias": alias, "csv_bytes": f.read(), "filename": fname})
    with open(os.path.join(DATA, "join_questions.csv"), encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    rows = []
    for q in questions:
        tol = float(q["tolerance"] or 0)
        for run in range(1, args.repeat + 1):
            t0 = time.time()
            res = run_multi_table_query(datasets, q["question"], f"eval-joins-{q['id']}-run{run}")
            ok, why = score(res, q["expected"], q["answer_type"], tol)
            first = (res.get("resultData") or [{}])[0]
            rows.append({"id": q["id"], "run": run, "correct": ok, "trap": q["trap"], "why": why,
                         "seconds": round(time.time() - t0, 1), "sql": res.get("code", ""),
                         "signature": _signature(res)})
            print(f"[{'OK  ' if ok else 'FAIL'}] {q['id']}#{run} {q['question'][:62]}"
                  + (f"  [{q['trap']}]" if q["trap"] else "") + ("" if ok else f"  <- {why}"), flush=True)

    k, n = sum(r["correct"] for r in rows), len(rows)
    lo, hi = wilson_interval(k, n)
    trap = [r for r in rows if r["trap"]]
    consistency = None
    if args.repeat > 1:
        by_q: dict[str, list[str]] = {}
        for r in rows:
            by_q.setdefault(r["id"], []).append(r["signature"])
        # A question only counts as consistent if every run produced an
        # answer and they all agree — repeated errors aren't "consistent".
        consistency = round(sum(len(set(v)) == 1 and "ERROR" not in v for v in by_q.values()) / len(by_q), 3)

    summary = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "accuracy": round(k / n, 3), "accuracy95": [round(lo, 3), round(hi, 3)], "runs": n,
        "trapAccuracy": round(sum(r["correct"] for r in trap) / len(trap), 3) if trap else None,
        "consistency": consistency, "results": rows,
    }
    out_dir = os.path.join(HERE, "eval_results", "joins")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nJOIN accuracy {k}/{n} ({100 * k / n:.1f}%, 95% interval {lo * 100:.1f}–{hi * 100:.1f}%)")
    if trap:
        print(f"Trap questions (fan-out / orphans / anti-joins): {sum(r['correct'] for r in trap)}/{len(trap)}")
    if consistency is not None:
        print(f"Consistency: {consistency * 100:.0f}% of questions gave the same answer on every run")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
