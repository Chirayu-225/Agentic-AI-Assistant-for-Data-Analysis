"""
eval_heldout.py — scores the pipeline on the HELD-OUT sets in
eval_data/heldout/ (built by eval_data/heldout/build_heldout.py) and
reports one pooled accuracy with a Wilson interval, plus a per-dataset
and per-category breakdown.

Why a separate held-out set: the dev set (eval_data/sample_*.csv, 24
rows, 9 questions) was iterated against until it hit 100%, so it no
longer measures anything. These three files are messy in the ways real
exports are and must never be used for debugging prompts. If a held-out
question fails, write a NEW dev question that reproduces the failure and
fix against that instead.

Usage:
    python eval_heldout.py                      # real LLM pipeline (GROQ_API_KEY)
    python eval_heldout.py --solver shortcut    # offline, no-LLM baseline
    python eval_heldout.py --repeat 3           # also measures consistency
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from eval_harness import SOLVERS, load_questions, run_eval, summarize, wilson_interval, write_outputs

HELDOUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data", "heldout")
SETS = [
    ("invoices_messy.csv", "invoices_questions.csv"),
    ("store_units_wide.xlsx", "store_units_questions.csv"),
    ("expenses_dirty.csv", "expenses_questions.csv"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--solver", choices=sorted(SOLVERS), default="llm")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args(argv)
    if args.solver == "llm" and not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Use --solver shortcut for the offline baseline.", file=sys.stderr)
        return 2

    pooled, per_set = [], {}
    for data_name, q_name in SETS:
        questions = load_questions(os.path.join(HELDOUT_DIR, q_name))
        with open(os.path.join(HELDOUT_DIR, data_name), "rb") as f:
            data = f.read()
        print(f"\n== {data_name}: {len(questions)} questions x {args.repeat}", flush=True)
        results = run_eval(
            data, data_name, questions, SOLVERS[args.solver], args.repeat,
            session_id=f"heldout-{int(time.time())}",
            on_result=lambda r: print(f"[{'OK  ' if r.correct else 'FAIL'}] {r.id}#{r.run} {r.question[:60]}"
                                      + ("" if r.correct else f"  <- {r.outcome}"), flush=True),
        )
        meta = {"dataset": data_name, "solver": args.solver, "split": "heldout",
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
        summary = summarize(results, args.repeat)
        write_outputs(results, summary, meta, os.path.join(args.out, "heldout"))
        per_set[data_name] = summary
        pooled.extend(results)

    k, n = sum(r.correct for r in pooled), len(pooled)
    lo, hi = wilson_interval(k, n)
    by_cat: dict[str, list[int]] = {}
    for r in pooled:
        by_cat.setdefault(r.category or "uncategorised", [0, 0])
        by_cat[r.category or "uncategorised"][0] += int(r.correct)
        by_cat[r.category or "uncategorised"][1] += 1

    combined = {
        "solver": args.solver, "repeat": args.repeat, "runs": n, "correct": k,
        "accuracy": round(k / n, 3) if n else 0.0, "accuracy95": [round(lo, 3), round(hi, 3)],
        "perDataset": {name: {"accuracy": s["accuracy"], "accuracy95": s["accuracy95"],
                              "outcomes": s["outcomes"]} for name, s in per_set.items()},
        "byCategory": {c: {"correct": a, "n": b, "accuracy": round(a / b, 3)} for c, (a, b) in sorted(by_cat.items())},
    }
    out_dir = os.path.join(args.out, "heldout")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pooled-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nHELD-OUT accuracy {combined['accuracy'] * 100:.1f}% "
          f"(95% interval {lo * 100:.1f}–{hi * 100:.1f}%, {k}/{n} runs)")
    for c, v in combined["byCategory"].items():
        print(f"  {c:<14} {v['correct']}/{v['n']}")
    print(f"Pooled summary: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
