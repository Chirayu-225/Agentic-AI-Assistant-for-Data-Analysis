"""
test_dup_log_truthful.py - the cleaning log never says a row was KEPT when it was removed.

The duplicate warnings are written early in rule_based_clean; a later rule removes rows
that repeat a unique record ID. Before this fix the log could say "8 row(s) ... KEPT"
about rows that were in fact removed (seen on eval_data/heldout/expenses_dirty.csv:
312 rows in, 300 out, 0 copies left, log still said 8 + 4 were kept).
Run: python -m tests.data_cleaning.test_dup_log_truthful
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_cleaning import cleaning  # noqa: E402

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def clean(df):
    out = cleaning.rule_based_clean(df)
    return out[0], out[1]


def kept_lines(log):
    return [l for l in log if "KEPT" in l]


def main():
    ids = [f"E{i:03d}" for i in range(20)]
    base = pd.DataFrame({"expense_id": ids, "dept": ["Sales", "Ops"] * 10,
                         "amount": [float(i * 10) for i in range(20)]})

    # 1. Exact copies that share a unique record ID: removed by the same-ID rule.
    df = pd.concat([base, base.iloc[[0, 1, 2]]], ignore_index=True)
    out, log = clean(df)
    check("copies with the same record ID are removed", len(out) == 20 and int(out.duplicated().sum()) == 0, f"{len(out)} rows")
    check("... and the log does not claim they were KEPT", not kept_lines(log), str(kept_lines(log)))
    check("... it says they were removed", any("Removed 3 repeated record" in l for l in log), str(log))

    # 2. Rows that only become identical after formatting, same ID: also removed, no KEPT line.
    df = pd.concat([base, base.iloc[[3, 4]].assign(dept=lambda d: d["dept"].str.upper())], ignore_index=True)
    out, log = clean(df)
    check("formatting-only copies with the same ID are removed, and not reported as KEPT",
          int(out.duplicated().sum()) == 0 and not kept_lines(log), str(kept_lines(log)))

    # 3. Genuine repeats with no ID column: really kept, and the count is right.
    noid = base.drop(columns="expense_id")
    df = pd.concat([noid, noid.iloc[[0, 1]]], ignore_index=True)
    out, log = clean(df)
    k = kept_lines(log)
    check("identical rows without an ID are still KEPT", len(out) == 22, f"{len(out)} rows")
    check("... and the log says how many, correctly",
          len(k) == 1 and k[0].startswith(f"⚠ {int(out.duplicated().sum())} row(s) are exact copies"), str(k))

    # 4. No ID column, copies only after formatting: kept and reported as such.
    df = pd.concat([noid, noid.iloc[[5]].assign(dept=lambda d: d["dept"].str.upper())], ignore_index=True)
    out, log = clean(df)
    k = kept_lines(log)
    check("rows identical only after formatting are KEPT and reported once",
          int(out.duplicated().sum()) == 1 and len(k) == 1 and "became identical after formatting" in k[0], str(k))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
