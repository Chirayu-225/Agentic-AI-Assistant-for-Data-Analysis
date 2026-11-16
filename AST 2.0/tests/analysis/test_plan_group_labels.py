"""
A grouped answer always names its groups.

Run from the "AST 2.0" folder:
    python -m tests.analysis.test_plan_group_labels

Found by the evaluation gate (INV04, INV05, STO04, EXP12): for "Which region
has the highest total?" the model wrote a plan that grouped by region but did
not select it, so the user saw a bare number, cross-checked and useless.
A number per group without the group's name cannot answer any question, so
the validator adds the group columns to the output itself.
"""
from __future__ import annotations

import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def main():
    from analysis import query_plan as qp, plan_verify as pv

    df = pd.DataFrame({
        "region": ["north", "south", "north", "east", "south", None],
        "product": ["a", "b", "a", "c", "b", "a"],
        "revenue": [100.0, 250.0, 50.0, 75.0, 25.0, 10.0],
    })

    print("=" * 70)
    print("PART 1 - group columns missing from select are added, first")
    print("=" * 70)
    raw = {"select": [{"agg": "sum", "column": "revenue", "as": "total"}],
           "group_by": ["region"], "order_by": [{"by": "total", "dir": "desc"}], "limit": 1}
    plan = qp.validate_plan(raw, df)
    frame = qp.execute_plan(plan, df).frame
    check("the result has the region column", "region" in frame.columns, str(list(frame.columns)))
    check("... placed before the number", list(frame.columns) == ["region", "total"], str(list(frame.columns)))
    check("the top row names the winning region", len(frame) == 1 and frame.iloc[0]["region"] == "south"
          and frame.iloc[0]["total"] == 275.0, frame.to_dict("records").__repr__())
    check("the recorded plan shows the added column (what ran is what is logged)",
          [s.get("column") for s in plan.to_dict()["select"]][0] == "region")
    check("both engines still agree", pv.verify(plan, df, frame).status == "verified")

    print("\n" + "=" * 70)
    print("PART 2 - two group columns, one already selected")
    print("=" * 70)
    raw = {"select": [{"column": "product"}, {"agg": "count", "column": None, "as": "n"}],
           "group_by": ["region", "product"]}
    plan = qp.validate_plan(raw, df)
    frame = qp.execute_plan(plan, df).frame
    check("the missing one is added, the present one is not duplicated",
          list(frame.columns) == ["region", "product", "n"], str(list(frame.columns)))
    check("verified", pv.verify(plan, df, frame).status == "verified")

    print("\n" + "=" * 70)
    print("PART 3 - plans that already name their groups are unchanged")
    print("=" * 70)
    raw = {"select": [{"agg": "sum", "column": "revenue", "as": "total"}, {"column": "region"}],
           "group_by": ["region"]}
    frame = qp.execute_plan(qp.validate_plan(raw, df), df).frame
    check("column order chosen by the plan is kept", list(frame.columns) == ["total", "region"], str(list(frame.columns)))
    raw = {"select": [{"agg": "avg", "column": "revenue", "as": "mean"}]}
    frame = qp.execute_plan(qp.validate_plan(raw, df), df).frame
    check("an ungrouped aggregate is still one bare value", list(frame.columns) == ["mean"] and len(frame) == 1)

    print("\n" + "=" * 70)
    print("PART 4 - an alias may not hide behind an added group column")
    print("=" * 70)
    raw = {"select": [{"agg": "sum", "column": "revenue", "as": "total"}], "group_by": ["region"],
           "order_by": [{"by": "region", "dir": "asc"}]}
    frame = qp.execute_plan(qp.validate_plan(raw, df), df).frame
    check("ordering by an added group column works", list(frame["region"].iloc[:3]) == ["east", "north", "south"],
          str(list(frame["region"])))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
