"""
Typed query plan: validation, compilation and execution (hazards H1 and H3).

Run from the "AST 2.0" folder:
    python -m tests.analysis.test_query_plan

The model proposes a JSON plan; code validates it against the real schema and
compiles it to parameterised SQL. These tests pin three properties:
  1. Anything outside the closed plan language is REJECTED with a reason.
  2. User-controlled text never reaches the SQL string (values are bound
     parameters, identifiers are quoted by us).
  3. Executed plans give exactly the same numbers as an independent pandas
     computation, with a deterministic row order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def main():
    from analysis.query_plan import PlanError, compile_plan, execute_plan, parse_plan, validate_plan

    df = pd.DataFrame({
        "region": ["north", "south", "north", "east", "south", "north", None],
        "product": ["a", "b", "a", "c", "b", "c", "a"],
        "revenue": [100.0, 250.0, 50.0, 75.0, 25.0, 300.0, 10.0],
        "units": [1, 5, 2, 3, 1, 6, 1],
        "paid": [True, True, False, True, False, True, True],
        "day": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-02-01", "2024-02-10",
                               "2024-03-01", "2024-03-15", "2024-04-01"]),
        "note": ["ok", "50% off", "a_b", None, "ok", "x", "y"],
    })

    def rejects(label, plan, expect=""):
        try:
            validate_plan(plan, df)
            check(label, False, "plan was accepted")
        except PlanError as e:
            check(label, expect.lower() in str(e).lower(), f"wrong reason: {e}")
        except Exception as e:
            check(label, False, f"crashed with {type(e).__name__}: {e}")

    print("=" * 70)
    print("PART 1 - plans outside the language are rejected, with a reason")
    print("=" * 70)
    rejects("unknown column", {"select": [{"column": "nope"}]}, "nope")
    rejects("unknown top-level key", {"select": [{"column": "region"}], "sql": "drop table t"}, "sql")
    rejects("unknown key inside an item", {"select": [{"column": "region", "expr": "1"}]}, "expr")
    rejects("injection through a column name", {"select": [{"column": 'region"; DROP TABLE t; --'}]}, "region")
    rejects("injection through an alias",
            {"select": [{"agg": "sum", "column": "revenue", "as": 'x"; DROP TABLE t; --'}]}, "alias")
    rejects("empty select", {"select": []}, "select")
    rejects("select missing", {}, "select")
    rejects("sum of a text column", {"select": [{"agg": "sum", "column": "region"}]}, "numeric")
    rejects("unknown aggregate", {"select": [{"agg": "stddev", "column": "revenue"}]}, "aggregate")
    rejects("unknown operator", {"select": [{"column": "region"}],
                                 "filters": [{"column": "region", "op": "like", "value": "n%"}]}, "operator")
    rejects("number compared with text", {"select": [{"column": "region"}],
                                          "filters": [{"column": "revenue", "op": ">", "value": "lots"}]}, "revenue")
    rejects("text column ordered with >", {"select": [{"column": "region"}],
                                           "filters": [{"column": "region", "op": ">", "value": "a"}]}, "region")
    rejects("between needs two values", {"select": [{"column": "region"}],
                                         "filters": [{"column": "revenue", "op": "between", "value": [1]}]}, "between")
    rejects("in needs a non-empty list", {"select": [{"column": "region"}],
                                          "filters": [{"column": "region", "op": "in", "value": []}]}, "in")
    rejects("is_null must not carry a value", {"select": [{"column": "region"}],
                                               "filters": [{"column": "region", "op": "is_null", "value": 1}]}, "value")
    rejects("NaN as a value", {"select": [{"column": "region"}],
                               "filters": [{"column": "revenue", "op": ">", "value": float("nan")}]}, "revenue")
    rejects("plain column mixed with an aggregate but not grouped",
            {"select": [{"column": "region"}, {"agg": "sum", "column": "revenue"}]}, "group")
    rejects("group_by without an aggregate", {"select": [{"column": "region"}], "group_by": ["region"]}, "aggregate")
    rejects("group_by unknown column", {"select": [{"agg": "count", "column": None}], "group_by": ["zzz"]}, "zzz")
    rejects("duplicate output names",
            {"select": [{"agg": "sum", "column": "revenue", "as": "t"}, {"agg": "avg", "column": "revenue", "as": "t"}]},
            "duplicate")
    rejects("alias that shadows a real column",
            {"select": [{"agg": "sum", "column": "revenue", "as": "units"}]}, "alias")
    rejects("order_by something not in the output",
            {"select": [{"column": "region"}], "order_by": [{"by": "units", "dir": "asc"}]}, "order")
    rejects("bad direction",
            {"select": [{"column": "region"}], "order_by": [{"by": "region", "dir": "sideways"}]}, "dir")
    rejects("limit too large", {"select": [{"column": "region"}], "limit": 10_000_000}, "limit")
    rejects("limit not an integer", {"select": [{"column": "region"}], "limit": "ten"}, "limit")
    rejects("a version we do not know", {"version": 99, "select": [{"column": "region"}]}, "version")
    rejects("not a dict at all", ["select"], "plan")

    print("\n" + "=" * 70)
    print("PART 2 - parsing model output")
    print("=" * 70)
    p = parse_plan('```json\n{"select": [{"column": "region"}]}\n```', df)
    check("fenced JSON is accepted", p is not None)
    p = parse_plan('Here you go: {"select": [{"column": "region"}]} hope it helps', df)
    check("JSON surrounded by chatter is accepted", p is not None)
    for label, text in [("garbage text", "I cannot do that"), ("empty string", ""),
                        ("oversized output", '{"select": [' + '{"column": "region"},' * 20000 + "]}")]:
        try:
            parse_plan(text, df)
            check(f"parse rejects: {label}", False, "accepted")
        except PlanError:
            check(f"parse rejects: {label}", True)

    print("\n" + "=" * 70)
    print("PART 3 - compilation: user text never reaches the SQL string")
    print("=" * 70)
    secret = "Robert'); DROP TABLE students;--"
    plan = validate_plan({"select": [{"column": "region"}],
                          "filters": [{"column": "region", "op": "=", "value": secret},
                                      {"column": "note", "op": "contains", "value": "50%"}]}, df)
    sql, params = compile_plan(plan, df)
    check("filter values are bound parameters, not in the SQL text", secret not in sql and "50" not in sql, sql)
    check("the values are in the parameter list", secret in params)
    weird = pd.DataFrame({'my "col"': [1, 2, 3], "grp": ["x", "y", "x"]})
    wplan = validate_plan({"select": [{"agg": "sum", "column": 'my "col"', "as": "total"}]}, weird)
    wres = execute_plan(wplan, weird)
    check("a column name containing a double quote works end to end", int(wres.frame["total"].iloc[0]) == 6)

    print("\n" + "=" * 70)
    print("PART 4 - executed plans equal an independent pandas computation")
    print("=" * 70)

    def run(plan_dict):
        return execute_plan(validate_plan(plan_dict, df), df).frame

    out = run({"select": [{"column": "region"}, {"agg": "sum", "column": "revenue", "as": "total"}],
               "group_by": ["region"], "order_by": [{"by": "total", "dir": "desc"}], "limit": 2})
    ref = df.groupby("region")["revenue"].sum().sort_values(ascending=False).head(2)
    check("group + sum + order desc + limit", out["region"].tolist() == ref.index.tolist()
          and np.allclose(out["total"], ref.values), repr(out.to_dict("list")))

    out = run({"select": [{"agg": "count", "column": None}]})
    check("count(*) counts every row, including nulls", int(out.iloc[0, 0]) == 7)
    out = run({"select": [{"agg": "count", "column": "*", "as": "n"}]})
    check('count with column "*" (the SQL spelling models use) means count all rows', int(out["n"].iloc[0]) == 7)
    rejects('"*" is only allowed for count', {"select": [{"agg": "sum", "column": "*"}]}, "column")
    rejects('"*" as a plain column is not a column', {"select": [{"column": "*"}]}, "column")
    out = run({"select": [{"agg": "count", "column": "region", "as": "n"}]})
    check("count(column) skips nulls", int(out["n"].iloc[0]) == 6)
    out = run({"select": [{"agg": "count_distinct", "column": "region", "as": "n"}]})
    check("count_distinct skips nulls", int(out["n"].iloc[0]) == 3)
    out = run({"select": [{"agg": "avg", "column": "revenue", "as": "m"}, {"agg": "median", "column": "revenue", "as": "md"},
                          {"agg": "min", "column": "revenue", "as": "lo"}, {"agg": "max", "column": "revenue", "as": "hi"}]})
    check("avg / median / min / max",
          np.isclose(out["m"][0], df.revenue.mean()) and np.isclose(out["md"][0], df.revenue.median())
          and out["lo"][0] == 10 and out["hi"][0] == 300)

    out = run({"select": [{"column": "product"}, {"column": "revenue"}],
               "filters": [{"column": "region", "op": "=", "value": "north"},
                           {"column": "revenue", "op": ">=", "value": 60}]})
    ref = df[(df.region == "north") & (df.revenue >= 60)][["product", "revenue"]]
    check("equals + >= filters, original row order kept", out["revenue"].tolist() == ref["revenue"].tolist())
    out = run({"select": [{"column": "revenue"}], "filters": [{"column": "revenue", "op": "between", "value": [50, 100]}]})
    check("between is inclusive on both ends", sorted(out["revenue"]) == [50.0, 75.0, 100.0])
    out = run({"select": [{"column": "product"}], "filters": [{"column": "product", "op": "in", "value": ["a", "c"]}]})
    check("in", len(out) == 5)
    out = run({"select": [{"column": "product"}], "filters": [{"column": "product", "op": "not_in", "value": ["a"]}]})
    check("not_in", len(out) == 4)
    out = run({"select": [{"column": "product"}], "filters": [{"column": "region", "op": "is_null"}]})
    check("is_null", len(out) == 1)
    out = run({"select": [{"column": "product"}], "filters": [{"column": "region", "op": "not_null"}]})
    check("not_null", len(out) == 6)
    out = run({"select": [{"column": "note"}], "filters": [{"column": "note", "op": "contains", "value": "50%"}]})
    check("contains treats % literally (not as a wildcard)", out["note"].tolist() == ["50% off"])
    out = run({"select": [{"column": "note"}], "filters": [{"column": "note", "op": "contains", "value": "a_b"}]})
    check("contains treats _ literally", out["note"].tolist() == ["a_b"])
    out = run({"select": [{"column": "note"}], "filters": [{"column": "note", "op": "contains", "value": "OK"}]})
    check("contains is case-insensitive", len(out) == 2)
    out = run({"select": [{"column": "units"}], "filters": [{"column": "paid", "op": "=", "value": False}]})
    check("boolean filter", sorted(out["units"]) == [1, 2])
    out = run({"select": [{"column": "units"}], "filters": [{"column": "day", "op": ">=", "value": "2024-03-01"}]})
    check("date filter with an ISO string", len(out) == 3)
    out = run({"select": [{"column": "revenue"}], "filters": [{"column": "region", "op": "!=", "value": "north"}]})
    check("!= excludes nulls (SQL semantics, documented)", len(out) == 3)

    tie = pd.DataFrame({"g": ["b", "a", "c", "a", "b", "c"], "v": [1, 1, 1, 1, 1, 1]})
    tp = validate_plan({"select": [{"column": "g"}, {"agg": "sum", "column": "v", "as": "s"}],
                        "group_by": ["g"], "order_by": [{"by": "s", "dir": "desc"}]}, tie)
    runs = [execute_plan(tp, tie).frame["g"].tolist() for _ in range(5)]
    check("ties break deterministically (same order every run)", all(r == ["a", "b", "c"] for r in runs), repr(runs))

    res = execute_plan(validate_plan({"select": [{"column": "region"}]}, df), df)
    check("default row cap applied", len(res.frame) <= 1000)
    check("the result carries the SQL and parameters for the audit record",
          "SELECT" in res.sql.upper() and isinstance(res.params, list))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
