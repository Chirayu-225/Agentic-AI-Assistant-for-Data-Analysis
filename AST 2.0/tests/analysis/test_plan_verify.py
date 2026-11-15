"""
Double computation and abstain (hazard H1: a wrong number shown as verified).

Run from the "AST 2.0" folder:
    python -m tests.analysis.test_plan_verify

Every query-plan answer is computed twice: by the SQL engine (query_plan) and
by an independent pandas implementation (plan_verify) that shares no code with
the SQL compiler. These tests pin four properties:
  1. On a healthy system the two computations agree (fixed cases + a seeded
     random differential test), so correct answers are never withheld.
  2. If the SQL engine is wrong in any of the classic ways (dropped filter,
     wrong aggregate, wrong order, wrong limit, altered value, missing row),
     the verifier says so.
  3. Float rounding noise from a different summation order is NOT a disagreement.
  4. In the real pipeline a disagreement means the answer is withheld
     ("I can't verify this"), not shown, and nothing falls back silently.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def make_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "east", "West", "nörth"], n).astype(object)
    region[rng.random(n) < 0.08] = None
    amount = np.round(rng.normal(100, 60, n), 2)
    amount[rng.random(n) < 0.07] = np.nan
    return pd.DataFrame({
        "region": region,
        "product": rng.choice(["a", "b", "c"], n),
        "amount": amount,
        "units": rng.integers(0, 6, n),
        "paid": rng.random(n) < 0.6,
        "day": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D"),
        "note": rng.choice(["ok", "50% off", "a_b", "Late", "late delivery"], n),
    })


def random_plan(rng: random.Random, df: pd.DataFrame) -> dict:
    num, txt = ["amount", "units"], ["region", "product", "note"]
    filters = []
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        kind = rng.choice(["num", "txt", "null", "bool", "date", "in", "between", "contains"])
        if kind == "num":
            filters.append({"column": rng.choice(num), "op": rng.choice(["<", "<=", ">", ">=", "=", "!="]),
                            "value": rng.choice([0, 2, 3, 50, 100.5, 150])})
        elif kind == "txt":
            filters.append({"column": rng.choice(txt), "op": rng.choice(["=", "!="]),
                            "value": rng.choice(["north", "a", "ok", "West", "zzz"])})
        elif kind == "null":
            filters.append({"column": rng.choice(["region", "amount"]), "op": rng.choice(["is_null", "not_null"])})
        elif kind == "bool":
            filters.append({"column": "paid", "op": "=", "value": rng.choice([True, False])})
        elif kind == "date":
            filters.append({"column": "day", "op": rng.choice(["<", ">="]), "value": "2024-03-01"})
        elif kind == "in":
            filters.append({"column": "region", "op": rng.choice(["in", "not_in"]), "value": ["north", "east"]})
        elif kind == "between":
            filters.append({"column": "amount", "op": "between", "value": [40, 120]})
        else:
            filters.append({"column": "note", "op": "contains", "value": rng.choice(["LATE", "50%", "a_b", "o"])})
    shape = rng.choice(["rows", "agg", "group", "group"])
    plan: dict = {"filters": filters}
    aggs = [("count", None), ("count", "region"), ("count_distinct", "product"), ("sum", "amount"), ("sum", "units"),
            ("avg", "amount"), ("min", "amount"), ("max", "units"), ("median", "amount"), ("min", "day"), ("max", "day")]
    if shape == "rows":
        cols = rng.sample(list(df.columns), rng.randint(1, 4))
        plan["select"] = [{"column": c} for c in cols]
        if rng.random() < 0.6:
            plan["order_by"] = [{"by": rng.choice(cols), "dir": rng.choice(["asc", "desc"])}]
    else:
        picked = rng.sample(aggs, rng.randint(1, 3))
        sel = [{"agg": a, "column": c, "as": f"m{k}"} for k, (a, c) in enumerate(picked)]
        if shape == "group":
            gb = rng.sample(["region", "product", "paid", "units", "day"], rng.randint(1, 2))
            plan["group_by"] = gb
            shown = [g for g in gb if rng.random() < 0.8]
            sel = [{"column": g} for g in shown] + sel
            if rng.random() < 0.7:
                plan["order_by"] = [{"by": rng.choice([s.get("as") or s["column"] for s in sel]),
                                     "dir": rng.choice(["asc", "desc"])}]
        plan["select"] = sel
    if rng.random() < 0.5:
        plan["limit"] = rng.choice([1, 3, 10, 500])
    return plan


def main():
    from analysis import query_plan as qp
    from analysis import plan_verify as pv

    df = make_df(400, seed=7)

    def both(plan_dict, frame_df=df):
        plan = qp.validate_plan(plan_dict, frame_df)
        return plan, qp.execute_plan(plan, frame_df).frame

    print("=" * 70)
    print("PART 1 - a healthy system: both computations agree")
    print("=" * 70)
    base = {"select": [{"column": "region"}, {"agg": "sum", "column": "amount", "as": "total"},
                       {"agg": "count", "column": None, "as": "n"}],
            "filters": [{"column": "units", "op": ">=", "value": 2}],
            "group_by": ["region"], "order_by": [{"by": "total", "dir": "desc"}], "limit": 3}
    plan, frame = both(base)
    v = pv.verify(plan, df, frame)
    check("grouped sum with filter, order and limit is verified", v.status == "verified", "; ".join(v.reasons))
    check("the verification lists the checks that ran", len(v.checks) >= 2, repr(v.checks))

    empty = df.iloc[0:0]
    for label, pd_ in [
        ("aggregate over zero rows", {"select": [{"agg": "sum", "column": "amount", "as": "s"}, {"agg": "count", "column": None, "as": "n"}],
                                      "filters": [{"column": "units", "op": ">", "value": 99}]}),
        ("group by with a NULL group", {"select": [{"column": "region"}, {"agg": "avg", "column": "amount", "as": "m"}], "group_by": ["region"]}),
        ("group key not shown in the output", {"select": [{"agg": "sum", "column": "units", "as": "s"}], "group_by": ["product"]}),
        ("plain rows keep the original order", {"select": [{"column": "note"}, {"column": "amount"}], "limit": 50}),
        ("order by a column with NULLs, descending", {"select": [{"column": "amount"}, {"column": "units"}],
                                                     "order_by": [{"by": "amount", "dir": "desc"}], "limit": 400}),
    ]:
        plan, frame = both(pd_)
        v = pv.verify(plan, df, frame)
        check(label, v.status == "verified", "; ".join(v.reasons))
    plan = qp.validate_plan({"select": [{"column": "region"}, {"agg": "count", "column": None, "as": "n"}], "group_by": ["region"]}, empty)
    v = pv.verify(plan, empty, qp.execute_plan(plan, empty).frame)
    check("an empty table", v.status == "verified", "; ".join(v.reasons))

    rng = random.Random(20261115)
    bad = []
    N = 150
    for k in range(N):
        d = make_df(rng.choice([1, 30, 250]), seed=k)
        p = random_plan(rng, d)
        try:
            plan = qp.validate_plan(p, d)
        except qp.PlanError:
            continue
        v = pv.verify(plan, d, qp.execute_plan(plan, d).frame)
        if v.status != "verified":
            bad.append((json.dumps(p), v.status, v.reasons[:2]))
    check(f"seeded random differential test: SQL engine == pandas reference on {N} random plans",
          not bad, f"{len(bad)} mismatch(es); first: {bad[:1]}")

    print("\n" + "=" * 70)
    print("PART 2 - a broken engine is caught")
    print("=" * 70)
    plan, good = both(base)

    def wrong(mutated):
        return qp.execute_plan(qp.validate_plan(mutated, df), df).frame

    faults = {
        "the filter was silently dropped": wrong({**base, "filters": []}),
        "the wrong aggregate ran (avg instead of sum)": wrong(
            {**base, "select": [base["select"][0], {"agg": "avg", "column": "amount", "as": "total"}, base["select"][2]]}),
        "sorted the wrong way": wrong({**base, "order_by": [{"by": "total", "dir": "asc"}], "limit": 1000}).head(3),
        "one row is missing": good.iloc[:-1],
        "an extra row appeared": pd.concat([good, good.iloc[[0]]], ignore_index=True),
        "a column was renamed": good.rename(columns={"total": "sum"}),
    }
    t = good.copy(); t.loc[0, "total"] = t.loc[0, "total"] + 0.01
    faults["one value is off by 0.01"] = t
    t = good.copy(); t.loc[0, "n"] = t.loc[0, "n"] + 1
    faults["one count is off by one"] = t
    t = good.copy(); t.loc[0, "region"] = "nowhere"
    faults["a group label is wrong"] = t
    faults["rows are right but in the wrong order"] = good.iloc[::-1].reset_index(drop=True)
    for label, frame in faults.items():
        v = pv.verify(plan, df, frame)
        check(f"caught: {label}", v.status == "disagreement" and bool(v.reasons), f"status={v.status}")

    print("\n" + "=" * 70)
    print("PART 3 - rounding noise is not a disagreement")
    print("=" * 70)
    t = good.copy(); t["total"] = t["total"] * (1 + 1e-13)
    check("a 1e-13 relative difference is accepted", pv.verify(plan, df, t).status == "verified")
    t = good.copy(); t["total"] = t["total"] * (1 + 1e-6)
    check("a 1e-6 relative difference is not", pv.verify(plan, df, t).status == "disagreement")

    print("\n" + "=" * 70)
    print("PART 4 - invariants hold on their own (they do not use the reference)")
    print("=" * 70)

    def inv(plan_dict, frame):
        return pv.check_invariants(qp.validate_plan(plan_dict, df), df, frame)

    p1 = {"select": [{"agg": "count", "column": None, "as": "n"}]}
    check("a count larger than the table is impossible", bool(inv(p1, pd.DataFrame({"n": [len(df) + 1]}))))
    check("a negative count is impossible", bool(inv(p1, pd.DataFrame({"n": [-1]}))))
    check("an ungrouped aggregate has exactly one row", bool(inv(p1, pd.DataFrame({"n": [1, 2]}))))
    p2 = {"select": [{"agg": "min", "column": "amount", "as": "lo"}, {"agg": "avg", "column": "amount", "as": "mid"},
                     {"agg": "max", "column": "amount", "as": "hi"}]}
    check("min <= avg <= max", bool(inv(p2, pd.DataFrame({"lo": [5.0], "mid": [4.0], "hi": [9.0]}))))
    check("a max outside the column's range is impossible",
          bool(inv(p2, pd.DataFrame({"lo": [5.0], "mid": [6.0], "hi": [1e9]}))))
    p3 = {"select": [{"column": "product"}, {"agg": "count", "column": None, "as": "n"},
                     {"agg": "sum", "column": "units", "as": "u"}], "group_by": ["product"]}
    okf = qp.execute_plan(qp.validate_plan(p3, df), df).frame
    check("a correct grouped result passes every invariant", not inv(p3, okf), repr(inv(p3, okf)))
    t = okf.copy(); t.loc[0, "n"] -= 5
    check("group counts must add up to the row count", bool(inv(p3, t)))
    t = okf.copy(); t.loc[0, "u"] += 7
    check("group sums must add up to the column total", bool(inv(p3, t)))
    check("more rows than the limit is impossible",
          bool(inv({"select": [{"column": "note"}], "limit": 2}, pd.DataFrame({"note": ["ok"] * 3}))))

    print("\n" + "=" * 70)
    print("PART 5 - in the real pipeline a disagreement is withheld, not shown")
    print("=" * 70)
    os.environ.pop("AST_QUERY_PLAN", None)
    from analysis import analysis_service as svc
    from analysis import query_engine as qe

    orig_llm, orig_exec, orig_sandbox, orig_ref = qe.llm_with_fallback, qp.execute_plan, svc.run_sandboxed, pv.reference_execute
    state = {"sandbox": 0, "llm": 0}

    def fake_llm(prompt, temperature=0.1, session_id=""):
        state["llm"] += 1
        return json.dumps(base)

    def counting_sandbox(*a, **k):
        state["sandbox"] += 1
        return orig_sandbox(*a, **k)

    def buggy_engine(plan_, df_):  # a compiler bug that drops every filter
        res = orig_exec(plan_, df_)
        nofilter = qp.Plan(select=plan_.select, filters=(), group_by=plan_.group_by,
                           order_by=plan_.order_by, limit=plan_.limit)
        return qp.PlanResult(frame=orig_exec(nofilter, df_).frame, sql=res.sql, params=res.params, plan=plan_)

    qe.llm_with_fallback, svc.run_sandboxed = fake_llm, counting_sandbox
    try:
        r = svc.run_analysis_from_cleaned_df(df, "total amount per region for 2+ units, top 3", "s1")
        ver = r.get("verification") or {}
        check("healthy: the answer is shown and marked verified",
              r.get("success") is True and ver.get("status") == "verified", repr(r)[:200])
        check("healthy: the basis says it was cross-checked", "cross-check" in str(r.get("basis", "")).lower(), repr(r.get("basis")))

        qp.execute_plan = buggy_engine
        state["sandbox"] = state["llm"] = 0
        r = svc.run_analysis_from_cleaned_df(df, "total amount per region for 2+ units, top 3", "s1")
        ver = r.get("verification") or {}
        check("broken engine: the request does not succeed", r.get("success") is False, repr(r)[:200])
        check("broken engine: no numbers are returned", r.get("resultData") is None and r.get("resultType") == "error")
        check("broken engine: marked as abstained with the reason",
              r.get("abstained") is True and ver.get("status") == "disagreement" and bool(ver.get("reasons")))
        check("broken engine: the message says it could not be verified", "verif" in str(r.get("error", "")).lower())
        check("broken engine: no silent fallback to generated code", state["sandbox"] == 0 and state["llm"] == 1,
              f"sandbox={state['sandbox']} llm={state['llm']}")
        qp.execute_plan = orig_exec

        def crashing_reference(*a, **k):
            raise RuntimeError("simulated verifier bug")
        pv.reference_execute = crashing_reference
        r = svc.run_analysis_from_cleaned_df(df, "total amount per region for 2+ units, top 3", "s1")
        ver = r.get("verification") or {}
        check("verifier crash: the answer is shown but labelled NOT cross-checked",
              r.get("success") is True and ver.get("status") == "unverified"
              and "not" in str(r.get("basis", "")).lower(), repr(r.get("basis")))
    finally:
        qe.llm_with_fallback, qp.execute_plan, svc.run_sandboxed, pv.reference_execute = orig_llm, orig_exec, orig_sandbox, orig_ref

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
