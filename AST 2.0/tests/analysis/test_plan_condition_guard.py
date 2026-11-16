"""
A dropped condition is never shown as cross-checked.

Run from the "AST 2.0" folder:
    python -m tests.analysis.test_plan_condition_guard

Found by the evaluation gate (EXP05): "What is the average expense amount,
excluding refunds?" -> the model wrote a plan with NO filter. Both engines
agreed on the average of everything, so a wrong number was shown as
cross-checked. The cross-check proves the plan was computed correctly; it
cannot know the plan ignored part of the question.

Two defences, both without trusting the model:
  1. GUARD  the question asks to leave something out ("excluding", "except",
            "without", ...) but the plan filters nothing -> the answer is
            still shown, labelled NOT cross-checked, with the reason.
  2. HINT   the column list given to the model says which number columns
            contain negative values, so "refunds" has something to map to.
"""
from __future__ import annotations

import json
import os

import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


NO_FILTER = json.dumps({"select": [{"agg": "avg", "column": "amount", "as": "avg_amount"}]})
WITH_FILTER = json.dumps({"select": [{"agg": "avg", "column": "amount", "as": "avg_amount"}],
                          "filters": [{"column": "amount", "op": ">", "value": 0}]})


def main():
    os.environ.pop("AST_QUERY_PLAN", None)
    from analysis import analysis_service as svc
    from analysis import plan_verify as pv
    from analysis import query_engine as qe
    from analysis import query_plan as qp

    df = pd.DataFrame({
        "dept": ["sales", "sales", "ops", "ops", "ops", "hr"],
        "amount": [100.0, -40.0, 250.0, 50.0, -10.0, 75.0],
        "units": [1, 2, 3, 4, 5, 6],
    })

    print("=" * 70)
    print("PART 1 - the guard on its own")
    print("=" * 70)
    bare = qp.validate_plan(json.loads(NO_FILTER), df)
    filtered = qp.validate_plan(json.loads(WITH_FILTER), df)
    for q in ("What is the average amount, excluding refunds?",
              "Average amount except the ops department",
              "average amount without negative values",
              "Total amount not including refunds",
              "average amount other than hr",
              "Average amount, EXCLUDING refunds",
              "average amount ignoring refunds",
              "What is the average amount if we exclude refunds?"):
        check(f"flagged: {q!r}", bool(pv.dropped_condition(q, bare)))
    for q in ("What is the average amount?",
              "Which dept has the highest amount?",
              "What is the exceptional amount average?",        # 'except' inside a word
              "average amount within the data"):                # 'without' is not 'within'
        check(f"not flagged: {q!r}", pv.dropped_condition(q, bare) is None, str(pv.dropped_condition(q, bare)))
    check("a plan that does filter is not flagged",
          pv.dropped_condition("average amount, excluding refunds", filtered) is None)
    check("the reason quotes the word it found",
          "excluding" in (pv.dropped_condition("average amount, excluding refunds", bare) or ""))

    print("\n" + "=" * 70)
    print("PART 2 - in the pipeline: shown, but not called cross-checked")
    print("=" * 70)
    orig = qe.llm_with_fallback

    def ask(question, reply):
        qe.llm_with_fallback = lambda prompt, temperature=0.1, session_id="", **k: reply
        try:
            return svc.run_analysis_from_cleaned_df(df, question, "s1")
        finally:
            qe.llm_with_fallback = orig

    out = ask("What is the average amount, excluding refunds?", NO_FILTER)
    v = out.get("verification") or {}
    check("the answer is still returned", out.get("success") is True and out.get("method") == "query_plan", repr(out)[:200])
    check("its status is unverified, not verified", v.get("status") == "unverified", repr(v))
    check("the reason is in the response", any("excluding" in r for r in v.get("reasons", [])), repr(v))
    check("the basis text does not claim a cross-check",
          "NOT cross-checked" in str(out.get("basis")) and "excluding" in str(out.get("basis")), str(out.get("basis")))

    out = ask("What is the average amount, excluding refunds?", WITH_FILTER)
    check("the same question with a filtering plan is verified",
          (out.get("verification") or {}).get("status") == "verified", repr(out.get("verification")))
    check("... and gives the right number", abs(out["resultData"][0]["avg_amount"] - 118.75) < 1e-9
          if isinstance(out.get("resultData"), list) else abs(float(out.get("resultData")) - 118.75) < 1e-9,
          repr(out.get("resultData")))

    out = ask("What is the average amount?", NO_FILTER)
    check("a question with no condition is verified as before",
          (out.get("verification") or {}).get("status") == "verified", repr(out.get("verification")))

    print("\n" + "=" * 70)
    print("PART 3 - the model is told which number columns go negative")
    print("=" * 70)
    desc = qe.describe_schema(df)
    amount_line = [l for l in desc.splitlines() if l.startswith("- amount")][0]
    units_line = [l for l in desc.splitlines() if l.startswith("- units")][0]
    check("the negative count is stated", "2 of 6" in amount_line and "negative" in amount_line, amount_line)
    check("columns with no negatives are unchanged", units_line == "- units (number)", units_line)
    allnan = qe.describe_schema(pd.DataFrame({"x": [float("nan")] * 3}))
    check("an all-missing number column does not crash or claim negatives", allnan.strip() == "- x (number)", allnan)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
