"""
The plan prompt tells the model when NOT to write a plan.

Run from the "AST 2.0" folder:
    python -m tests.analysis.test_plan_prompt_rules

Found by the evaluation gate:
  EXP05  "average ..., excluding refunds"  -> the plan had no filter at all.
  STO03/STO08/STO11  yearly totals on a table with one column per month ->
         the language cannot add columns together, and the model improvised
         (twelve separate sums, ranked by January) where it should have
         answered {"unsupported": true} and let the Python route handle it.

A prompt is a request, not a guarantee, so this test only pins that the rules
are present and survive the repair prompt. Whether the model obeys them is
measured by the evaluation gate (python -m evaluation.eval_gate).
"""
from __future__ import annotations

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def main():
    from analysis.plan_service import build_plan_prompt

    first = build_plan_prompt("region: text", "Which region is highest?")
    repair = build_plan_prompt("region: text", "Which region is highest?", error="bad", previous="{}")
    for name, p in (("first prompt", first), ("repair prompt", repair)):
        low = p.lower()
        check(f"{name}: every condition in the question must become a filter",
              "every condition" in low and "filter" in low)
        check(f"{name}: a condition that cannot be a filter means unsupported",
              "cannot be written as a filter" in low)
        check(f"{name}: combining several columns of one row means unsupported",
              "several columns" in low and "one column per month" in low)
        check(f"{name}: an output name must say what the value is", "never name an output" in low)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
