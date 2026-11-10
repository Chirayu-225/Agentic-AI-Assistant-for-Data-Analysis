"""
test_query_retry.py — the query panel's one repair retry (analysis_service),
with the LLM stubbed: a failing first attempt is shown to the model with its
error and the repaired code runs; a second failure reports both errors; a
success needs no retry. Run: python test_query_retry.py
"""
import sys

import pandas as pd

import query_engine
from analysis_service import run_analysis_from_cleaned_df

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

DF = pd.DataFrame({"team": ["a", "b", "a"], "pts": [3, 4, 5]})
Q = "points scored by team a in the matches where it played"  # not a shortcut

def stub(responses):
    prompts = []
    def fake(prompt, temperature=0.1, session_id=""):
        prompts.append(prompt)
        return responses[min(len(prompts) - 1, len(responses) - 1)]
    query_engine.llm_with_fallback = fake
    return prompts


def main():
    prompts = stub(["result = df.points.sum()", "result = int(df[df['team'] == 'a']['pts'].sum())"])
    r = run_analysis_from_cleaned_df(DF, Q, "t")
    check("repaired code answers correctly", r.get("success") and r.get("resultData") == 8, str(r)[:200])
    check("response says it was retried", r.get("retried") is True)
    check("repair prompt includes the failing code and its error",
          len(prompts) == 2 and "df.points.sum()" in prompts[1] and "Error:" in prompts[1])
    check("the code shown to the user is the code that actually ran", "df['team'] == 'a'" in r.get("code", ""))

    prompts = stub(["result = df.nope.sum()", "result = df.still_nope.sum()"])
    r = run_analysis_from_cleaned_df(DF, Q, "t2")
    check("two failures give a clean error, not a crash", r.get("success") is False and "first attempt" in (r.get("error") or ""), str(r)[:200])

    prompts = stub(["result = int(df['pts'].sum())"])
    r = run_analysis_from_cleaned_df(DF, Q, "t3")
    check("a first-time success makes exactly one LLM call", r.get("success") and len(prompts) == 1 and r.get("retried") is False)

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
        sys.exit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
