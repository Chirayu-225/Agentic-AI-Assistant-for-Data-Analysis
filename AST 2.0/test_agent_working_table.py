"""
test_agent_working_table.py — the agent's verified table becomes the session's working table.

Before this, run_cleaning_agent only RETURNED the table (as a preview); the next question read the original file again,
so the banner "used for your questions" was not true. Now the API route passes a hook that saves the verified table in
the session cache, which is where /analyze looks first. The hook runs only when the verifier kept at least one change,
and a failure to store is reported as "not kept", never as "used".

The model and the sandbox are stubbed; no Groq. Run: python test_agent_working_table.py
"""
import contextlib
import io
import os
from types import SimpleNamespace

import pandas as pd

FAIL = []
URL = "http://test/files/sales.csv"

DIAG = ("COLUMN: state\nTYPE: inconsistent_categories\nEVIDENCE: some states look odd\nACTION: fix them\n")
STATES = ["CA", "NY", "TX", "FL", "WA", "OR", "NV", "AZ", "CO", "UT", "MA", "NJ", "PA", "OH", "MI"]
BAD_TEXT = ("Traceback", ".py", "Error:", "Exception", "boom")


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def sales_csv() -> bytes:
    n = 120
    df = pd.DataFrame({
        "order": list(range(1, n + 1)),
        "state": [STATES[i % 15] for i in range(n)],
        "amount": [round(10 + i * 1.25, 2) for i in range(n)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def main():
    import cache_layer
    import cleaning_agent
    import file_reader
    import query_engine
    import sandbox_limits
    import session_cache
    from cleaning import rule_based_clean

    os.environ.pop("CACHE_ENABLED", None)
    data = sales_csv()
    cache_layer.clear_all()
    rule_cleaned, _ = cache_layer.cached_rule_clean(file_reader.read_tabular_file(data, "sales.csv"), data, rule_based_clean)

    def run(transform, hook):
        cache_layer.clear_all()
        session_cache.clear()

        def fake_model(prompt, temperature=0.1, primary="", session_id="", max_tokens=2048):
            if "Defects to fix" in prompt:
                return "REASONING:\n- state: tidy\n```python\ncleaned_df = df.copy()\n```"
            return DIAG
        query_engine.llm_with_fallback = fake_model

        def fake_sandbox(code, df, include_charts=False, timeout_seconds=20, **k):
            return SimpleNamespace(success=True, produced=True, value=transform(df.copy()), error=None)
        sandbox_limits.run_sandboxed = fake_sandbox
        with contextlib.redirect_stdout(io.StringIO()):
            return cleaning_agent.run_cleaning_agent(data, "sales.csv", session_id="s1", on_applied=hook)

    safe = lambda df: df.assign(amount_is_outlier=(df["amount"] > 100))                    # a new boolean flag column
    harmful = lambda df: df.assign(state=None)                                             # blanks every state
    mixed = lambda df: df.assign(amount_is_outlier=(df["amount"] > 100), state=None)       # one good change, one bad

    # 1. a safe fix is handed to the hook, once, and the response says it is used
    got = []
    r = run(safe, got.append)
    check("safe fix: the hook is called exactly once", len(got) == 1, str(len(got)))
    check("safe fix: the table it receives has the flag column and the original values",
          len(got) == 1 and "amount_is_outlier" in got[0].columns and got[0]["state"].tolist() == rule_cleaned["state"].tolist())
    check("safe fix: the response says the changes are used for questions", r.get("usedForQuestions") is True and r.get("applied") is True)
    check("safe fix: the banner text is true", "used for your questions" in r["message"])
    check("safe fix: the banner says how long the table is kept (the session cache expires)",
          f"kept for the next {session_cache.TTL_SECONDS // 60} minutes" in r["message"], r["message"])

    # 1b. the table is stored only AFTER the response was built: if building it fails, nothing is stored
    got = []
    real_records = cleaning_agent._records
    def broken_records(df):
        raise RuntimeError("boom while building the preview")
    cleaning_agent._records = broken_records
    try:
        r = run(safe, got.append)
    finally:
        cleaning_agent._records = real_records
    check("ordering: a failure while building the response is a safe error", r["success"] is False and "boom" not in str(r))
    check("ordering: ... and the table was NOT stored", got == [])

    # 2. a fix the verifier fully reverts keeps nothing: the hook is never called
    got = []
    r = run(harmful, got.append)
    check("harmful fix: nothing applied", r["applied"] is False, str(r["message"]))
    check("harmful fix: the hook is NOT called", got == [])
    check("harmful fix: the response does not claim the changes are used", "usedForQuestions" not in r)

    # 3. a mixed fix: the hook gets the REPAIRED table (good change kept, bad column reverted)
    got = []
    r = run(mixed, got.append)
    check("mixed fix: the hook gets the repaired table: flag kept, states untouched",
          len(got) == 1 and "amount_is_outlier" in got[0].columns and got[0]["state"].notna().all()
          and got[0]["state"].tolist() == rule_cleaned["state"].tolist())

    # 4. storing fails: the user is told the changes were NOT kept, and no exception text leaks
    def boom(table):
        raise RuntimeError("boom at C:/secret/store.py")
    r = run(safe, boom)
    check("store failure: the request still succeeds", r["success"] is True)
    check("store failure: usedForQuestions is False", r.get("usedForQuestions") is False)
    check("store failure: the message says not kept, never used", "not kept for your questions" in r["message"] and "used for your questions" not in r["message"])
    check("store failure: no exception text in the response", not any(b in str(r["message"]) for b in BAD_TEXT) and "boom" not in str(r))

    # 5. no hook (library use): nothing is stored and the response has no new field
    r = run(safe, None)
    check("no hook: no usedForQuestions field", "usedForQuestions" not in r)

    # 6. the real thing: the hook the API route uses, then read it back the way /analyze does
    r = run(safe, lambda t: session_cache.put("s1", URL, "sales.csv", data, t))
    hit = session_cache.get("s1", URL)
    check("session cache: the next question finds the agent's table", hit is not None and "amount_is_outlier" in hit.cleaned_df.columns)
    check("session cache: its original columns equal the rule-cleaned table",
          hit is not None and hit.cleaned_df[list(rule_cleaned.columns)].equals(rule_cleaned))
    check("session cache: another session does not see it", session_cache.get("other", URL) is None)
    r = run(harmful, lambda t: session_cache.put("s1", URL, "sales.csv", data, t))
    check("session cache: a fully reverted fix stores nothing", session_cache.get("s1", URL) is None)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
