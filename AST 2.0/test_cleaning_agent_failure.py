"""
test_cleaning_agent_failure.py — what the USER sees when the Data Quality
Agent fails. The response must carry a short, fixed, safe message: no Python
exception text, no traceback, no file paths. The technical detail goes to the
server log only, and the diagnosis stays visible. Also checks that the fix
prompt carries the "iterating a DataFrame gives column names" hint.

The LLM is stubbed everywhere (and the real provider call is made to fail
loudly, so a stray Groq call cannot happen). Run: python test_cleaning_agent_failure.py
"""
import contextlib
import io
import sys
from types import SimpleNamespace

import pandas as pd

import llm_provider
import query_engine

FAIL = []

SAFE_FIX = "The AI's suggested fix could not be applied. Nothing in your data was changed."

DIAG = ("COLUMN: country\nTYPE: inconsistent_categories\n"
        "EVIDENCE: 'USA' and 'U.S.' both appear\nACTION: map to one spelling\n")

# Things that must never reach the user.
FORBIDDEN = ["Traceback", "astype", "Error", "Exception", ".py", "C:\\", "C:/", "boom", "secret",
             "attribute", "line "]


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def csv_bytes():
    buf = io.BytesIO()
    pd.DataFrame({"country": ["USA", "U.S.", "Canada", "USA", "Canada"],
                  "orders": [1, 2, 3, 4, 5]}).to_csv(buf, index=False)
    return buf.getvalue()


def stub_llm(fix_code):
    """Diagnosis -> DIAG; fix prompt -> a reasoning block plus fix_code."""
    def fake(prompt, temperature=0.1, primary="", session_id="", max_tokens=2048):
        if "Defects to fix" in prompt:
            return "REASONING:\n- country: normalise spellings\n```python\n" + fix_code + "\n```"
        return DIAG
    query_engine.llm_with_fallback = fake


def run_agent():
    """Run the agent, return (response, captured server log)."""
    import cleaning_agent
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = cleaning_agent.run_cleaning_agent(csv_bytes(), "orders.csv", session_id="t")
    return r, buf.getvalue()


def leaks(text):
    return [f for f in FORBIDDEN if f in (text or "")]


def main():
    def no_groq(*a, **k):
        raise AssertionError("a real provider call was attempted in a test")
    llm_provider._call_groq = no_groq

    import cleaning_agent
    import sandbox_limits
    real_run_sandboxed = sandbox_limits.run_sandboxed
    real_read_csv = cleaning_agent._read_csv
    real_llm = query_engine.llm_with_fallback

    try:
        print("=" * 70)
        print("PART 1 — fix code that fails in the REAL sandbox (the 'str has no astype' idiom)")
        print("=" * 70)
        stub_llm("cleaned_df = df.copy()\n"
                 "for c in cleaned_df.select_dtypes(include='object'):\n"
                 "    cleaned_df[c] = c.astype(str)")
        r, log = run_agent()
        check("the run fails", r.get("success") is False, str(r)[:200])
        check("the user sees exactly the short safe message", r.get("error") == SAFE_FIX, repr(r.get("error")))
        check("no exception text, traceback or path in the user-facing error", not leaks(r.get("error")), str(leaks(r.get("error"))))
        check("the diagnosis stays visible", bool(r.get("defects")) and r["defects"][0].get("column") == "country", str(r.get("defects")))
        check("the fix reasoning stays visible", bool(r.get("reasoning")), str(r.get("reasoning")))
        check("the technical detail IS in the server log", "astype" in log and "[cleaning-agent]" in log, log[-300:])
        check("the failing fix code is in the server log", "--- fix code was:" in log and "select_dtypes" in log, log[-300:])

        print("\n" + "=" * 70)
        print("PART 2 — fix code that raises with a path in its message (real sandbox)")
        print("=" * 70)
        stub_llm("raise ValueError('boom reading C:\\\\Users\\\\secret\\\\x.py')")
        r, log = run_agent()
        check("the user sees exactly the short safe message", r.get("error") == SAFE_FIX, repr(r.get("error")))
        check("no exception text or path in the user-facing error", not leaks(r.get("error")), str(leaks(r.get("error"))))
        check("the diagnosis stays visible", bool(r.get("defects")))

        print("\n" + "=" * 70)
        print("PART 3 — sandbox returns a raw traceback with a file path (stubbed result)")
        print("=" * 70)
        stub_llm("cleaned_df = df.copy()")
        sandbox_limits.run_sandboxed = lambda *a, **k: SimpleNamespace(
            success=False, produced=False, value=None,
            error=('Traceback (most recent call last):\n  File "C:\\Users\\secret\\sandbox_limits.py", line 9, in _worker\n'
                   "AttributeError: 'str' object has no attribute 'astype'"))
        r, log = run_agent()
        sandbox_limits.run_sandboxed = real_run_sandboxed
        check("the user sees exactly the short safe message", r.get("error") == SAFE_FIX, repr(r.get("error")))
        check("no traceback or path in the user-facing error", not leaks(r.get("error")), str(leaks(r.get("error"))))
        check("the traceback and path are in the server log", "Traceback" in log and "sandbox_limits.py" in log, log[-300:])
        check("the whole response contains no traceback or path", not leaks(str({k: v for k, v in r.items() if k != "ruleLog"})),
              str(leaks(str(r))))

        print("\n" + "=" * 70)
        print("PART 4 — an unexpected exception inside the agent (the outer except)")
        print("=" * 70)

        def exploding_llm(*a, **k):
            raise RuntimeError("boom: key file C:\\Users\\secret\\key.py unreadable")
        query_engine.llm_with_fallback = exploding_llm
        r, log = run_agent()
        check("the run fails", r.get("success") is False)
        check("the user sees a short safe message", r.get("error") == cleaning_agent._SAFE_AGENT_ERROR
              and "Nothing in your data was changed" in r.get("error", ""), repr(r.get("error")))
        check("no exception text or path in the user-facing error", not leaks(r.get("error")), str(leaks(r.get("error"))))
        check("the traceback is in the server log", "RuntimeError" in log and "key.py" in log, log[-300:])

        print("\n" + "=" * 70)
        print("PART 5 — the file cannot be read (the CSV parse error)")
        print("=" * 70)

        def unreadable(*a, **k):
            raise OSError("cannot open C:\\Users\\secret\\orders.csv: permission denied")
        cleaning_agent._read_csv = unreadable
        r, log = run_agent()
        cleaning_agent._read_csv = real_read_csv
        check("the run fails", r.get("success") is False)
        check("the user sees a short safe message", r.get("error") == cleaning_agent._SAFE_READ_ERROR, repr(r.get("error")))
        check("no exception text, 'parse' wording or path in the user-facing error",
              not leaks(r.get("error")) and "parse" not in r.get("error", "").lower(), str(leaks(r.get("error"))))
        check("the detail is in the server log", "secret" in log and "orders.csv" in log, log[-300:])

        print("\n" + "=" * 70)
        print("PART 6 — the fix prompt carries the column-names hint")
        print("=" * 70)
        from profiler import build_defect_fix_prompt
        prompt = build_defect_fix_prompt([{"type": "outlier", "column": "x", "evidence": "e", "action": "a"}], "x, y")
        check("prompt says iterating a DataFrame gives column names", "for c in df.columns" in prompt and "column NAMES" in prompt)
        check("prompt says to use df[c]", "use df[c]" in prompt)
        check("prompt says how to select text columns", "select_dtypes(include=['object','string'])" in prompt)
        check("prompt still asks for the cleaned_df format", "cleaned_df = df.copy()" in prompt and "REASONING:" in prompt)
    finally:
        sandbox_limits.run_sandboxed = real_run_sandboxed
        cleaning_agent._read_csv = real_read_csv
        query_engine.llm_with_fallback = real_llm

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
        sys.exit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
