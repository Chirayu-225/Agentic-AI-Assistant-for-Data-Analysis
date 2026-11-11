"""
test_agent_fill.py — the agent FILLS blanks first (in code, from evidence in the table), then lets the model audit the rest.

What it fills: a blank another column proves (every row of one customer has the same postal code) and, for numbers with
no evidence, the column median. What it cannot fill becomes a QUESTION for the user (fill with "Unknown" / leave empty):
the response lists it, and the user's per-column answer goes through fill_service.run_fill(unknown_columns=[...]).
The fill happens in EVERY path: fix applied, fix failed, no defects found. The model and sandbox are stubbed; no Groq.

Run: python test_agent_fill.py
"""
import contextlib
import io
import os
from types import SimpleNamespace

import pandas as pd

FAIL = []
URL = "http://test/files/data.csv"
DIAG = "COLUMN: note\nTYPE: missing_values\nEVIDENCE: some notes are blank\nACTION: look at them\n"


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def make_csv(blanks=True) -> bytes:
    n = 120
    cust = [f"C{i:02d}" for i in range(30)]
    df = pd.DataFrame({
        "order": range(1, n + 1),
        "customer": [cust[i % 30] for i in range(n)],
        "postal": [f"P{(i % 30):03d}" for i in range(n)],
        "amount": [round(10 + i * 1.5, 2) for i in range(n)],
        "note": [["red", "blue", "green", "pink", "grey", "gold", "teal"][i % 7] for i in range(n)],
        "ordered": ["2024-01-%02d" % (1 + i % 28) for i in range(n)],
    })
    if blanks:
        df.loc[[5, 40, 77], "postal"] = None                    # provable from the same customer's other rows
        df.loc[df["customer"] == "C11", "postal"] = None         # C11: no known row, so no evidence (4 rows)
        df.loc[[3, 9, 22, 31, 50], "amount"] = None              # no relationship: the median
        df.loc[[2, 8, 14, 26], "note"] = None                    # no relationship: no evidence (4 blanks)
        df.loc[[7, 19], "ordered"] = None                        # a date: never guessed
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def main():
    import cache_layer
    import cleaning_agent
    import fill_service
    import llm_provider
    import query_engine
    import sandbox_limits
    import session_cache

    os.environ.pop("CACHE_ENABLED", None)
    data = make_csv()

    def run(hook, diag=DIAG, fix="cleaned_df = df.copy()", sandbox="identity", blanks=True):
        cache_layer.clear_all()
        session_cache.clear()

        def fake_model(prompt, temperature=0.1, primary="", session_id="", max_tokens=2048):
            if "Defects to fix" in prompt:
                return "REASONING:\n- note: look at it\n```python\n" + fix + "\n```"
            return diag
        query_engine.llm_with_fallback = fake_model
        llm_provider.llm_uncached_code_call = lambda prompt, **k: "no code here"

        def fake_sandbox(code, df, include_charts=False, timeout_seconds=20, **k):
            if sandbox == "fail":
                return SimpleNamespace(success=False, produced=False, value=None, error="NameError: boom")
            out = df.copy()
            if sandbox == "flag":
                out["amount_is_big"] = out["amount"] > 100
            return SimpleNamespace(success=True, produced=True, value=out, error=None)
        sandbox_limits.run_sandboxed = fake_sandbox
        with contextlib.redirect_stdout(io.StringIO()):
            return cleaning_agent.run_cleaning_agent(make_csv(blanks), "data.csv", session_id="s1", on_applied=hook)

    # 1. the fill block and the questions
    tables = []
    r = run(tables.append, sandbox="flag")
    fill = r.get("fill", {})
    meth = {(f["column"], f["method"]): f["count"] for f in fill.get("filled", [])}
    check("fill: a proven postal blank is filled from the same customer's other rows", meth.get(("postal", "inferred")) == 3, str(fill.get("filled")))
    check("fill: numbers with no evidence are NOT guessed (no median)", ("amount", "median") not in meth and ("amount", "text") not in meth)
    q = {s["column"]: s["count"] for s in fill.get("skipped", []) if s.get("kind") == "no_evidence"}
    check("questions: the text blanks with no evidence are listed per column, with counts", q == {"postal": 4, "amount": 5, "note": 4}, str(q))
    check("questions: remainingText is their total", fill.get("remainingText") == 8)
    check("fill: dates are listed as never guessed, not as a question", any(s["column"] == "ordered" and s.get("kind") != "no_evidence" for s in fill.get("skipped", [])))
    check("fill: totalFilled counts only what was filled", fill.get("totalFilled") == 3 and fill.get("kept") is True)
    check("message: says what was filled and that it is used for your questions", r["message"].startswith("Filled 3 missing values (3 from other rows that prove the value), used for your questions"), r["message"][:140])
    check("the AI's own safe change is still applied on top", r.get("applied") is True and "amount_is_big" in r.get("newColumns", []))

    # 2. what is stored for the next question: the filled table, then filled + the AI's safe change
    final = tables[-1]
    check("stored: the last table kept has the fills AND the AI's flag column", "amount_is_big" in final.columns and int(final["amount"].isna().sum()) == 5 and int(final["postal"].isna().sum()) == 4)
    check("stored: the filled table was kept first, before the model ran", len(tables) == 2 and "amount_is_big" not in tables[0].columns and int(tables[0]["amount"].isna().sum()) == 5)
    check("null card: counts the original columns only", r["after"]["nullValues"] < r["before"]["nullValues"] and r["after"]["nullValues"] == int(final.iloc[:, :6].isna().sum().sum()))

    # 3. the fill does not depend on the model: it happens when the AI's fix fails, and when no defect is found
    t2 = []
    r = run(t2.append, sandbox="fail")
    check("AI fix failed: still reports the fill and the questions", r["success"] is False and r["fill"]["totalFilled"] == 3 and r["fill"]["remainingText"] == 8)
    check("AI fix failed: the filled table was still kept for questions", len(t2) == 1 and int(t2[0]["amount"].isna().sum()) == 5)
    t3 = []
    r = run(t3.append, diag="")
    check("no defects found: still filled, and the banner says so", r["success"] is True and r["agentUsed"] is False and r["fill"]["totalFilled"] == 3 and r["message"].startswith("Filled 3"), str(r.get("message")))
    check("no defects found: the filled table was kept", len(t3) == 1)

    # 4. hook variants
    r = run(None)
    check("no hook (library use): kept is None and the message does not claim anything is used", r["fill"]["kept"] is None and "used for your questions" not in r["message"].split("Filled")[1].split(".")[0])

    def boom(table):
        raise RuntimeError("boom at C:/secret/store.py")
    r = run(boom)
    check("hook fails: kept is False and the message says not kept, no exception text", r["fill"]["kept"] is False and "not kept for your questions" in r["message"] and "boom" not in str(r) and "secret" not in str(r))

    # 5. a table with nothing to fill: no fill noise at all
    r = run(lambda t: None, blanks=False)
    check("nothing to fill: totalFilled 0 and the message is unchanged", r["fill"]["totalFilled"] == 0 and not r["message"].startswith("Filled"))

    # 6. the user's per-column answer
    cache_layer.clear_all(); session_cache.clear()
    run(lambda t: session_cache.put("s1", URL, "data.csv", data, t))
    rr = fill_service.run_fill(data, "data.csv", "s1", URL, unknown_columns=["note"])
    left = session_cache.get("s1", URL).cleaned_df
    check("answer 'Unknown' for note: only that column is labelled", int(left["note"].isna().sum()) == 0 and (left["note"] == "Unknown").sum() == 4 and int(left["postal"].isna().sum()) == 4, str(rr.get("message")))
    check("the other question is still open", rr["remainingText"] == 4 and any(s["column"] == "postal" and s.get("kind") == "no_evidence" for s in rr["skipped"]))
    rr = fill_service.run_fill(data, "data.csv", "s1", URL, unknown_columns=["postal"])
    left = session_cache.get("s1", URL).cleaned_df
    check("answer 'Unknown' for postal too: all text questions are closed, the date stays blank", rr["remainingText"] == 0 and int(left["postal"].isna().sum()) == 0 and int(left["ordered"].isna().sum()) == 2)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
