"""
test_cleaning_agent_verifier.py — run_cleaning_agent with the verifier in the loop (the model and, mostly, the sandbox
are stubbed; no Groq).

The agent asks a model for ONE fix script. Whatever the script does, the table that comes out is agent_verify's
"repaired" table: blanked / filled / renamed / changed values are reverted, structural damage discards the fix, safe
tidying and new flag columns stay. The response keeps every existing field and adds "applied", "reverted",
"keptChanges" and "message". The fix prompt carries the hard rules.

Run: python test_cleaning_agent_verifier.py
"""
import contextlib
import io
import os
import sys
from types import SimpleNamespace

import pandas as pd

FAIL = []

DIAG = ("COLUMN: state\nTYPE: inconsistent_categories\nEVIDENCE: some states look odd\nACTION: fix them\n"
        "COLUMN: territory\nTYPE: missing_values\nEVIDENCE: territory has blanks\nACTION: fill them\n")
STATES = ["CA", "NY", "TX", "FL", "WA", "OR", "NV", "AZ", "CO", "UT", "MA", "NJ", "PA", "OH", "MI"]
LINES = ["Planes", "Ships", "Trains", "Cars", "Trucks"]
EXISTING_KEYS = {"success", "filename", "ruleLog", "defects", "agentUsed", "reasoning", "before", "after", "preview", "previewCount"}
NEW_KEYS = {"applied", "reverted", "keptChanges", "message"}


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def sales_csv() -> bytes:
    n = 120
    df = pd.DataFrame({
        "order": list(range(1, n + 1)),
        "state": [STATES[i % 15] for i in range(n)],
        "territory": [["NA", "EMEA", "APAC", ""][i % 4] for i in range(n)],       # NA is a real value; "" is missing
        "productline": [LINES[i % 5] for i in range(n)],
        "amount": [round(10 + i * 1.25, 2) for i in range(n)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def main():
    import agent_verify
    import cache_layer
    import cleaning_agent
    import file_reader
    import query_engine
    import sandbox_limits
    from cleaning import rule_based_clean
    from profiler import build_defect_fix_prompt

    os.environ.pop("CACHE_ENABLED", None)
    data = sales_csv()
    cache_layer.clear_all()
    rule_cleaned, _ = cache_layer.cached_rule_clean(file_reader.read_tabular_file(data, "sales.csv"), data, rule_based_clean)
    base_nulls = int(rule_cleaned.isnull().sum().sum())
    check("setup: NA is kept as a real territory value, only the blanks are missing",
          int((rule_cleaned["territory"] == "NA").sum()) == 30 and int(rule_cleaned["territory"].isna().sum()) == 30)

    fix_reasoning = "REASONING:\n- state: tidy them\n- territory: proposal only: a person should decide what the blanks mean\n"
    seen_code = []

    def stub_model(fix_code="cleaned_df = df.copy()"):
        def fake(prompt, temperature=0.1, primary="", session_id="", max_tokens=2048):
            if "Defects to fix" in prompt:
                return fix_reasoning + "```python\n" + fix_code + "\n```"
            return DIAG
        query_engine.llm_with_fallback = fake

    def stub_sandbox(transform):
        """The sandbox 'runs' the fix by applying `transform` to the table it is given."""
        def fake(code, df, include_charts=False, timeout_seconds=20, **k):
            seen_code.append(code)
            return SimpleNamespace(success=True, produced=True, value=transform(df.copy()), error=None)
        sandbox_limits.run_sandboxed = fake

    def run(transform=None, fix_code="cleaned_df = df.copy()"):
        cache_layer.clear_all()
        stub_model(fix_code)
        if transform is not None:
            stub_sandbox(transform)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = cleaning_agent.run_cleaning_agent(data, "sales.csv", session_id="t")
        return r, buf.getvalue()

    BAD = ("Traceback", ".py", "C:\\", "Error:", "Exception")

    def clean_text(r):
        text = " ".join([r.get("message", "")] + list(r.get("reverted", [])) + [str(k) for k in r.get("keptChanges", [])])
        return not any(b in text for b in BAD)

    # ── 1. a fix that blanks real state values ─────────────────────────────────────────────────────────────────
    def blank_state(df):
        df.loc[df["state"].isin(["CA", "NY"]), "state"] = None
        return df
    r, _ = run(blank_state)
    check("blank a state: the response is a success with every old key and the new ones",
          r.get("success") is True and EXISTING_KEYS <= set(r) and NEW_KEYS <= set(r), str(sorted(r)))
    check("blank a state: nothing applied; the column is reverted and named in plain English",
          r["applied"] is False and r["keptChanges"] == [] and r["reverted"] == ["state: 16 existing values would have been blanked"], str(r["reverted"]))
    check("blank a state: the message says no data was changed", r["message"].startswith("No data was changed."), r["message"])
    check("blank a state: the null stats reflect what was applied (no new nulls)", r["after"]["nullValues"] == base_nulls, f"{r['after']} vs {base_nulls}")
    check("blank a state: the preview still shows the real states", {row["state"] for row in r["preview"]} <= set(STATES) and all(row["state"] for row in r["preview"]))

    # ── 2. filling the blanks with "Unknown" ───────────────────────────────────────────────────────────────────
    r, _ = run(lambda df: df.assign(territory=df["territory"].fillna("Unknown")))
    check("fill territory with 'Unknown': reverted as filled, the blanks stay blank",
          r["reverted"] == ["territory: 30 missing values would have been filled in"] and r["after"]["nullValues"] == base_nulls, str(r["reverted"]))

    # ── 3. renaming a category ─────────────────────────────────────────────────────────────────────────────────
    r, _ = run(lambda df: df.assign(productline=df["productline"].replace({"Planes": "Aircraft"})))
    check("rename Planes to Aircraft: reverted", r["reverted"] == ["productline: 24 existing values would have been changed"]
          and all(row["productline"] != "Aircraft" for row in r["preview"]), str(r["reverted"]))

    # ── 4. structural damage discards the whole fix ────────────────────────────────────────────────────────────
    r, _ = run(lambda df: df.drop(index=3))
    check("drop a row: the fix is discarded, the table keeps all its rows", r["applied"] is False and r["after"]["rows"] == len(rule_cleaned)
          and r["message"].startswith("No data was changed."), str(r.get("message")))
    r, _ = run(lambda df: df.sort_values("amount", ascending=False).reset_index(drop=True))
    check("reorder rows: discarded", r["applied"] is False and r["preview"][0]["order"] == 1, str(r["message"]))
    r, _ = run(lambda df: df.rename(columns={"state": "region"}))
    check("rename a column: discarded", r["applied"] is False and "region" not in r["preview"][0] and "state" in r["preview"][0])

    # ── 5. good fixes are applied ──────────────────────────────────────────────────────────────────────────────
    def good(df):
        df["productline"] = df["productline"].str.upper()
        df["amount_is_outlier"] = df["amount"] > 100
        return df
    r, _ = run(good)
    check("safe fix: applied, with the exact message", r["applied"] is True and r["message"] == "Applied 2 safe changes, used for your questions", r["message"])
    check("safe fix: the new flag column is in the result", "amount_is_outlier" in r["preview"][0] and r["after"]["columns"] == r["before"]["columns"] + 1,
          str(r["after"]))
    check("safe fix: keptChanges names both", {k["column"] for k in r["keptChanges"]} == {"productline", "amount_is_outlier"}, str(r["keptChanges"]))
    check("safe fix: nothing was reverted", r["reverted"] == [])

    # ── 6. mixed fix ───────────────────────────────────────────────────────────────────────────────────────────
    def mixed(df):
        df.loc[df["state"] == "TX", "state"] = None
        df["amount_is_outlier"] = df["amount"] > 100
        return df
    r, _ = run(mixed)
    check("mixed fix: the good change stays, the bad column is reverted, and the message says both",
          r["applied"] is True and r["message"] == "Applied 1 safe change; reverted 1 column that would have changed your values"
          and r["reverted"] == ["state: 8 existing values would have been blanked"], f"{r['message']} {r['reverted']}")

    # ── 7. proposal-only lines are display text; the code is the only thing executed ───────────────────────────
    seen_code.clear()
    r, _ = run(good, fix_code="cleaned_df = df.copy()\ncleaned_df['x'] = 1")
    check("'proposal only' lines are shown as reasoning and never executed",
          any("proposal only" in line for line in r["reasoning"]) and seen_code and all("proposal only" not in c for c in seen_code), str(seen_code))

    # ── 8. a verifier that crashes: the fix is discarded, no exception text reaches the user ────────────────────
    real_verify = agent_verify.verify_fix
    agent_verify.verify_fix = lambda b, a: (_ for _ in ()).throw(RuntimeError("boom C:\\secret\\x.py"))
    try:
        r, log = run(good)
    finally:
        agent_verify.verify_fix = real_verify
    check("a verifier that raises discards the fix and says so", r.get("success") is True and r["applied"] is False
          and r["message"].startswith("No data was changed."), str(r.get("message")))
    check("... without exception text or paths in any response field", "boom" not in str(r) and "secret" not in str(r) and clean_text(r))
    check("... and the detail went to the server log", "boom" in log)

    # ── 9. the verifier's own fault (tripwire) discards the fix ────────────────────────────────────────────────
    real_cmp = agent_verify._compare_column
    agent_verify._compare_column = lambda b, a, n: {"filled": 0, "blanked": 0, "changed": 0, "tidied": 0, "examples": {}}
    try:
        r, _ = run(blank_state)
    finally:
        agent_verify._compare_column = real_cmp
    check("tripwire through the agent: discarded, nulls did not rise", r["applied"] is False and r["after"]["nullValues"] == base_nulls
          and r["message"].startswith("No data was changed."), str(r.get("message")))

    # ── 10. the response text is clean in every case above ─────────────────────────────────────────────────────
    r, _ = run(blank_state)
    check("response text has no traceback, path or exception name", clean_text(r))

    # ── 11. one run through the REAL sandbox ───────────────────────────────────────────────────────────────────
    import importlib
    importlib.reload(sandbox_limits)
    r, _ = run(None, fix_code="cleaned_df = df.copy()\ncleaned_df['productline'] = cleaned_df['productline'].str.upper()")
    check("real sandbox: a safe fix runs and is applied", r.get("success") is True and r.get("applied") is True
          and r["preview"][0]["productline"].isupper(), str(r)[:300])
    r, _ = run(None, fix_code="cleaned_df = df.copy()\ncleaned_df['state'] = cleaned_df['state'].where(cleaned_df['state'] != 'CA')")
    check("real sandbox: a fix that blanks states is reverted", r.get("success") is True and r.get("applied") is False
          and r["reverted"] and r["reverted"][0].startswith("state: 8 existing values would have been blanked"), str(r)[:300])

    # ── 12. the fix prompt carries the hard rules ──────────────────────────────────────────────────────────────
    prompt = build_defect_fix_prompt([{"type": "inconsistent_categories", "column": "state", "evidence": "e", "action": "a"}], "state, territory")
    for rule in ("NEVER replace an existing value with NaN", "NEVER fill missing values", "NEVER rename or merge categories",
                 "NEVER drop rows or columns", "ONLY add new boolean flag columns", "proposal only:"):
        check(f"fix prompt contains the rule: {rule}", rule in prompt)
    check("fix prompt no longer encourages imputing or merging", "Not Applicable" not in prompt and "canonical" not in prompt.lower()
          and "map every variant" not in prompt)
    check("fix prompt keeps the format the agent parses and the column-names hint",
          "Defects to fix" in prompt and "REASONING:" in prompt and "cleaned_df" in prompt and "gives column NAMES" in prompt)
    check("fix prompt names the defects it was given", "'state'" in prompt and "inconsistent_categories" in prompt)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
