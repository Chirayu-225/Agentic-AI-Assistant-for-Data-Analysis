"""
test_agent_validators.py — the agent runs the deterministic value repairs (value_validators.py) after the fill step.

Checked: the repairs are reported (block + banner sentence) and become the working table; a blank a validator makes is not
filled back in; a clean table gets no validators block and no store; a crashing validator cannot break the agent.
The model and the sandbox are stubbed; no Groq.   Run: python test_agent_validators.py
"""
import contextlib
import io
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

FAIL = []
DIAG = "COLUMN: region\nTYPE: inconsistent_categories\nEVIDENCE: odd\nACTION: look\n"


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def orders(n=60, broken=True):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(1, n + 1)],
        "customer": [f"Customer {i}" for i in range(n)],
        "region": [["East", "West", "North", "South"][i % 4] for i in range(n)],
        "quantity": rng.integers(1, 20, n),
        "unit_price": np.round(rng.uniform(5, 500, n), 2),
        "order_date": (pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D")).strftime("%Y-%m-%d"),
    })
    if broken:
        df.loc[[4, 11], "quantity"] = [-14, -3]                       # impossible
        df.loc[10, ["customer", "region"]] = ["South", "Priya Menon"]  # swapped
        df.loc[20, "unit_price"] = np.nan                              # an ORIGINAL blank: the fill step may fill this one
        df.loc[[3], "order_date"] = ["2098-06-19"]                     # future
    return df


def csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def main():
    import cache_layer
    import cleaning_agent
    import query_engine
    import sandbox_limits
    import session_cache

    os.environ.pop("CACHE_ENABLED", None)

    def run(df, hook=None):
        cache_layer.clear_all()
        session_cache.clear()

        def fake_model(prompt, temperature=0.1, primary="", session_id="", max_tokens=2048):
            if "Defects to fix" in prompt:
                return "REASONING:\n- region: nothing to change\n```python\ncleaned_df = df.copy()\n```"
            return DIAG
        query_engine.llm_with_fallback = fake_model
        sandbox_limits.run_sandboxed = lambda code, d, include_charts=False, timeout_seconds=20, **k: \
            SimpleNamespace(success=True, produced=True, value=d.copy(), error=None)
        with contextlib.redirect_stdout(io.StringIO()):
            return cleaning_agent.run_cleaning_agent(csv_bytes(df), "orders.csv", session_id="t", on_applied=hook)

    # 1. a broken table: the repairs run, are reported, and become the working table
    got = []
    r = run(orders(), got.append)
    check("broken table: the agent succeeds", r["success"] is True, str(r.get("error")))
    rules = sorted(e["rule"] for e in r.get("validators", {}).get("entries", []))
    check("broken table: impossible negatives, a swap and a future date are all repaired",
          {"impossible_negative", "swapped_values", "future_date"} <= set(rules), str(rules))
    check("broken table: the block carries counts and examples",
          all(e["count"] >= 1 and e["examples"] for e in r["validators"]["entries"]))
    check("broken table: the banner leads with what was repaired", r["message"].startswith("Repaired "), r["message"][:90])
    check("broken table: ... and says the table is used for questions, with how long",
          "used for your questions for the next 30 minutes" in r["message"], r["message"])
    check("broken table: the working table was stored exactly once", len(got) == 1, str(len(got)))
    t = got[0]
    check("working table: the two negative quantities are blank (not flipped, not filled)", t["quantity"].isna().sum() == 2 and (t["quantity"].dropna() > 0).all(),
          str(t["quantity"].isna().sum()))
    check("working table: the swapped pair is back in place", t["customer"][10] == "Priya Menon" and t["region"][10] == "South")
    check("working table: the future date is blank", pd.isna(t["order_date"][3]))
    check("working table: the validators' blanks were NOT filled back in (the fill step ran first)",
          t["quantity"].isna().sum() == 2 and pd.isna(t["order_date"][3]))
    check("response: the stats are over the repaired table", r["after"]["rows"] == 60)

    # 2. a clean table: nothing to report, nothing stored
    got = []
    r = run(orders(broken=False), got.append)
    check("clean table: no validators block", "validators" not in r)
    check("clean table: nothing stored for later questions", got == [])
    check("clean table: the banner does not mention repairs", not str(r.get("message", "")).startswith("Repaired"))

    # 3. the validators crash: the agent carries on without them
    import value_validators
    real = value_validators.validate_values
    value_validators.validate_values = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom C:/secret/x.py"))
    try:
        got = []
        r = run(orders(), got.append)
    finally:
        value_validators.validate_values = real
    check("validator crash: the agent still succeeds", r["success"] is True)
    check("validator crash: no validators block and nothing leaks", "validators" not in r and "boom" not in str(r) and "secret" not in str(r))

    # 4. storing fails: the banner says not kept, never used
    def boom(table):
        raise RuntimeError("store down")
    r = run(orders(), boom)
    check("store failure: the repairs are reported but the table is said NOT to be kept",
          r["success"] is True and "but not kept for your questions" in r["message"] and "used for your questions" not in r["message"], r.get("message", "")[:140])

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
