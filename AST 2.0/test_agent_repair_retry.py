"""
test_agent_repair_retry.py — when the sandbox rejects the agent's fix script, the agent makes ONE repair attempt.

Seen live: the model wrote `import re`, the sandbox blocks every import ("Security block: import 're' blocked"), and the
whole fix was thrown away although every flag column in it was safe. Now the model is shown its own (scrubbed) error
once. If the repaired code runs, it goes through the verifier like any other; if it fails again the user gets the same
safe message as before. Never more than one repair. The model and the sandbox are stubbed; no Groq.

Run: python test_agent_repair_retry.py
"""
import contextlib
import io
import os
from types import SimpleNamespace

import pandas as pd

FAIL = []
DIAG = "COLUMN: state\nTYPE: inconsistent_categories\nEVIDENCE: odd states\nACTION: flag them\n"
STATES = ["CA", "NY", "TX", "FL", "WA", "OR", "NV", "AZ", "CO", "UT", "MA", "NJ", "PA", "OH", "MI"]


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def sales_csv() -> bytes:
    n = 120
    df = pd.DataFrame({"order": list(range(1, n + 1)), "state": [STATES[i % 15] for i in range(n)],
                       "amount": [round(10 + i * 1.25, 2) for i in range(n)]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def main():
    import cache_layer
    import cleaning_agent
    import llm_provider
    import query_engine
    import sandbox_limits

    os.environ.pop("CACHE_ENABLED", None)
    data = sales_csv()
    BAD_CODE = "import re\ncleaned_df = df.copy()\ncleaned_df['amount_is_big'] = df['amount'] > 100"
    GOOD_CODE = "cleaned_df = df.copy()\ncleaned_df['amount_is_big'] = df['amount'] > 100"
    ERR = "Security block: import 're' blocked in C:/Users/x/secret/sandbox.py"
    state = {"fix_prompts": [], "repair_prompts": [], "sandbox_codes": []}

    def run(repair_code, repair_runs=True):
        cache_layer.clear_all()
        state["fix_prompts"].clear(); state["repair_prompts"].clear(); state["sandbox_codes"].clear()

        def fake_model(prompt, temperature=0.1, primary="", session_id="", max_tokens=2048):
            if "Defects to fix" in prompt:
                state["fix_prompts"].append(prompt)
                return "REASONING:\n- state: flag the odd ones\n```python\n" + BAD_CODE + "\n```"
            return DIAG
        query_engine.llm_with_fallback = fake_model

        def fake_repair(prompt, temperature=0.1, max_tokens=4096, **k):
            state["repair_prompts"].append(prompt)
            if repair_code is None:
                return "I cannot do that."
            return "REASONING:\n- state: flag the odd ones\n```python\n" + repair_code + "\n```"
        llm_provider.llm_uncached_code_call = fake_repair

        def fake_sandbox(code, df, include_charts=False, timeout_seconds=20, **k):
            state["sandbox_codes"].append(code)
            if "import " in code:
                return SimpleNamespace(success=False, produced=False, value=None, error=ERR)
            if not repair_runs:
                return SimpleNamespace(success=False, produced=False, value=None, error="NameError: x")
            out = df.copy()
            if "['state'] = None" in code:                       # the "harmful repair" case really blanks every state
                out["state"] = None
            else:
                out["amount_is_big"] = out["amount"] > 100
            return SimpleNamespace(success=True, produced=True, value=out, error=None)
        sandbox_limits.run_sandboxed = fake_sandbox
        with contextlib.redirect_stdout(io.StringIO()):
            return cleaning_agent.run_cleaning_agent(data, "sales.csv", session_id="t")

    # 1. the blocked import is repaired on the second attempt
    r = run(GOOD_CODE)
    check("repair: the request succeeds", r["success"] is True, str(r.get("error")))
    check("repair: the flag column the repaired code adds is kept", r.get("applied") is True and "amount_is_big" in r.get("newColumns", []))
    check("repair: exactly one repair call, two sandbox runs", len(state["repair_prompts"]) == 1 and len(state["sandbox_codes"]) == 2)
    p = state["repair_prompts"][0]
    check("repair prompt: carries the sandbox error", "Security block: import 're' blocked" in p)
    check("repair prompt: the path in the error is scrubbed", "secret" not in p and "C:/Users" not in p)
    check("repair prompt: says imports are blocked and never to write one", "ALL import statements are blocked" in p and "never write the word import" in p)
    check("repair prompt: restates the hard rules", "never blank, fill, rename or drop" in p)
    check("repair prompt: starts from the normal fix prompt", p.startswith(state["fix_prompts"][0]))

    # 2. the repaired code is still verified like any other: a harmful repair is reverted
    r = run("cleaned_df = df.copy()\ncleaned_df['state'] = None")
    check("harmful repair: the verifier still reverts it", r["success"] is True and r["applied"] is False and any("state" in s for s in r["reverted"]))

    # 3. the repair also fails: the same safe message, and no third attempt
    r = run(GOOD_CODE, repair_runs=False)
    check("repair fails too: success False with the safe fix error", r["success"] is False and "could not be applied" in r["error"])
    check("repair fails too: still exactly one repair call", len(state["repair_prompts"]) == 1)
    check("repair fails too: no exception text or path in the response", "NameError" not in str(r) and "secret" not in str(r))

    # 4. the model returns no usable code for the repair: the safe failure, no crash
    r = run(None)
    check("no usable repair code: the safe failure", r["success"] is False and "could not be applied" in r["error"])

    # 5. the repair call itself raises: the safe failure, no crash, nothing leaks
    def raising(prompt, **k):
        raise RuntimeError("boom at C:/secret/provider.py")
    cache_layer.clear_all()
    llm_provider.llm_uncached_code_call = raising
    with contextlib.redirect_stdout(io.StringIO()):
        r = cleaning_agent.run_cleaning_agent(data, "sales.csv", session_id="t")
    check("repair call raises: the safe failure and nothing leaks", r["success"] is False and "boom" not in str(r) and "secret" not in str(r))

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
