"""test_chart_retry.py — a chart whose first code is blocked gets one repair retry.
Run: python3 test_chart_retry.py"""
import sys, types
import pandas as pd
# llm_provider imports streamlit/groq at module load; the model call is replaced
# below, so lightweight stand-ins are enough when they aren't installed.
for _name in ("streamlit", "groq"):
    try:
        __import__(_name)
    except ImportError:
        _m = types.ModuleType(_name)
        _m.cache_data = lambda *a, **k: (lambda f: f)
        _m.Groq = object
        sys.modules[_name] = _m
import query_engine
import auto_analyze_service as svc

FAIL = []
def check(l, c, d=""):
    print(f"[{'OK' if c else 'FAIL'}] {l}" + (f" — {d}" if d and not c else ""))
    if not c: FAIL.append(l)

def main():
    df = pd.DataFrame({"region": ["N", "S", "N"], "units": [1, 2, 3]})
    plan = {"title": "t", "supports": "s", "spec": "bar of units by region"}
    bad = "result = px.bar(df.query('units > 0'), x='region', y='units')"
    good = "g = df.groupby('region')['units'].sum().reset_index()\nresult = px.bar(g, x='region', y='units')"
    calls = []
    def fake(prompt, temperature=0.1, session_id=""):
        calls.append(prompt)
        return bad if len(calls) == 1 else good
    query_engine.llm_with_fallback = fake
    r = svc._run_one_chart(plan, "region, units", df, "s1")
    check("first attempt blocked, second succeeds", r["resultType"] == "chart", str(r))
    check("exactly two model calls", len(calls) == 2, str(len(calls)))
    check("repair prompt contains the error", len(calls) > 1 and "blocked" in calls[1].lower())

    calls.clear()
    query_engine.llm_with_fallback = lambda p, temperature=0.1, session_id="": (calls.append(p) or bad)
    r = svc._run_one_chart(plan, "region, units", df, "s2")
    check("gives up after one retry (no infinite loop)", r["resultType"] == "error" and len(calls) == 2)
    check("error is surfaced, not swallowed", bool(r["error"]))

    calls.clear()
    query_engine.llm_with_fallback = lambda p, temperature=0.1, session_id="": (calls.append(p) or good)
    r = svc._run_one_chart(plan, "region, units", df, "s3")
    check("good code first time = one call, no retry", r["resultType"] == "chart" and len(calls) == 1)

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
    print("DONE — all checks passed.")

if __name__ == "__main__":
    main()
