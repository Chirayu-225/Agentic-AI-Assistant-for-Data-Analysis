"""
Resilience of the model layer: a second provider, a circuit breaker and a
per-request deadline. (Hazard H5: a failing or slow provider must not take
the service down or hold requests open.)

Run from the "AST 2.0" folder:
    python -m tests.llm.test_resilience

No network and no API key: both providers are stubs and time is a fake clock.
"""
from __future__ import annotations

import os
import tempfile

os.environ["LLM_USAGE_PATH"] = os.path.join(tempfile.mkdtemp(), "llm_usage.json")
for _k in ("FALLBACK_LLM_BASE_URL", "FALLBACK_LLM_API_KEY", "FALLBACK_CODE_MODEL", "FALLBACK_CHAT_MODEL",
           "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_CODE_MODEL", "GEMINI_CHAT_MODEL", "GEMINI_BASE_URL"):
    os.environ.pop(_k, None)
os.environ["CACHE_ENABLED"] = "0"
os.environ["LLM_DAILY_CALL_LIMIT"] = "0"

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def main():
    from llm import llm_provider as lp
    from llm import resilience as rs

    clock = {"t": 1000.0}
    rs.set_clock(lambda: clock["t"])
    lp.time.sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    calls = {"primary": 0, "fallback": 0}
    mode = {"primary": "ok", "fallback": "ok"}
    seen = {}

    def primary(prompt, temperature, model, max_tokens):
        calls["primary"] += 1
        seen["primary_timeout"] = rs.call_timeout(90)
        if mode["primary"] == "down":
            raise ConnectionError("connection refused")
        if mode["primary"] == "slow":
            clock["t"] += 50
        return "from groq"

    def fallback(prompt, temperature, slot, max_tokens):
        calls["fallback"] += 1
        if mode["fallback"] == "down":
            raise ConnectionError("gemini unreachable")
        return "from gemini"
    lp._call_groq, lp._call_fallback = primary, fallback

    def reset(primary_mode="ok", fallback_mode="ok"):
        rs.reset_breakers()
        calls.update(primary=0, fallback=0)
        mode.update(primary=primary_mode, fallback=fallback_mode)

    print("=" * 70 + "\nPART 1 - Gemini as the second provider\n" + "=" * 70)
    check("with no second provider configured, there is no fallback", not lp.fallback_configured())
    os.environ["GEMINI_API_KEY"] = "test-key"
    check("a Gemini key alone configures the fallback", lp.fallback_configured())
    base, key, model = lp._fallback_config("code")
    check("it uses Google's OpenAI-compatible endpoint",
          base == "https://generativelanguage.googleapis.com/v1beta/openai" and key == "test-key", base)
    check("the model has a default and can be overridden", bool(model)
          and (os.environ.__setitem__("GEMINI_MODEL", "my-model") or lp._fallback_config("chat")[2] == "my-model"))
    os.environ.pop("GEMINI_MODEL")
    os.environ.update(FALLBACK_LLM_BASE_URL="https://example.test/v1", FALLBACK_LLM_API_KEY="k", FALLBACK_CODE_MODEL="m")
    check("an explicit FALLBACK_* setting still wins over the Gemini preset", lp._fallback_config("code")[0] == "https://example.test/v1")
    for k in ("FALLBACK_LLM_BASE_URL", "FALLBACK_LLM_API_KEY", "FALLBACK_CODE_MODEL"):
        os.environ.pop(k)

    print("\n" + "=" * 70 + "\nPART 2 - failover\n" + "=" * 70)
    reset()
    with rs.request_deadline(120) as req:
        out = lp.llm_uncached_code_call("p")
    check("a healthy primary answers and the fallback is not touched",
          out == "from groq" and calls == {"primary": 1, "fallback": 0} and not req.fallback_used)
    reset(primary_mode="down")
    with rs.request_deadline(120) as req:
        out = lp.llm_uncached_code_call("p")
    check("when the primary fails, the same prompt is answered by the fallback", out == "from gemini" and calls["fallback"] == 1)
    check("... and the request knows a fallback answered", req.fallback_used is True)
    reset(primary_mode="down", fallback_mode="down")
    try:
        lp.llm_uncached_code_call("p")
        check("when both providers fail, the call fails", False)
    except RuntimeError as e:
        check("when both providers fail, the call fails with both reasons",
              "primary failed" in str(e) and "fallback failed" in str(e), str(e)[:160])

    print("\n" + "=" * 70 + "\nPART 3 - circuit breaker\n" + "=" * 70)
    reset(primary_mode="down")
    for _ in range(rs.FAILURE_THRESHOLD):
        lp.llm_uncached_code_call("p")
    check(f"after {rs.FAILURE_THRESHOLD} failures in a row the primary's circuit is open", rs.breaker("primary").state() == "open")
    before = calls["primary"]
    for _ in range(5):
        lp.llm_uncached_code_call("p")
    check("while it is open the primary is NOT called (no time wasted on a dead provider)", calls["primary"] == before)
    check("... and requests are still answered by the fallback", calls["fallback"] == rs.FAILURE_THRESHOLD + 5)
    clock["t"] += rs.COOLDOWN_SECONDS + 1
    mode["primary"] = "ok"
    out = lp.llm_uncached_code_call("p")
    check("after the cooldown one trial call is let through, and success closes the circuit",
          out == "from groq" and rs.breaker("primary").state() == "closed")

    reset(primary_mode="down")
    for _ in range(rs.FAILURE_THRESHOLD):
        lp.llm_uncached_code_call("p")
    clock["t"] += rs.COOLDOWN_SECONDS + 1
    before = calls["primary"]
    lp.llm_uncached_code_call("p")
    check("a failed trial call re-opens the circuit", calls["primary"] == before + 1 and rs.breaker("primary").state() == "open")
    lp.llm_uncached_code_call("p")
    check("... and the next request does not try the primary again", calls["primary"] == before + 1)

    reset(primary_mode="down")
    lp.llm_uncached_code_call("p")
    mode["primary"] = "ok"
    lp.llm_uncached_code_call("p")
    mode["primary"] = "down"
    for _ in range(rs.FAILURE_THRESHOLD - 1):
        lp.llm_uncached_code_call("p")
    check("failures that are not consecutive do not open the circuit", rs.breaker("primary").state() == "closed")

    os.environ.pop("GEMINI_API_KEY")
    reset(primary_mode="down")
    for _ in range(rs.FAILURE_THRESHOLD):
        try:
            lp.llm_uncached_code_call("p")
        except Exception:
            pass
    before = calls["primary"]
    try:
        lp.llm_uncached_code_call("p")
        check("with no fallback, an open circuit fails fast", False)
    except rs.ProviderUnavailable as e:
        check("with no fallback, an open circuit fails fast with a readable message",
              calls["primary"] == before and "temporarily unavailable" in str(e), str(e))
    os.environ["GEMINI_API_KEY"] = "test-key"

    b = rs.CircuitBreaker("unit")
    b.record_failure(cooldown=600)
    check("a daily-quota failure opens the circuit at once, for its own longer cooldown",
          b.state() == "open" and (clock.__setitem__("t", clock["t"] + 300) or b.state() == "open")
          and (clock.__setitem__("t", clock["t"] + 301) or b.state() == "half_open"))

    print("\n" + "=" * 70 + "\nPART 4 - one deadline per request\n" + "=" * 70)
    reset()
    with rs.request_deadline(120):
        lp.llm_uncached_code_call("p")
    check("a model call is given no more time than the request has left", seen["primary_timeout"] == 90)
    with rs.request_deadline(30):
        clock["t"] += 10
        lp.llm_uncached_code_call("p")
        check("... so its timeout shrinks as the request ages", abs(seen["primary_timeout"] - 20) < 1e-6, str(seen["primary_timeout"]))
    reset(primary_mode="slow")
    before = calls["primary"]
    try:
        with rs.request_deadline(120):
            for _ in range(10):
                lp.llm_uncached_code_call("p")
        check("a request that keeps calling a slow model is stopped", False)
    except rs.DeadlineExceeded as e:
        made = calls["primary"] - before
        check("a request that keeps calling a slow model is stopped at its deadline, not after 10 calls",
              made == 3 and "took too long" in str(e), f"{made} calls; {e}")
    with rs.request_deadline(120):
        with rs.request_deadline(600) as inner:
            check("a nested scope cannot extend the deadline it is inside", inner.remaining() <= 120)
    reset()
    check("outside any request scope a call has the default timeout and no deadline",
          lp.llm_uncached_code_call("p") == "from groq" and seen["primary_timeout"] == 90)
    reset(primary_mode="down")
    try:
        with rs.request_deadline(120):
            clock["t"] += 200
            lp.llm_uncached_code_call("p")
        check("an expired request does not start a model call", False)
    except rs.DeadlineExceeded:
        check("an expired request does not start a model call at all", calls == {"primary": 0, "fallback": 0}, str(calls))
    check("... and a deadline is not counted as a provider failure", rs.breaker("primary").state() == "closed")

    print("\n" + "=" * 70 + "\nPART 5 - the pipeline reports it\n" + "=" * 70)
    import pandas as pd
    from analysis import analysis_service as svc
    os.environ.pop("AST_QUERY_PLAN", None)
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("AST_DATASET_DB", None)
    plan = '{"select":[{"agg":"sum","column":"amount","as":"total"}],"group_by":["region"]}'
    lp._call_groq = lambda *a: (_ for _ in ()).throw(ConnectionError("down"))
    lp._call_fallback = lambda *a: plan
    reset()
    df = pd.DataFrame({"region": ["n", "s", "n"], "amount": [1.0, 2.0, 3.0]})
    out = svc.run_analysis_from_cleaned_df(df, "Give me the sum of amount for each region, largest first", "sess-r1")
    check("an answer produced by the second provider says so",
          out.get("success") is True and out.get("llmFallbackUsed") is True, str({k: out.get(k) for k in ("success", "error", "llmFallbackUsed")}))
    lp._call_groq = lambda *a: plan
    reset()
    out = svc.run_analysis_from_cleaned_df(df, "Give me the sum of amount for each region, largest first", "sess-r2")
    check("an answer from the primary does not carry that flag", out.get("success") is True and "llmFallbackUsed" not in out)

    def slow(*a):
        clock["t"] += 500
        return "not json"
    lp._call_groq = slow
    reset()
    out = svc.run_analysis_from_cleaned_df(df, "Give me the sum of amount for each region, largest first", "sess-r3")
    check("a request that runs out of time ends with a clear message, not a hang",
          out.get("success") is False and "took too long" in str(out.get("error")), str(out.get("error"))[:200])

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
