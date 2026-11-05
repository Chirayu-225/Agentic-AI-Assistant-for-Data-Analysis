"""
test_llm_ratelimit.py — provider rate-limit (HTTP 429) handling in llm_provider,
and how the eval harness treats a provider outage. No network, no API key:
the provider call is stubbed. Run: python test_llm_ratelimit.py

Covers: a short 429 is waited out and retried; a daily-quota 429 fails fast with
a user-safe message (no organisation id) and does not burn the daily call
budget; other errors pass through untouched; the fallback provider still gets
its turn; and eval accuracy excludes provider outages instead of scoring them
as wrong answers.
"""
import os
import sys
import tempfile

# Point the usage counter at a throwaway file BEFORE importing, so this test
# never touches the real logs/llm_usage.json.
os.environ["LLM_USAGE_PATH"] = os.path.join(tempfile.mkdtemp(), "llm_usage.json")
os.environ.pop("FALLBACK_LLM_BASE_URL", None)
os.environ.pop("FALLBACK_LLM_API_KEY", None)
os.environ.pop("FALLBACK_CODE_MODEL", None)

import llm_provider as lp

_REAL_CALL_GROQ = lp._call_groq  # later tests replace lp._call_groq with stubs
from eval_harness import QuestionResult, classify_failure, summarize

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

SHORT_429 = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `m` in organization "
             "`org_SECRET123` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7900, "
             "Requested 600. Please try again in 2s. Need more tokens?'}}")
DAILY_429 = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `m` in organization "
             "`org_SECRET123` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199325, "
             "Requested 1200. Please try again in 14m22.5s. Need more tokens?'}}")

def reset():
    lp._usage.update(day=lp._today(), calls=0)
    sleeps = []
    lp.time.sleep = lambda s: sleeps.append(s)
    return sleeps

# ── wait-time parsing ────────────────────────────────────────────────────────
class Resp:  # minimal stand-in for an HTTP response carrying a Retry-After header
    def __init__(self, h): self.headers = h
class WithHeader(Exception):
    response = Resp({"retry-after": "7"})

check("parses 1m23.4s", abs(lp._retry_after_seconds(Exception("Please try again in 1m23.4s.")) - 83.4) < 1e-6)
check("parses 850ms", abs(lp._retry_after_seconds(Exception("Please try again in 850ms.")) - 0.85) < 1e-6)
check("parses 2h3m4s", lp._retry_after_seconds(Exception("Please try again in 2h3m4s.")) == 7384)
check("no wait hint -> None", lp._retry_after_seconds(Exception("Error code: 429")) is None)
check("Retry-After header wins", lp._retry_after_seconds(WithHeader("try again in 99s")) == 7.0)
check("detects 429 by text", lp._is_rate_limit(Exception(SHORT_429)))
check("an ordinary error is not a rate limit", not lp._is_rate_limit(RuntimeError("boom")))

# ── short 429: wait, retry, succeed ──────────────────────────────────────────
sleeps = reset()
calls = []
def flaky(prompt, temperature, model, max_tokens):
    calls.append(1)
    if len(calls) == 1:
        raise Exception(SHORT_429)
    return "ok"
lp._call_groq = flaky
out = lp._call("p", 0.1, "m")
check("short 429 is retried and succeeds", out == "ok" and len(calls) == 2, f"out={out!r} calls={len(calls)}")
check("waited the provider's 2s (+ margin)", sleeps == [2.5], str(sleeps))
check("a retried call counts once against the daily budget", lp._usage["calls"] == 1, str(lp._usage))

# ── daily quota: fail fast, safe message, budget refunded ────────────────────
sleeps = reset()
lp._call_groq = lambda *a, **k: (_ for _ in ()).throw(Exception(DAILY_429))
try:
    lp._call("p", 0.1, "m")
    raised = None
except Exception as e:
    raised = e
check("daily quota raises ProviderRateLimited", isinstance(raised, lp.ProviderRateLimited), repr(raised))
check("flagged as a daily limit", getattr(raised, "daily", None) is True)
check("message never leaks the organisation id", raised is not None and "org_" not in str(raised) and "SECRET" not in str(raised), str(raised))
check("no pointless sleeping on a daily quota", sleeps == [], str(sleeps))
check("rejected call does not burn the daily budget", lp._usage["calls"] == 0, str(lp._usage))

# ── retries are bounded ──────────────────────────────────────────────────────
sleeps = reset()
n = []
def always_short(*a, **k):
    n.append(1)
    raise Exception(SHORT_429)
lp._call_groq = always_short
try:
    lp._call("p", 0.1, "m")
    raised = None
except Exception as e:
    raised = e
check("gives up after the retry cap", isinstance(raised, lp.ProviderRateLimited) and len(n) == 1 + lp._MAX_RATE_RETRIES, f"attempts={len(n)}")

# ── other errors are untouched ───────────────────────────────────────────────
reset()
lp._call_groq = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    lp._call("p", 0.1, "m")
    raised = None
except Exception as e:
    raised = e
check("non-429 errors propagate unchanged", type(raised) is RuntimeError and str(raised) == "boom", repr(raised))
check("a failed non-429 call still counts", lp._usage["calls"] == 1)

# ── fallback provider still gets its turn after a rate limit ─────────────────
reset()
os.environ.update(FALLBACK_LLM_BASE_URL="http://fallback.invalid/v1", FALLBACK_LLM_API_KEY="k", FALLBACK_CODE_MODEL="fm")
lp._call_groq = lambda *a, **k: (_ for _ in ()).throw(Exception(DAILY_429))
lp._call_fallback = lambda prompt, temperature, slot, max_tokens: "from-fallback"
out = lp._call("p", 0.1, lp._CODE_MODEL)
check("fallback answers when the primary is rate-limited", out == "from-fallback", repr(out))
for k in ("FALLBACK_LLM_BASE_URL", "FALLBACK_LLM_API_KEY", "FALLBACK_CODE_MODEL"):
    os.environ.pop(k, None)

# ── the user-facing wrapper keeps the friendly text ──────────────────────────
reset()
lp._call_groq = lambda *a, **k: (_ for _ in ()).throw(Exception(DAILY_429))
lp._clear_code_cache()
try:
    lp.llm_code_call("a unique prompt", session_id="s")
    raised = None
except Exception as e:
    raised = e
check("llm_code_call surfaces a readable message", raised is not None and "daily usage quota" in str(raised) and "org_" not in str(raised), str(raised))

# ── eval harness: provider outage is not a wrong answer ──────────────────────
friendly = "Analysis pipeline failed: LLM code call failed: " + lp._rate_limit_message(True, 862.0)
check("friendly quota message -> infra_error", classify_failure(friendly) == "infra_error", classify_failure(friendly))
check("raw provider 429 text -> infra_error", classify_failure("Analysis pipeline failed: LLM code call failed: " + SHORT_429) == "infra_error")
check("app's own daily cap -> infra_error", classify_failure("The daily AI usage limit for this demo has been reached.") == "infra_error")
check("a real pipeline bug is still pipeline_error", classify_failure("Analysis pipeline failed: could not parse model output") == "pipeline_error")
check("security blocks are unaffected", classify_failure("Security block: import is not allowed") == "security_block")

def qr(i, outcome, correct=False):
    return QuestionResult(id=f"q{i}", run=1, question="?", category="c", answer_type="scalar", expected="1",
                          actual="1" if correct else "", correct=correct, executed=correct, outcome=outcome,
                          detail="", latency_s=1.0, signature="1" if correct else "")
results = [qr(1, "correct", True), qr(2, "correct", True), qr(3, "wrong_answer")] + [qr(i, "infra_error") for i in range(4, 7)]
s = summarize(results, 1)
check("accuracy ignores provider outages", s["accuracy"] == round(2 / 3, 3), str(s["accuracy"]))
check("scored runs exclude outages", s["runs"] == 3, str(s["runs"]))
check("outages are reported, not hidden", s["infraErrors"] == 3 and s["outcomes"].get("infra_error") == 3, str(s["outcomes"]))
s_all = summarize([qr(i, "infra_error") for i in range(3)], 1)
check("an all-outage run does not crash and scores nothing", s_all["runs"] == 0 and s_all["infraErrors"] == 3)

# ── token accounting (the quota is in tokens; the call cap counts calls) ──────
class _Obj:
    def __init__(self, **k): self.__dict__.update(k)

class _FakeClient:
    def __init__(self, usage): self._usage = usage
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kw):
        return _Obj(choices=[_Obj(message=_Obj(content="answer"), finish_reason="stop")], usage=self._usage)

lp._client = _FakeClient(_Obj(prompt_tokens=700, completion_tokens=120))
before = lp.token_stats()
out = _REAL_CALL_GROQ("p", 0.1, "some-model", 100)
after = lp.token_stats()
check("the provider's reported tokens are accumulated per call",
      out == "answer" and after["calls"] - before["calls"] == 1 and after["prompt"] - before["prompt"] == 700 and after["completion"] - before["completion"] == 120, after)
lp._client = _FakeClient(None)
b2 = lp.token_stats()
_REAL_CALL_GROQ("p", 0.1, "some-model", 100)
a2 = lp.token_stats()
check("a response without usage is counted as unreported, not as zero tokens", a2["unreported"] - b2["unreported"] == 1 and a2["calls"] == b2["calls"], a2)
lp._client = None

print()
if FAIL:
    print(f"{len(FAIL)} check(s) failed: {FAIL}")
    sys.exit(1)
print("all checks passed")
