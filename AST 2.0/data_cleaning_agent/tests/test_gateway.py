"""Gateway behaviour with scripted fake providers (no network)."""
import json

import pytest
import requests

from llm.cache import LLMCache
from llm.gateway import (Gateway, GatewayConfig, GeminiProvider, GroqProvider, LLMInvalidOutput,
                         LLMUnavailable, OllamaProvider, PayloadRejected, Provider, ProviderError,
                         ProviderReply, call_llm, check_payload)
from llm.limiter import TokenBucket

GOOD = json.dumps({"proposals": [{"issue_id": "i1", "operation": {"op": "drop_duplicates"}}]})
PAYLOAD = {"issues": [{"id": "i1", "type": "duplicates"}]}


class Fake(Provider):
    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.calls = []
    def is_configured(self): return True
    def complete(self, system, user, slot, max_tokens):
        self.calls.append((slot, max_tokens, user))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ProviderReply):
            return item
        return ProviderReply(item, "fake-model")


def gw(providers, **kw):
    sleeps = []
    t = {"now": 0.0}
    g = Gateway(providers, cache=LLMCache(":memory:"),
                limiter=TokenBucket(6000, 100, clock=lambda: t["now"], sleep=lambda s: None),
                sleep=lambda s: (sleeps.append(s), t.__setitem__("now", t["now"] + s)),
                clock=lambda: t["now"], **kw)
    g.sleeps = sleeps
    g.t = t
    return g


def test_happy_path_and_cache_hit_costs_nothing():
    p = Fake("a", [GOOD])
    g = gw([p])
    r = g.call("propose_batch", PAYLOAD)
    assert r.proposals[0].issue_id == "i1"
    assert g.calls_used == 1 and g.provider_requests == 1
    g.call("propose_batch", PAYLOAD)
    assert g.calls_used == 1 and g.cache_hits == 1 and len(p.calls) == 1


def test_reupload_same_dataset_zero_calls_new_gateway_shared_cache():
    cache = LLMCache(":memory:")
    p = Fake("a", [GOOD])
    g1 = Gateway([p], cache=cache, limiter=TokenBucket(6000, 100))
    g1.call("propose_batch", PAYLOAD)
    g2 = Gateway([Fake("a", [])], cache=cache, limiter=TokenBucket(6000, 100))
    g2.call("propose_batch", PAYLOAD)
    assert g2.calls_used == 0 and g2.cache_hits == 1


def test_retry_with_backoff_then_success():
    p = Fake("a", [ProviderError("x", retryable=True, status=500),
                   ProviderError("x", retryable=True, status=429), GOOD])
    g = gw([p])
    g.call("propose_batch", PAYLOAD)
    assert g.sleeps == [1.0, 2.0]
    assert g.calls_used == 1 and g.provider_requests == 3


def test_retry_after_honoured():
    p = Fake("a", [ProviderError("x", retryable=True, retry_after=7.0), GOOD])
    g = gw([p])
    g.call("propose_batch", PAYLOAD)
    assert g.sleeps == [7.0]


def test_exhausted_retries_fall_back_to_next_provider():
    a = Fake("a", [ProviderError("x", retryable=True)] * 3)
    b = Fake("b", [GOOD])
    g = gw([a, b])
    g.call("propose_batch", PAYLOAD)
    assert len(a.calls) == 3 and len(b.calls) == 1
    assert g.calls_used == 1 and g.by_provider == {"b": 1}


def test_permanent_error_skips_provider_for_session():
    a = Fake("a", [ProviderError("bad key", status=401, permanent=True)])
    b = Fake("b", [GOOD, GOOD])
    g = gw([a, b])
    g.call("propose_batch", PAYLOAD)
    g.call("propose_batch", {"issues": [{"id": "other"}]})
    assert len(a.calls) == 1 and len(b.calls) == 2


def test_daily_quota_no_retry_and_cooldown():
    a = Fake("a", [ProviderError("tpd", retryable=False, daily=True)])
    b = Fake("b", [GOOD])
    g = gw([a, b])
    g.call("propose_batch", PAYLOAD)
    assert g.sleeps == [] and len(a.calls) == 1


def test_long_retry_after_switches_provider():
    a = Fake("a", [ProviderError("x", retryable=True, retry_after=300.0)])
    b = Fake("b", [GOOD])
    g = gw([a, b])
    g.call("propose_batch", PAYLOAD)
    assert g.sleeps == [] and len(b.calls) == 1


def test_cooldown_expires():
    a = Fake("a", [ProviderError("x", retryable=True)] * 3 + [GOOD])
    g = gw([a], cooldown_s=60)
    with pytest.raises(LLMUnavailable):
        g.call("propose_batch", PAYLOAD)
    with pytest.raises(LLMUnavailable):  # still cooling down: provider not even tried
        g.call("propose_batch", PAYLOAD)
    assert len(a.calls) == 3
    g.t["now"] += 61
    g.call("propose_batch", PAYLOAD)
    assert len(a.calls) == 4


def test_all_providers_fail_raises_unavailable():
    g = gw([Fake("a", [ProviderError("x", retryable=True)] * 3),
            Fake("b", [ProviderError("y", permanent=True)])])
    with pytest.raises(LLMUnavailable) as e:
        g.call("propose_batch", PAYLOAD)
    assert "x" in str(e.value) and "y" in str(e.value)


def test_no_providers():
    with pytest.raises(LLMUnavailable, match="No AI provider"):
        gw([]).call("propose_batch", PAYLOAD)


def test_invalid_json_retried_once_then_ok():
    p = Fake("a", ["not json at all", GOOD])
    g = gw([p])
    g.call("propose_batch", PAYLOAD)
    assert g.invalid_retries == 1 and g.calls_used == 1 and g.provider_requests == 2
    assert "ONLY the JSON" in p.calls[1][2]


def test_invalid_json_twice_raises_and_does_not_try_other_provider():
    a = Fake("a", ["nope", "still nope"])
    b = Fake("b", [GOOD])
    g = gw([a, b])
    with pytest.raises(LLMInvalidOutput):
        g.call("propose_batch", PAYLOAD)
    assert b.calls == []


def test_truncated_reply_retries_with_double_max_tokens():
    p = Fake("a", [ProviderReply('{"proposals": [', "m", truncated=True), GOOD])
    g = gw([p])
    g.call("propose_batch", PAYLOAD)
    assert p.calls[1][1] == p.calls[0][1] * 2


def test_code_fences_tolerated():
    g = gw([Fake("a", ["```json\n" + GOOD + "\n```"])])
    assert g.call("propose_batch", PAYLOAD).proposals


def test_failed_calls_are_not_cached():
    p = Fake("a", ["bad", "bad", GOOD])
    g = gw([p])
    with pytest.raises(LLMInvalidOutput):
        g.call("propose_batch", PAYLOAD)
    assert g.cache.count() == 0
    g.call("propose_batch", PAYLOAD)
    assert g.cache.count() == 1


def test_stale_cache_entry_recomputed():
    p = Fake("a", [GOOD])
    g = gw([p])
    from llm.cache import make_key
    g.cache.put(make_key("propose_batch", "v1", PAYLOAD), "propose_batch", '{"wrong": 1}', "x", "y")
    g.call("propose_batch", PAYLOAD)
    assert g.calls_used == 1


def test_session_budget():
    g = gw([Fake("a", [GOOD, GOOD, GOOD])], session_budget=2)
    g.call("propose_batch", {"n": 1})
    g.call("propose_batch", {"n": 2})
    with pytest.raises(LLMUnavailable, match="limit"):
        g.call("propose_batch", {"n": 3})
    g.call("propose_batch", {"n": 1})  # cache hits remain free past the budget
    assert g.calls_used == 2


def test_local_rate_limit_gives_unavailable():
    g = Gateway([Fake("a", [GOOD])], cache=LLMCache(":memory:"),
                limiter=TokenBucket(1, 1, clock=lambda: 0.0, sleep=lambda s: None), max_limiter_wait=5)
    g.limiter.try_acquire()
    with pytest.raises(LLMUnavailable, match="rate limit"):
        g.call("propose_batch", PAYLOAD)


def test_unknown_task():
    with pytest.raises(ValueError):
        gw([]).call("nope", {})


def test_call_llm_uses_given_gateway():
    g = gw([Fake("a", [GOOD])])
    assert call_llm("propose_batch", PAYLOAD, gateway=g).proposals
    assert g.stats()["calls_used"] == 1 and g.usage_label == "LLM calls used: 1"


# ---- payload guard ----------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"rows": [1]}, {"a": {"DataFrame": 1}}, {"x": [{"csv": "a,b"}]},
    {"samples": list(range(9))}, {"note": "x" * 501}, {"a": "x" * 400, "b": ["y" * 400] * 60},
])
def test_payload_rejected(bad):
    with pytest.raises(PayloadRejected):
        check_payload(bad)


def test_payload_ok():
    check_payload({"columns": [{"name": "a", "samples": list(range(8))}]})


def test_rejected_payload_never_reaches_provider():
    p = Fake("a", [GOOD])
    with pytest.raises(PayloadRejected):
        gw([p]).call("propose_batch", {"rows": [[1, 2]]})
    assert p.calls == []


# ---- providers: request shape and HTTP mapping --------------------------------
class Resp:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code, self._b, self.text, self.headers = status, body, text, headers or {}
    def json(self):
        if self._b is None:
            raise ValueError
        return self._b


def poster(resp, seen):
    def post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json)
        if isinstance(resp, Exception):
            raise resp
        return resp
    return post


def test_groq_request_shape_and_slots():
    seen = {}
    r = Resp(body={"choices": [{"message": {"content": GOOD}, "finish_reason": "stop"}]})
    p = GroqProvider("KEY", "openai/gpt-oss-20b", "llama-70b", post=poster(r, seen))
    out = p.complete("sys", "usr", "small", 100)
    assert out.text == GOOD and not out.truncated
    assert seen["headers"]["Authorization"] == "Bearer KEY"
    j = seen["json"]
    assert j["model"] == "openai/gpt-oss-20b" and j["temperature"] == 0 and j["max_tokens"] == 100
    assert j["response_format"] == {"type": "json_object"} and j["reasoning_effort"] == "low"
    p.complete("sys", "usr", "large", 100)
    assert seen["json"]["model"] == "llama-70b" and "reasoning_effort" not in seen["json"]


def test_groq_truncation_flag():
    r = Resp(body={"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]})
    assert GroqProvider("k", "m", "m", post=poster(r, {})).complete("s", "u", "small", 5).truncated


def test_gemini_key_in_header_not_url():
    seen = {}
    r = Resp(body={"candidates": [{"content": {"parts": [{"text": GOOD}]}, "finishReason": "STOP"}]})
    out = GeminiProvider("SECRETKEY", "gm", post=poster(r, seen)).complete("s", "u", "small", 50)
    assert out.text == GOOD
    assert "SECRETKEY" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "SECRETKEY"
    assert seen["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_blocked_answer_is_error():
    r = Resp(body={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ProviderError):
        GeminiProvider("k", "m", post=poster(r, {})).complete("s", "u", "small", 5)


def test_ollama_shape():
    seen = {}
    r = Resp(body={"message": {"content": GOOD}, "done_reason": "stop"})
    out = OllamaProvider("llama3", "http://h:1/", post=poster(r, seen)).complete("s", "u", "small", 9)
    assert out.text == GOOD and seen["url"] == "http://h:1/api/chat"
    assert seen["json"]["format"] == "json" and seen["json"]["stream"] is False


@pytest.mark.parametrize("status,kw", [
    (429, dict(retryable=True)), (500, dict(retryable=True)), (503, dict(retryable=True)),
    (401, dict(permanent=True)), (404, dict(permanent=True)), (400, dict(retryable=False)),
])
def test_http_status_mapping(status, kw):
    p = GroqProvider("k", "m", "m", post=poster(Resp(status=status, text="oops"), {}))
    with pytest.raises(ProviderError) as e:
        p.complete("s", "u", "small", 5)
    for k, v in kw.items():
        assert getattr(e.value, k) is v


def test_429_retry_after_and_daily():
    p = GroqProvider("k", "m", "m", post=poster(Resp(429, text="x", headers={"retry-after": "4"}), {}))
    with pytest.raises(ProviderError) as e:
        p.complete("s", "u", "small", 5)
    assert e.value.retry_after == 4.0 and not e.value.daily
    p = GroqProvider("k", "m", "m", post=poster(Resp(429, text="Limit reached: tokens per day (TPD)"), {}))
    with pytest.raises(ProviderError) as e:
        p.complete("s", "u", "small", 5)
    assert e.value.daily and not e.value.retryable


@pytest.mark.parametrize("exc", [requests.Timeout(), requests.ConnectionError(), requests.RequestException()])
def test_transport_errors_retryable(exc):
    p = GroqProvider("k", "m", "m", post=poster(exc, {}))
    with pytest.raises(ProviderError) as e:
        p.complete("s", "u", "small", 5)
    assert e.value.retryable


def test_non_json_200_and_odd_shapes_are_errors():
    for body in (None, {}, {"choices": []}):
        p = GroqProvider("k", "m", "m", post=poster(Resp(200, body), {}))
        with pytest.raises(ProviderError):
            p.complete("s", "u", "small", 5)


# ---- config ---------------------------------------------------------------------
def test_config_from_env_and_provider_order():
    cfg = GatewayConfig.from_env({"GROQ_API_KEY": "g", "GEMINI_API_KEY": "m", "LLM_PROVIDER_ORDER": "gemini, groq",
                                  "LLM_MAX_RPM": "10", "LLM_SESSION_BUDGET": "5", "GEMINI_MODEL": "gx"})
    assert [p.name for p in cfg.build_providers()] == ["gemini", "groq"]
    assert cfg.max_rpm == 10 and cfg.session_budget == 5 and cfg.gemini_model == "gx"


def test_config_empty_env_has_no_providers():
    cfg = GatewayConfig.from_env({})
    assert cfg.build_providers() == []
    assert Gateway.from_config(GatewayConfig(cache_path=":memory:")).providers == []


def test_ollama_only_when_model_set():
    assert [p.name for p in GatewayConfig.from_env({"OLLAMA_MODEL": "q"}).build_providers()] == ["ollama"]
