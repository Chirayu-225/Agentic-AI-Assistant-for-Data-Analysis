"""Limiter, cache, task specs."""
import pytest

from core.schemas import OP_NAMES
from llm.cache import LLMCache, make_key
from llm.limiter import TokenBucket
from llm.tasks import OP_CATALOG, TASKS, BatchProposalResponse, FreeTextResponse


class FakeTime:
    def __init__(self): self.t = 0.0; self.slept = []
    def clock(self): return self.t
    def sleep(self, s): self.slept.append(s); self.t += s


def bucket(rate=60, cap=2):
    ft = FakeTime()
    return TokenBucket(rate, cap, clock=ft.clock, sleep=ft.sleep), ft


def test_burst_then_wait():
    b, ft = bucket()
    assert b.try_acquire() and b.try_acquire() and not b.try_acquire()
    assert b.acquire() is True
    assert ft.slept and ft.t == pytest.approx(1.0)


def test_max_wait_refuses_without_taking_token():
    b, ft = bucket(rate=6, cap=1)  # one token every 10 s
    assert b.try_acquire()
    assert b.acquire(max_wait=5) is False
    assert ft.slept == []
    assert b.acquire(max_wait=15) is True


def test_long_run_rate_respected():
    b, ft = bucket(rate=20, cap=1)
    for _ in range(11):
        b.acquire()
    assert ft.t >= 10 * 3.0 - 1e-6  # 20/min = one per 3 s


def test_bad_rate():
    with pytest.raises(ValueError):
        TokenBucket(0)


def test_cache_roundtrip_and_persistence(tmp_path):
    p = str(tmp_path / "c.sqlite")
    c = LLMCache(p)
    c.put("k", "t", '{"a":1}', "groq", "m")
    c.close()
    c2 = LLMCache(p)
    assert c2.get("k") == '{"a":1}' and c2.count() == 1
    c2.clear()
    assert c2.get("k") is None


def test_cache_unwritable_path_is_a_miss_not_a_crash(tmp_path):
    c = LLMCache(str(tmp_path / "no" / "such" / "dir" / "x.sqlite"))
    c.put("k", "t", "{}", "p", "m")
    assert c.get("k") is None or c.get("k") == "{}"


def test_key_properties():
    a = make_key("t", "v1", {"x": 1, "y": [1, 2]})
    assert a == make_key("t", "v1", {"y": [1, 2], "x": 1})
    assert a != make_key("t", "v2", {"x": 1, "y": [1, 2]})
    assert a != make_key("u", "v1", {"x": 1, "y": [1, 2]})
    assert a != make_key("t", "v1", {"x": 2, "y": [1, 2]})


def test_op_catalog_mentions_every_op():
    for name in OP_NAMES:
        assert name in OP_CATALOG


def test_task_specs():
    assert set(TASKS) == {"propose_batch", "free_text_fix"}
    assert TASKS["propose_batch"].slot == "small" and TASKS["free_text_fix"].slot == "large"


def test_free_text_exactly_one():
    FreeTextResponse.model_validate({"refusal": "no"})
    FreeTextResponse.model_validate({"operation": {"op": "drop_duplicates"}})
    with pytest.raises(ValueError):
        FreeTextResponse.model_validate({})
    with pytest.raises(ValueError):
        FreeTextResponse.model_validate({"refusal": "x", "operation": {"op": "drop_duplicates"}})


def test_batch_ignores_extras_and_caps():
    r = BatchProposalResponse.model_validate(
        {"proposals": [{"issue_id": "a", "operation": {"op": "x", "junk": 1}}], "zzz": 1})
    assert r.proposals[0].operation.op == "x"
    with pytest.raises(ValueError):
        BatchProposalResponse.model_validate(
            {"proposals": [{"issue_id": str(i), "operation": {"op": "x"}} for i in range(101)]})
