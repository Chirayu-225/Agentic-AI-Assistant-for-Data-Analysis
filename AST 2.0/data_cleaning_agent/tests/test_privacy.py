"""Personal data must not survive into a payload (both the AST backend and the fallback)."""
import importlib.util
import json
import os
import sys

import pytest

from llm.privacy import WITHHELD, Sanitizer
from llm.gateway import Gateway, Provider, ProviderReply
from llm.cache import LLMCache
from llm.limiter import TokenBucket

BACKENDS = ["fallback"] + (["ast"] if importlib.util.find_spec("pii_redaction") or
                           os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                               os.path.abspath(__file__)))), "pii_redaction.py")) else [])


def make(backend):
    if backend == "ast":
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(parent)
    return Sanitizer(backend)


PAYLOAD = {
    "issues": [
        {"id": "i1", "type": "missing", "column": "email", "samples": ["a@b.com", "c@d.org"],
         "evidence": {"min": "a@b.com"}},
        {"id": "i2", "type": "duplicates", "column": None, "samples": ["Ravi | ravi@x.com | 9876543210"]},
        {"id": "i3", "type": "outliers", "column": "income", "samples": ["50000", "999999"]},
    ],
    "columns": [{"name": "phone", "sample_values": ["9876543210"], "top_values": [["9876543210", 3]]},
                {"name": "city", "sample_values": ["Pune", "Mumbai"]}],
    "user_message": "email me at someone@example.com please",
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_personal_values_removed(backend):
    s = make(backend)
    out = s.clean(PAYLOAD)
    blob = json.dumps(out)
    for secret in ("a@b.com", "c@d.org", "ravi@x.com", "9876543210", "someone@example.com", "Ravi"):
        assert secret not in blob, secret
    assert out["issues"][0]["samples"] == [WITHHELD]
    assert out["issues"][1]["samples"] == []  # no single column: dropped entirely
    assert out["issues"][2]["samples"] == ["50000", "999999"]  # harmless columns untouched
    assert out["columns"][1]["sample_values"] == ["Pune", "Mumbai"]
    assert {"email", "phone"} <= s.columns_withheld
    assert out["issues"][0]["column"] == "email"  # structure preserved


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_not_modified(backend):
    before = json.dumps(PAYLOAD, sort_keys=True)
    make(backend).clean(PAYLOAD)
    assert json.dumps(PAYLOAD, sort_keys=True) == before


@pytest.mark.parametrize("backend", BACKENDS)
def test_switch_off_keeps_column_values_but_still_redacts_patterns(backend, monkeypatch):
    monkeypatch.setenv("PII_WITHHOLD_PERSONAL_COLUMNS", "0")
    out = make(backend).clean({"issues": [{"column": "email", "samples": ["plain text"]}]})
    assert out["issues"][0]["samples"] == ["plain text"]


class Spy(Provider):
    name = "spy"
    def __init__(self): self.seen = []
    def is_configured(self): return True
    def complete(self, system, user, slot, max_tokens):
        self.seen.append(user)
        return ProviderReply('{"proposals": []}', "m")


def test_gateway_sanitizes_before_provider_sees_anything():
    spy = Spy()
    g = Gateway([spy], cache=LLMCache(":memory:"), limiter=TokenBucket(6000, 100))
    g.call("propose_batch", PAYLOAD)
    assert "a@b.com" not in spy.seen[0] and "9876543210" not in spy.seen[0]
    assert "someone@example.com" not in spy.seen[0]
    st = g.stats()
    assert "email" in st["personal_columns_withheld"] and st["privacy_backend"] in ("ast", "fallback")
