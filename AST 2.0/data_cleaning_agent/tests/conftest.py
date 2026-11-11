"""Block real network access in every test: the suite must never call an LLM API."""
import pytest
import requests


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test tried to use the real network")
    monkeypatch.setattr(requests.sessions.Session, "request", boom)
