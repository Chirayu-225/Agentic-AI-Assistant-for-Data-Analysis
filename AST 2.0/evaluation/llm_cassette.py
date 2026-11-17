"""
Record and replay model responses, so evaluations run free and repeatably.

WHY
    The model is the one part of the pipeline that is slow, rate-limited and
    non-deterministic. Recording its responses once ("record" mode, real calls)
    and replaying them afterwards ("replay" mode, no network) means the whole
    pipeline around the model - prompts, plan validation, SQL, verification,
    sandbox - can be re-tested on every change at zero cost, and any change in
    the result is caused by OUR code, not by the model's mood.

RULES
    * Replay never calls the model. A prompt that is not in the cassette is a
      MISS: it raises, it is counted, and the gate fails with "re-record".
      A changed prompt is a release-relevant change, so this is deliberate.
    * Prompts are matched after removing the random per-request fence tag
      (security/prompt_injection_guard.py), which differs on every call.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from typing import Callable, Optional

_TAG = re.compile(r"BEGIN UNTRUSTED DATA ([0-9A-Za-z]{4,64})")


class CassetteMiss(RuntimeError):
    """Replay was asked for a prompt that was never recorded."""


def normalize_prompt(prompt: str) -> str:
    out = prompt
    for tag in set(_TAG.findall(prompt)):
        out = out.replace(tag, "<TAG>")
    return out.replace("\r\n", "\n")


def _key(prompt: str, temperature: float) -> str:
    h = hashlib.sha256()
    h.update(f"{float(temperature):.3f}\n".encode())
    h.update(normalize_prompt(prompt).encode("utf-8"))
    return h.hexdigest()[:32]


class Cassette:
    def __init__(self, path: str, mode: str = "replay", live: Optional[Callable] = None):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        self.path, self.mode, self.live = path, mode, live
        self.entries: dict = {}
        self.misses: list = []
        self.calls = 0
        self._cursor: dict = {}
        if mode == "replay":
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, encoding="utf-8") as f:
                self.entries = json.load(f)["entries"]

    def __call__(self, prompt_text: str, temperature: float = 0.1, **kwargs) -> str:
        self.calls += 1
        key = _key(prompt_text, temperature)
        if self.mode == "record":
            response = self.live(prompt_text, temperature, **kwargs)
            entry = self.entries.setdefault(key, {"head": normalize_prompt(prompt_text)[:90], "responses": []})
            entry["responses"].append(response)
            return response
        entry = self.entries.get(key)
        if entry is None:
            head = normalize_prompt(prompt_text)[:90]
            self.misses.append(head)
            raise CassetteMiss(f"no recorded response for this prompt (it changed since recording): {head!r}")
        n = self._cursor.get(key, 0)
        self._cursor[key] = n + 1
        return entry["responses"][min(n, len(entry["responses"]) - 1)]

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": self.entries}, f, indent=1, sort_keys=True)

    @contextlib.contextmanager
    def installed(self):
        """Route the analysis pipeline's model calls through this cassette."""
        from analysis import query_engine as qe
        original = qe.llm_with_fallback
        if self.mode == "record" and self.live is None:
            self.live = original
        qe.llm_with_fallback = self
        try:
            yield self
        finally:
            qe.llm_with_fallback = original
