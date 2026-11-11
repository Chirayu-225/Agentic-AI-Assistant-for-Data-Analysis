"""
llm/privacy.py - nothing personal leaves the machine inside a prompt.

Gateway.call() runs every payload through `sanitize_payload` BEFORE the size/raw-data guard and
before the cache key is computed, so no caller can forget it. Two layers, as in AST:

  1. column-name withholding: values of columns that look personal (email, phone, name, ...)
     are replaced by a placeholder, including samples, top values, min/max and evidence;
     samples that belong to no single column (e.g. duplicate rows) are always dropped;
  2. pattern redaction of every remaining free-text string (emails, cards, Aadhaar, phones...).

When AST's own `pii_redaction.py` is importable (this package sits in `AST 2.0/`), its
classifier and redactor are used, so there is ONE definition of "personal" across the project
and the PII_WITHHOLD_PERSONAL_COLUMNS switch is honoured. Otherwise a smaller built-in fallback
runs. `backend` reports which one is active.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from typing import Any, Callable

WITHHELD = "<personal data withheld>"

# Keys that carry values taken from the data.
VALUE_KEYS = frozenset({
    "samples", "sample_values", "top_values", "examples", "values", "distinct_values", "categories",
    "min", "max", "mean", "median", "mode", "evidence", "before", "after", "value", "mapping",
})
# Keys whose strings are structural (names, ids, enums) and are left untouched.
STRUCTURAL_KEYS = frozenset({
    "op", "type", "id", "issue_id", "column", "name", "col", "strategy", "kind", "dtype", "task",
    "severity", "confidence", "source", "method", "format", "columns", "subset", "group_by",
})
COLUMN_KEYS = ("column", "name", "col")


# ---- fallback implementation ---------------------------------------------------------
_PERSONAL_TOKENS = {
    "email", "mail", "phone", "mobile", "ssn", "aadhaar", "aadhar", "pan", "passport", "address",
    "dob", "birthdate", "birthday", "surname", "firstname", "lastname", "fullname", "iban", "card",
}
_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("CARD", re.compile(r"(?<![\d.])(?:\d[ -]?){13,19}(?![\d.])")),
    ("PHONE", re.compile(r"(?<![\d.])(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3,5}\)?[ -]?)\d{3}[ -]?\d{3,4}(?![\d.])")),
]


def _fallback_classify(name: object) -> str | None:
    toks = [t for t in re.split(r"[^a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name)).lower()) if t]
    compact = "".join(toks)
    if compact in _PERSONAL_TOKENS or (toks and toks[-1] in _PERSONAL_TOKENS):
        return "personal"
    if len(toks) > 1 and toks[-1] == "name" and toks[0] in {"first", "last", "full", "middle", "customer",
                                                            "employee", "patient", "user", "person"}:
        return "personal"
    return None


def _fallback_redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pat in _PATTERNS:
        text, n = pat.subn(f"[REDACTED_{label}]", text)
        if n:
            counts[label] = n
    return text, counts


def _load_ast_backend() -> tuple[Callable, Callable, Callable] | None:
    if os.environ.get("DCA_PRIVACY_BACKEND", "").lower() == "fallback":
        return None
    for attempt in (0, 1):
        try:
            mod = importlib.import_module("pii_redaction")
            return mod.redact_pii, mod.classify_column_name, mod.withholding_enabled
        except Exception:
            if attempt:
                return None
            parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent not in sys.path:
                sys.path.append(parent)  # AST 2.0/, where pii_redaction.py lives
    return None


class Sanitizer:
    def __init__(self, backend: str | None = None):
        loaded = None if backend == "fallback" else _load_ast_backend()
        if backend == "ast" and loaded is None:
            raise ImportError("AST's pii_redaction.py is not importable")
        if loaded:
            self.backend = "ast"
            self._redact, self._classify, self._withhold_on = loaded
        else:
            self.backend = "fallback"
            self._redact, self._classify = _fallback_redact, _fallback_classify
            self._withhold_on = lambda: os.environ.get("PII_WITHHOLD_PERSONAL_COLUMNS", "1") != "0"
        self.columns_withheld: set[str] = set()
        self.redactions = 0

    def _personal(self, name: Any) -> bool:
        return isinstance(name, str) and self._withhold_on() and self._classify(name) == "personal"

    def _text(self, s: str) -> str:
        out, counts = self._redact(s)
        self.redactions += sum(counts.values())
        return out

    def _withhold(self, node: Any) -> Any:
        if isinstance(node, (list, tuple)):
            return [WITHHELD] if node else []
        if isinstance(node, dict):
            return {WITHHELD: len(node)} if node else {}
        return WITHHELD if node is not None else None

    def clean(self, node: Any, *, structural: bool = False, withheld: bool = False) -> Any:
        if isinstance(node, dict):
            col = next((node[k] for k in COLUMN_KEYS if isinstance(node.get(k), str)), None)
            node_personal = self._personal(col)
            if node_personal:
                self.columns_withheld.add(col)
            out: dict[Any, Any] = {}
            for k, v in node.items():
                ks = str(k)
                if ks in VALUE_KEYS and (node_personal or (ks in ("samples", "sample_values", "examples")
                                                            and col is None)):
                    out[k] = self._withhold(v) if node_personal else []  # no-column samples: dropped
                elif self._personal(ks) and ks not in STRUCTURAL_KEYS and isinstance(v, (dict, list)):
                    self.columns_withheld.add(ks)  # a mapping keyed by a personal column name
                    out[k] = self._withhold(v)
                else:
                    out[k] = self.clean(v, structural=ks in STRUCTURAL_KEYS)
            return out
        if isinstance(node, (list, tuple)):
            return [self.clean(x, structural=structural) for x in node]
        if isinstance(node, str) and not structural:
            return self._text(node)
        return node


def sanitize_payload(payload: dict, sanitizer: Sanitizer | None = None) -> dict:
    """Return a cleaned deep copy of `payload` (the input is not modified)."""
    return (sanitizer or Sanitizer()).clean(payload)
