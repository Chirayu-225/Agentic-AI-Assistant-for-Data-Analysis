"""
insight_verifier.py — deterministic check that every number an insight
states actually came out of a sandboxed computation.

WHY THIS EXISTS
    Auto-Analyze used to ask the chat model for five insights written
    straight from a statistical profile. Nothing checked the numbers in
    that prose, so "Q4 is 2.3x Q1" could be invented and still be shown
    with the same weight as a computed answer. The pipeline is now:

        plan (LLM proposes question + pandas code)
          -> compute (sandbox runs the code against the real data)
          -> write (LLM phrases a finding from the computed result only)
          -> verify (THIS MODULE: every number in the finding must match a
             number in the computed result, or the insight is dropped)

    Verification is plain Python, not another model call, so it cannot
    be talked out of its answer.

WORDING CHECKS
    Matching numbers isn't enough: "grew from 190,989 to 362,813, more than
    doubling" has two real numbers and a false claim (1.9x). Ratio words
    (double, triple, half...) must be supported by the figures in the same
    sentence, "majority" needs a share above 50%, and qualifiers a single
    figure can't establish ("consistently", "every month") are rejected.

WHAT COUNTS AS A MATCH
    A number in the finding matches if some number in the evidence
    (the computed result) agrees with it at the precision it was written
    at — "39%" matches 0.3912 (as a share) or 39.12, "$1.2M" matches
    1,204,331, "3,600" matches 3600.0. Small bare integers (<= 10) that are
    not followed by %, x or a currency/magnitude marker are treated as
    structure ("the top 3 regions") rather than claims.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

_MAX_EVIDENCE_NUMBERS = 5000

_NUM_RE = re.compile(
    r"""
    (?P<cur>[$€£₹])?\s*
    (?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)
    \s*(?P<suffix>%|[kKmMbB](?![a-zA-Z])|bn|mn|lakh|crore|x|×)?
    """,
    re.VERBOSE,
)

_MAGNITUDE = {"k": 1e3, "m": 1e6, "mn": 1e6, "b": 1e9, "bn": 1e9, "lakh": 1e5, "crore": 1e7}


def extract_claimed_numbers(text: str) -> list[dict]:
    """Numbers stated in prose, with how precisely they were written."""
    claims = []
    for m in _NUM_RE.finditer(text):
        raw = m.group("num")
        # Skip years and things that look like ids glued to words (Q4, FY2024).
        start = m.start("num")
        if start > 0 and text[start - 1].isalpha():
            continue
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        suffix = (m.group("suffix") or "").lower()
        mult = _MAGNITUDE.get(suffix, 1.0)
        structural = (
            not m.group("cur") and suffix == "" and decimals == 0
            and float(value).is_integer() and 0 <= value <= 10
        )
        if not m.group("cur") and suffix == "" and decimals == 0 and 1900 <= value <= 2100:
            structural = True  # a year, not a measured quantity
        claims.append({
            "text": m.group(0).strip(),
            "value": value * mult,
            "decimals": decimals,
            "suffix": suffix,
            "scaled": mult != 1.0,
            "structural": structural,
        })
    return claims


def _iter_numbers(obj: Any) -> Iterable[float]:
    if obj is None:
        return
    if isinstance(obj, (bool, np.bool_)):
        return
    if isinstance(obj, (int, float, np.integer, np.floating)):
        f = float(obj)
        if math.isfinite(f):
            yield f
        return
    if isinstance(obj, pd.DataFrame):
        yield float(len(obj))
        for col in obj.columns:
            yield from _series_numbers(obj[col])
        if not isinstance(obj.index, pd.RangeIndex):
            yield from _series_numbers(pd.Series(obj.index))
        return
    if isinstance(obj, pd.Series):
        yield float(len(obj))
        yield from _series_numbers(obj)
        yield from _series_numbers(pd.Series(obj.index))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_numbers(v)
        return
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        for v in obj:
            yield from _iter_numbers(v)
        return
    if isinstance(obj, str):
        for c in extract_claimed_numbers(obj):
            yield c["value"]


def _series_numbers(series: pd.Series) -> Iterable[float]:
    """Numbers in a column, including ones stored as text ("₹95,235.87",
    "12%") — read the way a person would read the cell."""
    numeric = pd.to_numeric(series, errors="coerce")
    yield from _iter_numbers(numeric.dropna().tolist())
    text_cells = series[numeric.isna() & series.notna()]
    for cell in text_cells.astype(str).head(500):
        for c in extract_claimed_numbers(cell):
            yield c["value"]


def evidence_numbers(value: Any) -> list[float]:
    nums = []
    for n in _iter_numbers(value):
        nums.append(n)
        if len(nums) >= _MAX_EVIDENCE_NUMBERS:
            break
    return nums


def _agrees(claim: dict, candidate: float) -> bool:
    target = claim["value"]
    if claim["scaled"]:
        # "$1.2M" is a rounded figure: accept anything that rounds to it.
        return abs(candidate - target) <= 0.05 * abs(target) + 1e-9
    tol = 0.5 * 10 ** (-claim["decimals"])
    if abs(candidate - target) <= tol + 1e-9:
        return True
    # Allow 1% relative slack for long figures written loosely (74,100 vs 74,123.5).
    if abs(target) >= 100 and abs(candidate - target) <= 0.01 * abs(target):
        return True
    return False


def verify_claim(claim: dict, evidence: list[float]) -> bool:
    if claim["structural"]:
        return True
    for e in evidence:
        if _agrees(claim, e):
            return True
        if claim["suffix"] == "%" and _agrees(claim, e * 100):
            return True  # computed as a share (0.39), written as 39%
        if claim["suffix"] in ("x", "×") and _agrees(claim, e):
            return True
    return False


# Words that assert a size relationship between two figures. The finding
# must contain two figures whose ratio actually supports the word.
_RATIO_WORDS = [
    # (pattern, minimum ratio between the largest and smallest figure, label)
    (re.compile(r"\bmore than (doubl\w*|twice)\b", re.I), 2.0, "more than double"),
    (re.compile(r"\b(doubl\w*|twice)\b", re.I), 1.9, "double"),
    (re.compile(r"\bmore than tripl\w*\b", re.I), 3.0, "more than triple"),
    (re.compile(r"\b(tripl\w*|three times)\b", re.I), 2.85, "triple"),
    (re.compile(r"\b(quadrupl\w*|four times)\b", re.I), 3.8, "quadruple"),
    (re.compile(r"\b(halv\w*|half)\b", re.I), 1.8, "half"),
]
# Claims about every period or member that a single computed figure can't
# establish ("consistently", "every month"). A finding using one is dropped;
# the writer prompt tells the model not to use them.
_UNVERIFIABLE_QUALIFIERS = re.compile(
    r"\b(consistently|steadily|always|never|without exception|every (single )?(month|year|quarter|week|day))\b",
    re.I,
)
_MAJORITY = re.compile(r"\b(majority|most of)\b", re.I)


def _wording_problems(finding: str, claims: list[dict]) -> list[str]:
    problems = []
    figures = sorted(abs(c["value"]) for c in claims if c["suffix"] not in ("x", "×") and c["value"] != 0)
    for pattern, min_ratio, label in _RATIO_WORDS:
        m = pattern.search(finding)
        if not m:
            continue
        ratio = (figures[-1] / figures[0]) if len(figures) >= 2 else None
        # "almost / nearly / roughly doubled" claims less than "doubled"
        hedged = re.search(r"\b(almost|nearly|roughly|about|approximately|close to)\s+$",
                           finding[: m.start()], re.I)
        needed = min_ratio * 0.9 if hedged else min_ratio
        if ratio is None or ratio < needed:
            shown = f"{ratio:.2f}x" if ratio else "no two figures to compare"
            problems.append(f"'{m.group(0)}' not supported ({shown})")
        break  # the most specific pattern that matched decides
    q = _UNVERIFIABLE_QUALIFIERS.search(finding)
    if q:
        problems.append(f"'{q.group(0)}' can't be established from one computed result")
    if _MAJORITY.search(finding):
        pcts = [c["value"] for c in claims if c["suffix"] == "%"]
        if not any(p > 50 for p in pcts):
            problems.append("'majority' without a share above 50%")
    return problems


_CURRENCY_IN_TEXT = [
    (re.compile(r"₹|\brs\.?(?=\s*\d)|\binr\b", re.I), "₹"),
    (re.compile(r"\$|\busd\b", re.I), "$"),
    (re.compile(r"€|\beur\b", re.I), "€"),
    (re.compile(r"£|\bgbp\b", re.I), "£"),
]
# Data artefacts that are never a business finding.
_ARTEFACTS = re.compile(r"\bNaN\b|\bNone\b|\bnull\b|(?<![\d.])-?[01]\.000\b")


def _presentation_problems(finding: str, currencies: set | None) -> list[str]:
    problems = []
    if currencies is not None:
        allowed = {c.split()[0] for c in currencies}  # "₹ (INR)" -> "₹"
        for pattern, symbol in _CURRENCY_IN_TEXT:
            if pattern.search(finding) and symbol not in allowed:
                problems.append(
                    f"uses {symbol} but the data's currency is "
                    f"{' / '.join(sorted(allowed)) if allowed else 'not stated'}"
                )
    if _ARTEFACTS.search(finding):
        problems.append("reports a raw data artefact (NaN / raw ratio), not a business finding")
    return problems


def verify_finding(finding: str, evidence_value: Any, currencies: set | None = None) -> dict:
    """Returns {verified, checked: [...], unsupported: [...]}.

    `currencies` is the set of currency labels found in the raw data (e.g.
    {"₹ (INR)"}); when given, a finding using any other currency symbol is
    rejected (an empty set means the data has no currency, so no symbol is
    allowed). None skips the currency check.

    A finding with no quantified claim at all is NOT verified — the whole
    point of Auto-Analyze insights is a quantified, computed finding."""
    evidence = evidence_numbers(evidence_value)
    claims = [c for c in extract_claimed_numbers(finding) if not c["structural"]]
    unsupported = [c["text"] for c in claims if not verify_claim(c, evidence)]
    unsupported += _wording_problems(finding, claims)
    unsupported += _presentation_problems(finding, currencies)
    return {
        "verified": bool(claims) and not unsupported,
        "checked": [c["text"] for c in claims],
        "unsupported": unsupported,
    }
