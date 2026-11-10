"""
data_checks.py — deterministic, read-only checks on an uploaded table. DataFrame in, plain dicts out.

    check_dataframe(df) -> {"columns": [one record per column], "cards": [...], "n_rows": int}

No LLM, no I/O, no state, and the input is never modified. It imports only re, collections, pandas and the pure
detect_encoding module (for the encoding cards). It must NOT import cleaning, analysis_service, llm_provider or the
sandbox: this module is meant to be callable on a table before any of that machinery runs.

CHECK 1 — FORMAT: does a column that is mostly one shape contain cells in another shape?
  The SIGNATURE of a value: str(value), trimmed, then every ASCII digit -> 9, uppercase letter -> A, lowercase letter
  -> a, every other character kept ("(555) 867-5309" -> "(999) 999-9999"; "AB-12" -> "AA-99"). Missing values and
  the nine cleaner sentinels (_SENTINELS, trimmed, case-insensitive) are not counted.

  A column is checked only if ALL gates pass, in this order; the first one that fails is its skip_reason:
    fewer_than_100_values        at least 100 non-blank values
    less_than_half_have_digits   at least 50% of the non-blank values contain a digit
    too_many_formats             distinct signatures <= max(20, 5% of the non-blank values)
    plain_numbers                NOT every signature is a plain number (^[+-]?9+(\\.9+)?$): ids, counts and prices
                                 differ in length by nature, which is not a format problem

  Verdict for a checked column, D = share of the most common signature:
    D = 100%                     "consistent"  no card; message "<col>: all values use the format <sig>."
    D >= 90% (a minority exists) "drift"       one card per minority signature, at most the 2 largest per column;
                                 every card carries the column-level message with the TOTAL minority count
    D <  90%                     "variance"    ONE informational card, no per-row flags: two legitimate styles
                                 (or many) in one column is not an invalid value
  Ties between signatures are broken by the signature string, so the same input always gives the same output.

CHECK 2 — ENCODING ARTIFACTS: see detect_encoding.py (one card per text column). Independent of the format gates.

CARDS OVERALL
  Format cards and encoding cards are ranked together by the number of cells they affect (descending), then column
  position, then signature (then reason, only so the order is total). At most 5 cards per file. Encoding cards do
  not count against the 2-per-column format limit but do count toward the 5-per-file cap. At most 2 of the 5 may be
  variance cards (the first two in column order, as they all affect 0 cells), so they cannot push out other cards.

DECISIONS THE SPEC LEFT OPEN (so they can be changed in one place)
  * A variance card flags no cell, so for ranking it affects 0 cells (count 0) and ranks after every real finding.
  * Encoding cards get a message made here ("<col>: <count> of <of> values contain encoding artifacts (e.g. <example>).") and a signature of None, so every card has the same keys.
  * For a skipped column that got as far as the signature count (>= 100 non-blank values), the descriptive fields
    (dominant signature, number of signatures, top signatures) are still filled in; verdict stays "skipped".
  * Shares are floats in the records and whole percents (rounded half up) in the messages.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from detect_encoding import find_encoding_artifacts

# Keep a COPY of the cleaner's sentinel set (cleaning._SENTINELS). test_data_checks.py asserts the two are equal.
_SENTINELS = frozenset({"nan", "none", "null", "n/a", "na", "--", "-", "", "nat"})

MIN_VALUES = 100
MIN_DIGIT_SHARE_PCT = 50
DRIFT_MIN_SHARE_PCT = 90
MAX_SIGNATURES_FLOOR = 20
MAX_SIGNATURES_PCT = 5
MAX_MINORITY_CARDS_PER_COLUMN = 2
MAX_CARDS_PER_FILE = 5
MAX_VARIANCE_CARDS_PER_FILE = 2
TOP_SIGNATURES = 5

VERDICT_DRIFT, VERDICT_VARIANCE, VERDICT_CONSISTENT, VERDICT_SKIPPED = "drift", "variance", "consistent", "skipped"
REASON_DRIFT, REASON_VARIANCE, REASON_ENCODING = "format_drift", "format_variance", "encoding_artifact"

_PLAIN_NUMBER = re.compile(r"^[+-]?9+(\.9+)?$")


class _SignatureTable(dict):
    """The one str.translate table. Characters are classified the first time they are seen and then cached, so
    non-ASCII letters (é, Ã, ß) are handled without enumerating Unicode up front."""

    def __missing__(self, codepoint: int) -> str:
        ch = chr(codepoint)
        if "0" <= ch <= "9":
            out = "9"
        elif ch.isalpha() and ch.isupper():
            out = "A"
        elif ch.isalpha() and ch.islower():
            out = "a"
        else:
            out = ch
        self[codepoint] = out
        return out


_TABLE = _SignatureTable()


def signature(value) -> str:
    """The shape of one value (see the module docstring)."""
    return str(value).strip().translate(_TABLE)


def _pct(count: int, total: int) -> str:
    """Whole percent, rounded half up, in integer arithmetic (no float surprises at 90% or 97%)."""
    return f"{(200 * count + total) // (2 * total)}%"


def _non_blank_text(series: pd.Series) -> pd.Series:
    """Trimmed text of every non-missing, non-sentinel value, in row order."""
    text = series[series.notna()].astype(str).str.strip()
    return text[~text.str.lower().isin(_SENTINELS)]


def _count_signatures(text: pd.Series):
    """-> (Counter of signature -> count, {signature: first value in row order with it})."""
    counts: Counter = Counter()
    examples: dict = {}
    seen: dict = {}
    for value in text.tolist():
        sig = seen.get(value)
        if sig is None:
            sig = value.translate(_TABLE)
            seen[value] = sig
        counts[sig] += 1
        if sig not in examples:
            examples[sig] = value
    return counts, examples


def _record(name, n_checked: int) -> dict:
    return {"column": name, "verdict": VERDICT_SKIPPED, "skip_reason": None, "n_checked": n_checked,
            "dominant_signature": None, "dominant_share": None, "n_signatures": None, "top_signatures": [],
            "minority_cells": 0, "message": ""}


def _card(reason: str, name, index: int, signature_: str | None, count: int, of: int, example: str, message: str,
          **extra) -> dict:
    card = {"reason": reason, "column": name, "column_index": index, "signature": signature_, "count": count,
            "of": of, "example": example, "message": message}
    card.update(extra)
    return card


def _check_column(name, index: int, series: pd.Series):
    """-> (column record, format cards for this column)."""
    text = _non_blank_text(series)
    n = len(text)
    rec = _record(name, n)
    if n < MIN_VALUES:
        rec["skip_reason"] = "fewer_than_100_values"
        return rec, []

    counts, examples = _count_signatures(text)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))      # count desc, then signature string
    dominant, top = ordered[0]
    rec["dominant_signature"] = dominant
    rec["dominant_share"] = top / n
    rec["n_signatures"] = len(ordered)
    rec["top_signatures"] = [{"signature": s, "count": c, "share": c / n, "example": examples[s]}
                             for s, c in ordered[:TOP_SIGNATURES]]

    with_digit = sum(c for s, c in ordered if "9" in s)
    if 100 * with_digit < MIN_DIGIT_SHARE_PCT * n:
        rec["skip_reason"] = "less_than_half_have_digits"
        return rec, []
    # distinct signatures <= max(20, 5% of n)  <=>  NOT (more than 20 AND more than 5% of n), in integer arithmetic
    if len(ordered) > MAX_SIGNATURES_FLOOR and 100 * len(ordered) > MAX_SIGNATURES_PCT * n:
        rec["skip_reason"] = "too_many_formats"
        return rec, []
    if all(_PLAIN_NUMBER.match(s) for s, _ in ordered):
        rec["skip_reason"] = "plain_numbers"
        return rec, []

    col = str(name)
    minority_cells = n - top
    rec["minority_cells"] = minority_cells

    if len(ordered) == 1:
        rec["verdict"] = VERDICT_CONSISTENT
        rec["message"] = f"{col}: all values use the format {dominant}."
        return rec, []

    if 100 * top >= DRIFT_MIN_SHARE_PCT * n:
        rec["verdict"] = VERDICT_DRIFT
        minority = ordered[1:]
        rec["message"] = (f"{col}: {_pct(top, n)} of values use the format {dominant}; {minority_cells} values use "
                          f"another format (e.g. {examples[minority[0][0]]}). These may be legitimate variants.")
        cards = [_card(REASON_DRIFT, name, index, s, c, n, examples[s], rec["message"],
                       dominant_signature=dominant, minority_cells=minority_cells)
                 for s, c in minority[:MAX_MINORITY_CARDS_PER_COLUMN]]
        return rec, cards

    rec["verdict"] = VERDICT_VARIANCE
    rec["message"] = (f"{col} mixes {len(ordered)} formats; the most common covers {_pct(top, n)}. Counts of "
                      f"inconsistent values here may be formatting differences, not errors.")
    return rec, [_card(REASON_VARIANCE, name, index, None, 0, n, examples[dominant], rec["message"],
                       dominant_signature=dominant, minority_cells=minority_cells, informational=True)]


def _encoding_card(card: dict, index: int) -> dict:
    col = str(card["column"])
    message = f"{col}: {card['count']} of {card['of']} values contain encoding artifacts (e.g. {card['example']})."
    return _card(REASON_ENCODING, card["column"], index, None, card["count"], card["of"], card["example"], message,
                 kinds=list(card["kinds"]))


def check_dataframe(df: pd.DataFrame) -> dict:
    """Run every check. The input is only read, never changed. Same input -> identical output and order."""
    records: list[dict] = []
    cards: list[dict] = []
    for i in range(df.shape[1]):
        series = df.iloc[:, i]                                   # by position: duplicate column names are fine
        rec, format_cards = _check_column(df.columns[i], i, series)
        records.append(rec)
        cards.extend(format_cards)
        for enc in find_encoding_artifacts(df.iloc[:, [i]]):     # at most one; the position is known here
            cards.append(_encoding_card(enc, i))
    cards.sort(key=lambda c: (-c["count"], c["column_index"], c["signature"] or "", c["reason"]))
    kept: list[dict] = []
    n_variance = 0
    for card in cards:
        if card["reason"] == REASON_VARIANCE:
            if n_variance >= MAX_VARIANCE_CARDS_PER_FILE:       # informational cards must not crowd out findings
                continue
            n_variance += 1
        kept.append(card)
        if len(kept) == MAX_CARDS_PER_FILE:
            break
    return {"columns": records, "cards": kept, "n_rows": int(len(df))}
