"""
pii_redaction.py — masks likely personally-identifiable values out of text
BEFORE it is embedded into any LLM prompt, and scans uploaded tables so the
user can see what was found.

WHY THIS EXISTS
    Every dataset this app profiles gets summarized into text handed to a
    third-party LLM API (Groq) — see profiler.format_profile_for_llm and
    format_defect_profile_for_llm, which include actual distinct category
    values (build_defect_profile's "all_distinct_values"), and
    rag_engine.build_rag_prompt/build_insight_extraction_prompt, which
    embed real passages of an uploaded document. If a column happens to
    hold emails, phone numbers, card numbers, or national ID numbers —
    entirely plausible in a real business dataset (a customer list, an
    HR export, an invoice register) — those values would otherwise be
    sent to Groq's API verbatim.

    This module is deliberately a LAST LINE before the network call, not
    a replacement for not collecting sensitive data in the first place —
    see the module docstring in prompt_injection_guard.py for the
    parallel reasoning about untrusted data more generally.

THREE LAYERS (cheapest and most reliable first)
    1. PATTERN REDACTION (redact_pii): regexes for emails, UPI IDs, card
       numbers, Aadhaar, US SSN, PAN, Indian mobile numbers and separator-
       formatted phone numbers. Validated where a checksum exists (Luhn
       for cards, Verhoeff for Aadhaar) so a random long number is not
       mistaken for an ID.
    2. COLUMN-NAME WITHHOLDING (personal_columns / withhold_personal_in_profile
       / mask_personal_columns): a person's NAME has no reliable pattern,
       so regexes can never catch it. But a column called first_name,
       customer_name or billing_address tells us what it holds. For those
       columns the values are withheld from every LLM prompt entirely.
       A bare column called just "name" is ambiguous (could be a product),
       so it is only REPORTED for review, not withheld.
    3. OPTIONAL NER (set PII_USE_NER=1): a spaCy named-entity pass that
       catches person names inside free text. Off by default because it
       needs an extra dependency and model download.

WHAT THIS DOES NOT DO
    - It never touches the DataFrame itself, sandboxed execution, or
      anything returned to the frontend — analysis results still show
      the user their own real data. This only scrubs text handed to the
      external LLM API.
    - Pattern redaction is a floor, not a guarantee. A name inside a free-
      text column, a street address typed into a "notes" field, or an ID
      in an unusual format can still get through unless NER is enabled.
    - Redaction is deliberately biased toward over-redacting: a false
      positive costs the LLM one masked value, a false negative leaks one.
    - It does not decide whether the user has the right to upload the
      data. It redacts either way.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# CHECKSUM VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

# Verhoeff tables (public-domain algorithm; used by India's Aadhaar numbers).
_V_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_V_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_V_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    """True if the digit string passes the Verhoeff checksum."""
    digits = [int(c) for c in number if c.isdigit()]
    if not digits:
        return False
    c = 0
    for i, digit in enumerate(reversed(digits)):
        c = _V_D[c][_V_P[i % 8][digit]]
    return c == 0


def verhoeff_check_digit(number: str) -> int:
    """Check digit to append to `number` so the result passes verhoeff_valid.
    Used by tests to build a valid Aadhaar-shaped number."""
    c = 0
    for i, ch in enumerate(reversed([d for d in number if d.isdigit()])):
        c = _V_D[c][_V_P[(i + 1) % 8][int(ch)]]
    return _V_INV[c]


def luhn_valid(number: str) -> bool:
    """True if the digit string passes the Luhn checksum (payment cards)."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_ok(s: str) -> bool:
    # A Luhn-valid number is a card. A 13-19 digit number that is visibly
    # grouped with spaces/dashes also reads as a card even if it fails
    # Luhn (a typo). A bare unseparated run that fails Luhn is far more
    # likely a timestamp or an ID, so it is left alone.
    return luhn_valid(s) or bool(re.search(r"[ -]", s))


def _aadhaar_ok(s: str) -> bool:
    return verhoeff_valid(s)


# ─────────────────────────────────────────────────────────────────────────────
# PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

# Digit patterns refuse to start mid-number or end before a decimal part
# ("mean=7234567890.5" is a statistic, not a phone number).
_NB = r"(?<![\d.])"       # not preceded by a digit or dot
_NA = r"(?!\d|\.\d)"      # not followed by a digit or ".digit"

_UPI_HANDLES = (
    "okhdfcbank|okicici|okaxis|oksbi|ybl|ibl|axl|paytm|upi|apl|fbl|sbi|icici|"
    "hdfcbank|axisbank|pnb|barodampay|aubank|cnrb|ezetap|idfcbank|jupiteraxis|"
    "freecharge|postbank|rbl|yesbank|kotak|airtel|jio"
)

# (label, compiled pattern, optional validator). Order matters: more
# specific / checksum-validated patterns first, the loose phone pattern last.
_PATTERNS: list[tuple[str, re.Pattern, Optional[Callable[[str], bool]]]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), None),
    ("UPI_ID", re.compile(rf"\b[\w.\-]{{2,64}}@(?:{_UPI_HANDLES})\b", re.IGNORECASE), None),
    ("CREDIT_CARD", re.compile(rf"{_NB}\d(?:[ -]?\d){{12,18}}{_NA}"), _card_ok),
    # 12 digits, first digit 2-9, optionally grouped 4-4-4, Verhoeff-valid.
    ("AADHAAR", re.compile(rf"{_NB}[2-9]\d{{3}}[ -]?\d{{4}}[ -]?\d{{4}}{_NA}"), _aadhaar_ok),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    # PAN: 5 letters, 4 digits, 1 letter; 4th letter is the holder type.
    ("PAN", re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b"), None),
    # Indian mobile: optional +91/91/0 prefix, starts 6-9, 10 digits,
    # optionally split 5-5 ("98765 43210").
    ("INDIAN_MOBILE", re.compile(rf"{_NB}(?:\+?91[ -]?|0)?[6-9]\d{{4}}[ -]?\d{{5}}{_NA}"), None),
    # Separator-formatted phone numbers of any country. Requires a
    # separator so plain numeric IDs/quantities are not flagged.
    ("PHONE", re.compile(
        r"(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{2,4}[ -]\d{3,4}[ -]\d{3,4}(?:[ -]\d{2,4})?"
    ), None),
]

PATTERN_LABELS = [label for label, _, _ in _PATTERNS]

# ── optional NER hook ───────────────────────────────────────────────────────
# A backend is any callable text -> list of (start, end) spans of PERSON names.
_ner_backend: Optional[Callable[[str], list[tuple[int, int]]]] = None
_ner_init_attempted = False


def set_ner_backend(fn: Optional[Callable[[str], list[tuple[int, int]]]]) -> None:
    """Install (or clear, with None) a name-detection backend. Tests use
    this to inject a fake; production uses _load_spacy_backend() via
    PII_USE_NER=1."""
    global _ner_backend, _ner_init_attempted
    _ner_backend = fn
    _ner_init_attempted = fn is not None


def _load_spacy_backend() -> Optional[Callable[[str], list[tuple[int, int]]]]:
    """Adapter for spaCy's small English model. NOT exercised by this
    repo's tests (spaCy isn't a project dependency) — the hook around it
    (set_ner_backend + _apply_ner) is tested with a fake instead. Fails
    soft: any import/model problem just disables NER with one log line."""
    try:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except Exception as e:  # ImportError or missing model
        print(f"[pii_redaction] PII_USE_NER=1 but spaCy/en_core_web_sm unavailable ({e}); "
              f"continuing with pattern redaction only", flush=True)
        return None

    def spans(text: str) -> list[tuple[int, int]]:
        return [(e.start_char, e.end_char) for e in nlp(text).ents if e.label_ == "PERSON"]

    return spans


def _apply_ner(text: str) -> tuple[str, int]:
    global _ner_backend, _ner_init_attempted
    if _ner_backend is None and not _ner_init_attempted and os.environ.get("PII_USE_NER") == "1":
        _ner_init_attempted = True
        _ner_backend = _load_spacy_backend()
    if _ner_backend is None:
        return text, 0
    try:
        spans = sorted(_ner_backend(text), reverse=True)
    except Exception as e:
        print(f"[pii_redaction] NER backend failed ({e}); skipping name redaction for this text", flush=True)
        return text, 0
    for start, end in spans:
        text = text[:start] + "[REDACTED_NAME]" + text[end:]
    return text, len(spans)


# ─────────────────────────────────────────────────────────────────────────────
# REDACTION
# ─────────────────────────────────────────────────────────────────────────────

def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Returns (redacted_text, counts) where counts maps each category
    name to how many matches were replaced. An empty counts dict means
    nothing matched — the common case, and not itself worth logging."""
    counts: dict[str, int] = {}
    for label, pattern, validator in _PATTERNS:
        n = 0

        def repl(m: re.Match, _label=label, _validator=validator) -> str:
            nonlocal n
            s = m.group(0)
            if _validator is None or _validator(s):
                n += 1
                return f"[REDACTED_{_label}]"
            return s

        text = pattern.sub(repl, text)
        if n:
            counts[label] = n

    text, n_names = _apply_ner(text)
    if n_names:
        counts["NAME"] = n_names
    return text, counts


def redact_pii_logged(text: str, context: str) -> str:
    """Same as redact_pii, but prints a one-line notice and writes an
    audit event whenever something was actually redacted — so a real
    redaction is debuggable from server logs instead of silently changing
    what the LLM sees. Never logs the redacted VALUE itself, only the
    category and count."""
    redacted, counts = redact_pii(text)
    if counts:
        summary = ", ".join(f"{n}x {label}" for label, n in counts.items())
        print(f"[pii_redaction] {context}: redacted {summary} before sending to LLM", flush=True)
        try:
            import audit_log
            audit_log.record("pii_redacted", context=context, counts=counts)
        except Exception:
            pass  # audit must never break redaction
    return redacted


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN-NAME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_PERSON_PREFIXES = {
    "first", "last", "middle", "full", "given", "family", "sur", "maiden",
    "customer", "client", "employee", "patient", "student", "person", "user",
    "owner", "contact", "applicant", "member", "guest", "passenger", "driver",
    "parent", "guardian", "staff", "emp", "cust", "teacher", "faculty",
    "candidate", "nominee", "beneficiary", "borrower", "holder", "cardholder",
    "father", "mother", "spouse", "nominee",
}
# A column whose LAST word is one of these holds that kind of personal value.
_PERSONAL_LAST_TOKENS = {
    "email", "mail", "phone", "mobile", "contact", "address", "aadhaar", "aadhar",
    "ssn", "passport", "dob", "birthdate", "birthday", "pan", "surname",
}
_PERSONAL_COMPACT = {
    "firstname", "lastname", "middlename", "fullname", "givenname", "surname",
    "familyname", "maidenname", "dateofbirth", "dob", "birthdate", "emailaddress",
    "phonenumber", "mobilenumber", "aadhaarnumber", "aadharnumber", "pannumber",
    "passportnumber", "homeaddress", "streetaddress", "mailingaddress",
}


def _tokens(name: object) -> list[str]:
    s = str(name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)          # camelCase -> words
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def classify_column_name(name: object) -> Optional[str]:
    """'personal'  — the name says it holds personal data (withheld from prompts)
       'review'    — ambiguous (a bare 'name'); reported, not withheld
       None        — nothing suspicious"""
    toks = _tokens(name)
    if not toks:
        return None
    compact = "".join(toks)
    if compact in _PERSONAL_COMPACT:
        return "personal"
    last = toks[-1]
    if last in _PERSONAL_LAST_TOKENS and not (len(toks) > 1 and last == "contact"):
        return "personal"
    if last == "name" and len(toks) > 1 and any(t in _PERSON_PREFIXES for t in toks[:-1]):
        return "personal"
    if compact == "name":
        return "review"
    return None


def withholding_enabled() -> bool:
    return os.environ.get("PII_WITHHOLD_PERSONAL_COLUMNS", "1") != "0"


def personal_columns(columns) -> list:
    """Columns whose VALUES must never reach an LLM prompt, judged by name."""
    if not withholding_enabled():
        return []
    return [c for c in columns if classify_column_name(c) == "personal"]


WITHHELD = "<personal data withheld>"


def mask_personal_columns(df):
    """Copy of `df` with personal-looking columns replaced by a placeholder.
    For the few prompts that embed real sample rows."""
    cols = [c for c in personal_columns(df.columns) if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = WITHHELD
    return out


def withhold_personal_in_profile(profile: dict) -> dict:
    """Returns a shallow copy of a profiler profile dict with every
    value-bearing entry for personal columns replaced: category values,
    and numeric min/max (a phone-number column's min/max ARE real phone
    numbers). Works for both build_dataset_profile and build_defect_profile
    shapes."""
    cols = set(map(str, personal_columns(
        set(profile.get("top_categories", {})) | set(profile.get("categorical_values", {}))
        | set(profile.get("numeric_stats", {})) | set(profile.get("outliers", {}))
    )))
    if not cols:
        return profile
    out = dict(profile)
    if "top_categories" in out:
        out["top_categories"] = {
            c: ({WITHHELD: len(v)} if str(c) in cols else v) for c, v in out["top_categories"].items()
        }
    if "categorical_values" in out:
        out["categorical_values"] = {
            c: ({"all_distinct_values": {WITHHELD: len(v.get("all_distinct_values", {}))},
                 "note": "values withheld: column name indicates personal data"}
                if str(c) in cols else v)
            for c, v in out["categorical_values"].items()
        }
    for key in ("numeric_stats", "outliers"):
        if key in out:
            out[key] = {c: v for c, v in out[key].items() if str(c) not in cols}
    if "strong_correlations" in out:
        out["strong_correlations"] = [
            c for c in out["strong_correlations"]
            if str(c.get("col_a")) not in cols and str(c.get("col_b")) not in cols
        ]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME SCAN — what the user sees BEFORE anything is sent to an LLM
# ─────────────────────────────────────────────────────────────────────────────

_SCAN_VALUES_PER_COLUMN = 1000
_HIGH_PCT = 20.0


def scan_dataframe(df) -> dict:
    """Scans a table and reports which columns look like personal data and
    why. Pure reporting: it changes nothing. The /profile endpoint returns
    this as `piiReport` so the UI can show it next to the preview.

    Severity per column: 'high' if >= 20% of sampled values match a PII
    pattern, or the column name says it holds personal data; 'medium' if
    some values match; 'low' for an ambiguous bare 'name' column.
    """
    flagged = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) > _SCAN_VALUES_PER_COLUMN:
            series = series.sample(_SCAN_VALUES_PER_COLUMN, random_state=42)
        # Whole numbers stored as float (9876543210.0) -> "9876543210"
        values = []
        for v in series:
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            values.append(str(v))

        categories: dict[str, dict] = {}
        for v in values:
            _, counts = redact_pii(v)
            for label in counts:
                if label == "NAME":
                    continue
                categories.setdefault(label, {"count": 0})["count"] += 1
        n = max(len(values), 1)
        for info in categories.values():
            info["pct"] = round(info["count"] / n * 100, 1)

        name_hint = classify_column_name(col)
        if not categories and not name_hint:
            continue

        top_pct = max((i["pct"] for i in categories.values()), default=0.0)
        if name_hint == "personal" or top_pct >= _HIGH_PCT:
            severity = "high"
        elif categories:
            severity = "medium"
        else:
            severity = "low"

        flagged.append({
            "column": str(col),
            "severity": severity,
            "categories": categories,
            "nameHint": name_hint,
            "withheldFromLlm": bool(name_hint == "personal" and withholding_enabled()),
        })

    order = {"high": 0, "medium": 1, "low": 2}
    flagged.sort(key=lambda f: (order[f["severity"]], f["column"]))
    risk = "none"
    if any(f["severity"] == "high" for f in flagged):
        risk = "high"
    elif flagged:
        risk = "medium" if any(f["severity"] == "medium" for f in flagged) else "low"

    if not flagged:
        summary = "No personal data patterns found. Values are still redacted from LLM prompts if any appear later."
    else:
        high = [f["column"] for f in flagged if f["severity"] == "high"]
        parts = [f"{len(flagged)} column(s) may contain personal data"]
        if high:
            parts.append("most likely: " + ", ".join(high[:5]))
        parts.append(
            "emails, phone numbers and ID numbers are masked, and columns named like "
            "personal data are withheld, before anything is sent to the AI model"
        )
        summary = "; ".join(parts) + "."

    return {
        "risk": risk,
        "flaggedColumns": flagged,
        "summary": summary,
        "scannedValuesPerColumn": _SCAN_VALUES_PER_COLUMN,
    }
