"""
core/detectors.py - deterministic issue detection. No LLM, no randomness.

Each detector takes the DataFrame (+ its profile) and returns Issue objects whose
`evidence` is JSON-safe. rules.py turns issues into operations; the LLM only ever
sees the compact issue evidence for issues the rules cannot resolve.

Known limits (by design, kept simple and explainable):
  * numbers are parsed strictly: "1,200" or "$5" are not treated as numeric
  * case / near-duplicate detection only looks at categorical (low-cardinality) text
  * date formats come from a fixed candidate list
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from core.profiler import ColumnProfile, DatasetProfile, profile_dataframe
from core.schemas import Issue
from core.utils import clean_ws, is_numeric, is_text, null_mask, short, string_values

try:  # rapidfuzz is preferred; difflib keeps the core usable without it
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # pragma: no cover - exercised only when rapidfuzz is absent
    _fuzz = None

NEAR_DUP_THRESHOLD = 90
OUTLIER_K = 1.5
MIN_OUTLIER_ROWS = 8
MAX_NEAR_DUP_UNIQUE = 200
DATE_SAMPLE = 2000

# Candidate formats, in priority order. Day-first precedes month-first on purpose
# (assumption documented in evidence["assumed"] when the column is ambiguous).
DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
    "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%y", "%m/%d/%y",
    "%d %b %Y", "%d-%b-%Y", "%b %d, %Y", "%B %d, %Y", "%d %B %Y",
]
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_LIKE = re.compile(  # non-capturing groups only (pandas warns about match groups)
    rf"(?:\d{{1,4}}[-/.]\d{{1,2}}[-/.]\d{{1,4}})|(?:\d{{1,2}}[ -]{_MONTHS}[ -,]*\d{{2,4}})|(?:{_MONTHS} \d{{1,2}},? \d{{2,4}})",
    re.IGNORECASE,
)
_IDENTIFIER_NAME = re.compile(
    r"(^|[_\s\-])(zip|zipcode|postal|phone|mobile|tel|pin|code|ssn|aadhaar|account|acct|id)($|[_\s\-])",
    re.IGNORECASE,
)


def _issue(
    type_: str, column: str | None, affected: int, n_rows: int,
    evidence: dict, samples: list[str] | None = None,
) -> Issue:
    return Issue(
        id=f"{type_}:{column}" if column else type_,
        type=type_,  # type: ignore[arg-type]
        column=column,
        affected_rows=int(affected),
        affected_pct=round(100 * affected / n_rows, 2) if n_rows else 0.0,
        evidence=evidence,
        samples=(samples or [])[:8],
    )


# --------------------------------------------------------------------------- #
# Individual detectors
# --------------------------------------------------------------------------- #
def detect_nulls(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        if c.null_count == 0:
            continue
        s = df[c.name]
        blanks = int((null_mask(s) & s.notna()).sum())
        out.append(_issue(
            "nulls", c.name, c.null_count, c.n_rows,
            {"null_pct": c.null_pct, "kind": c.kind, "dtype": c.dtype, "skew": c.skew,
             "blank_strings": blanks, "looks_like_id": c.looks_like_id},
            c.sample_values,
        ))
    return out


def detect_exact_duplicates(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    if profile.duplicate_rows == 0:
        return []
    dup_rows = df[df.duplicated(keep="first")].head(3)
    samples = [" | ".join(short(v, 20) for v in row)[:100] for row in dup_rows.itertuples(index=False)]
    return [_issue("exact_duplicates", None, profile.duplicate_rows, profile.n_rows,
                   {"extra_copies": profile.duplicate_rows}, samples)]


def _numeric_text_facts(strings: pd.Series) -> dict:
    """Facts about a column of strings that all parse as numbers."""
    stripped = strings.str.strip()
    leading_zero = bool(stripped.str.match(r"^[+-]?0\d").any())
    longest = int(stripped.str.replace(r"\D", "", regex=True).str.len().max() or 0)
    num = pd.to_numeric(stripped, errors="coerce")
    integral = bool((np.mod(num.to_numpy(dtype="float64"), 1) == 0).all())
    return {"leading_zeros": leading_zero, "max_digits": longest,
            "suggested_dtype": "int" if integral else "float"}


def detect_mixed_types(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        s = df[c.name]
        if not is_text(s) or c.kind == "empty":
            continue
        vals = string_values(s[~null_mask(s)])
        if len(vals) < 5:
            continue
        stripped = vals.str.strip()
        num = pd.to_numeric(stripped, errors="coerce")
        share = float(num.notna().mean())
        if share == 1.0:
            facts = _numeric_text_facts(vals)
            identifier = (
                facts["leading_zeros"] or facts["max_digits"] > 15 or bool(_IDENTIFIER_NAME.search(c.name))
            )
            out.append(_issue(
                "mixed_types", c.name, len(vals), c.n_rows,
                {"subtype": "numeric_as_text", "identifier_like": identifier, **facts,
                 "numeric_share": 1.0},
                c.sample_values,
            ))
        elif 0.5 <= share < 1.0:
            bad = stripped[num.isna()]
            top_bad = [v for v, _ in Counter(bad).most_common(8)]
            out.append(_issue(
                "mixed_types", c.name, len(bad), c.n_rows,
                {"subtype": "numeric_with_garbage", "numeric_share": round(share, 4),
                 "bad_distinct": int(bad.nunique()), "bad_values": top_bad},
                [short(v) for v in top_bad],
            ))
    return out


def _parse_rate(text: pd.Series, fmt: str) -> float:
    return float(pd.to_datetime(text, format=fmt, errors="coerce").notna().mean())


def detect_date_issues(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        s = df[c.name]
        if not is_text(s) or c.kind == "empty":
            continue
        vals = string_values(s[~null_mask(s)]).str.strip()
        if len(vals) < 5:
            continue
        step = max(len(vals) // DATE_SAMPLE, 1)
        sample = vals.iloc[::step]
        if float(sample.str.contains(_DATE_LIKE).mean()) < 0.6:
            continue
        rates = {f: _parse_rate(sample, f) for f in DATE_FORMATS}
        best = max(rates.values())
        if best < 0.8:
            continue
        tied = [f for f in DATE_FORMATS if rates[f] == best]
        fmt = tied[0]
        ambiguous = len(tied) > 1
        full = pd.to_datetime(vals, format=fmt, errors="coerce")
        bad_mask = full.isna()
        n_bad = int(bad_mask.sum())
        out.append(_issue(
            "date_like_strings", c.name, len(vals), c.n_rows,
            {"format": fmt, "parse_rate": round(1 - n_bad / len(vals), 4), "ambiguous": ambiguous,
             "alternatives": [f for f in tied[1:]], "assumed": "day-first" if ambiguous else None,
             "unparseable_count": n_bad},
            c.sample_values,
        ))
        if n_bad:
            bad_vals = [v for v, _ in Counter(vals[bad_mask]).most_common(8)]
            out.append(_issue(
                "unparseable_dates", c.name, n_bad, c.n_rows,
                {"format": fmt, "bad_distinct": int(vals[bad_mask].nunique())},
                [short(v) for v in bad_vals],
            ))
    return out


def detect_whitespace(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        s = df[c.name]
        if not is_text(s):
            continue
        txt = string_values(s)
        if txt.empty:
            continue
        normalised = txt.map(clean_ws)
        need = txt != normalised
        n = int(need.sum())
        if n == 0:
            continue
        edge = int((txt != txt.str.strip()).sum())
        out.append(_issue(
            "whitespace", c.name, n, c.n_rows,
            {"leading_trailing": edge, "internal_runs": n - edge if n >= edge else 0},
            [short(v) for v in Counter(txt[need]).keys()][:8],
        ))
    return out


def _case_style(v: str) -> str | None:
    if v.lower() == v.upper():  # no cased letters ("123", "-")
        return None
    if v == v.lower():
        return "lower"
    if v == v.upper():
        return "upper"
    if v == " ".join(w[:1].upper() + w[1:].lower() for w in v.split(" ")):
        return "title"
    return "mixed"


def detect_case_inconsistency(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        s = df[c.name]
        if c.kind != "categorical" or not is_text(s):
            continue
        txt = string_values(s).map(clean_ws)
        txt = txt[txt != ""]
        counts = txt.value_counts()
        groups: dict[str, list[tuple[str, int]]] = {}
        for v, n in counts.items():
            groups.setdefault(v.lower(), []).append((v, int(n)))
        clashing = {k: g for k, g in groups.items() if len(g) > 1}
        if not clashing:
            continue
        style_counts: Counter = Counter()
        for g in clashing.values():
            for v, n in g:
                st = _case_style(v)
                if st in ("lower", "upper", "title"):
                    style_counts[st] += n
        total = sum(style_counts.values())
        dominant, dom_n = (style_counts.most_common(1)[0] if style_counts else ("lower", 0))
        affected = sum(n for g in clashing.values() for _, n in g)
        examples = [{"variants": {v: n for v, n in g}} for g in list(clashing.values())[:5]]
        out.append(_issue(
            "case_inconsistency", c.name, affected, c.n_rows,
            {"clashing_groups": len(clashing), "dominant_style": dominant,
             "dominant_share": round(dom_n / total, 4) if total else 0.0, "examples": examples},
            [short(v) for g in list(clashing.values())[:4] for v, _ in g][:8],
        ))
    return out


def detect_outliers(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        if c.kind != "numeric" or c.looks_like_id or c.n_unique < 6:
            continue
        x = df[c.name].astype("float64").dropna()
        if len(x) < MIN_OUTLIER_ROWS:
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - OUTLIER_K * iqr, q3 + OUTLIER_K * iqr
        low, high = x < lo, x > hi
        n = int(low.sum() + high.sum())
        if n == 0:
            continue
        extremes = x[low | high]
        far = extremes.reindex((extremes - x.median()).abs().sort_values(ascending=False).index)
        out.append(_issue(
            "outliers", c.name, n, c.n_rows,
            {"method": "iqr", "k": OUTLIER_K, "q1": round(float(q1), 6), "q3": round(float(q3), 6),
             "lower_bound": round(float(lo), 6), "upper_bound": round(float(hi), 6),
             "n_low": int(low.sum()), "n_high": int(high.sum()), "min": c.min, "max": c.max,
             "median": c.median},
            [short(float(v)) for v in far.head(8)],
        ))
    return out


def detect_constant_columns(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        if c.n_rows < 2 or c.n_unique != 1:
            continue
        out.append(_issue(
            "constant_column", c.name, c.n_rows, c.n_rows,
            {"value": c.sample_values[0], "null_pct": c.null_pct}, c.sample_values,
        ))
    return out


def _norm_key(v: str) -> str:
    return clean_ws(re.sub(r"[^\w\s]", "", v.lower()))


def _similar(a: str, b: str) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False
    if a.isdigit() or b.isdigit():
        return False
    # "Group 1" vs "Group 2" differ only in digits: different things, not typos.
    if re.sub(r"\d+", "#", a) == re.sub(r"\d+", "#", b):
        return False
    if _fuzz is not None:
        score = max(_fuzz.ratio(a, b), _fuzz.token_sort_ratio(a, b))
    else:  # pragma: no cover
        score = 100 * SequenceMatcher(None, a, b).ratio()
    return score >= NEAR_DUP_THRESHOLD


def detect_near_duplicate_categories(df: pd.DataFrame, profile: DatasetProfile) -> list[Issue]:
    out = []
    for c in profile.columns:
        s = df[c.name]
        if c.kind != "categorical" or not is_text(s) or not (2 <= c.n_unique <= MAX_NEAR_DUP_UNIQUE):
            continue
        counts = string_values(s).map(clean_ws)
        counts = counts[counts != ""].value_counts()
        values = [str(v) for v in counts.index]
        keys = {v: _norm_key(v) for v in values}

        parent = {v: v for v in values}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(values):
            for b in values[i + 1:]:
                if keys[a] == keys[b] or _similar(keys[a], keys[b]):
                    parent[find(a)] = find(b)

        clusters: dict[str, list[str]] = {}
        for v in values:
            clusters.setdefault(find(v), []).append(v)

        found = []
        affected = 0
        for members in clusters.values():
            if len(members) < 2:
                continue
            if len({m.lower() for m in members}) < 2:  # case-only: the case detector owns it
                continue
            # canonical = most frequent; ties -> alphabetically first (deterministic)
            canonical = sorted(members, key=lambda m: (-int(counts[m]), m))[0]
            variants = [{"value": m, "count": int(counts[m])} for m in members if m != canonical]
            affected += sum(v["count"] for v in variants)
            found.append({"canonical": canonical, "canonical_count": int(counts[canonical]),
                          "variants": variants[:8]})
        if not found:
            continue
        samples = [f"{v['value']} -> {f['canonical']}" for f in found for v in f["variants"]][:8]
        out.append(_issue(
            "near_duplicate_categories", c.name, affected, c.n_rows,
            {"clusters": found[:10], "n_clusters": len(found), "threshold": NEAR_DUP_THRESHOLD},
            samples,
        ))
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
_DETECTORS = (
    detect_exact_duplicates,
    detect_whitespace,
    detect_case_inconsistency,
    detect_near_duplicate_categories,
    detect_date_issues,
    detect_mixed_types,
    detect_nulls,
    detect_outliers,
    detect_constant_columns,
)


def detect_all(df: pd.DataFrame, profile: DatasetProfile | None = None) -> list[Issue]:
    """Run every detector. Order is fixed, so output is reproducible."""
    profile = profile or profile_dataframe(df)
    issues: list[Issue] = []
    for fn in _DETECTORS:
        issues.extend(fn(df, profile))
    return issues
