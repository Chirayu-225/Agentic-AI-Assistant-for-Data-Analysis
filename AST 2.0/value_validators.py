"""
value_validators.py — deterministic, evidence-based repairs for the VALUE defects the experiment found the old pipeline
gets silently wrong (eval_results/exp): impossible negatives, future dates, an end before its start, mixed date formats,
swapped values, units written in the text, percent-scale mix-ups, magnitude (decimal-shift) outliers, near-duplicate rows.

No model is involved. Every rule
  * needs EVIDENCE in the table itself before it acts (a column name AND a distribution, or two columns that prove each
    other), and abstains when the evidence is not there;
  * changes only a small minority of a column (MAX_SHARE); a rule that would change more is a broken heuristic, so it
    does nothing;
  * blanks what cannot be recovered (a negative quantity, a date in 2098) instead of guessing, and restores only what the
    table itself pins down (a unit in the cell, a swapped pair, a value that is exactly 100x its column);
  * reports every change: column, rule, count, evidence and before/after examples, so nothing is silent.

validate_values(df) -> (repaired copy, report). The input is never modified. Running it again changes nothing.
This is NOT the cleaner: cleaning.py (legacy and safe) is untouched. It runs in the data-quality agent's deterministic
phase, after the fill step, so its blanks are never filled back in.
"""
from __future__ import annotations

import collections
import datetime as _dt
import difflib
import re

import numpy as np
import pandas as pd

MAX_SHARE = 0.25            # a rule that would change more than this share of a column abstains
MIN_VALUES = 20             # columns with fewer filled cells are too small for evidence
EXAMPLES = 3


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _tokens(name) -> list:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def _short(v, n=40) -> str:
    t = str(v)
    return t if len(t) <= n else t[: n - 1] + "…"


def _entry(column, rule, action, count, evidence, examples) -> dict:
    return {"column": str(column), "rule": rule, "action": action, "count": int(count), "evidence": evidence,
            "examples": examples[:EXAMPLES]}


# ── 1. a unit written in the cell ("18670 g", "15.40 kg") ───────────────────────────────────────────────────────
_UNIT_CELL = re.compile(r"^\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*([A-Za-z]{1,3})\s*$")
_FAMILIES = {
    "mass": {"mg": 1e-6, "g": 1e-3, "kg": 1.0, "lb": 0.45359237, "lbs": 0.45359237, "oz": 0.028349523125},
    "length": {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3, "in": 0.0254, "ft": 0.3048},
    "volume": {"ml": 1e-3, "l": 1.0},
}


def _fix_units(df, out, report):
    for col in df.columns:
        s = out[col]
        if not _is_text(s):
            continue
        vals = s.dropna().astype(str)
        if len(vals) < MIN_VALUES:
            continue
        m = vals.map(lambda t: _UNIT_CELL.match(t))
        if m.isna().any():                                  # every filled cell must be "number unit"
            continue
        units = m.map(lambda x: x.group(2).lower())
        fam = [f for f, table in _FAMILIES.items() if set(units) <= set(table)]
        if not fam:
            continue
        table = _FAMILIES[fam[0]]
        counts = collections.Counter(units)
        base = max(counts, key=lambda u: (counts[u], u))    # the majority unit; ties broken by name, deterministically
        nums = m.map(lambda x: float(x.group(1).replace(",", "")))
        conv = (nums * units.map(table) / table[base]).round(6)
        minority = int((units != base).sum())
        before = s.copy()
        out[col] = pd.to_numeric(pd.Series(conv.reindex(s.index), index=s.index), errors="coerce")
        ex = [{"row": int(i), "before": _short(before[i]), "after": float(out[col][i])} for i in units[units != base].index[:EXAMPLES]]
        mix = ", ".join(f"{n:,} in {u}" for u, n in counts.most_common())
        report.append(_entry(col, "unit_in_text", "converted" if minority else "read_as_numbers", minority or len(vals),
                             f"every value in '{col}' carries a unit ({mix}); read as numbers in {base}", ex))


# ── 2. percent scale: 5 among 0.05, 0.1, 0.15 ───────────────────────────────────────────────────────────────────
_PCT_NAME = re.compile(r"(discount|pct|percent|percentage|rate|ratio|share|margin|tax|interest|growth|yield|commission)")


def _fix_percent_scale(df, out, report):
    for col in df.columns:
        s = out[col]
        if not _is_number(s) or not _PCT_NAME.search(str(col).lower()):
            continue
        v = s.dropna().astype(float)
        if len(v) < MIN_VALUES or (v < 0).any():
            continue
        # the reverse mistake: a column of whole percentages (5, 10, 15) with a few fractions (0.05, 0.1) among them. Every
        # fraction x 100 must already occur as a value of the column, or it may be a genuine small percentage (0.5%)
        frac = v[(v > 0) & (v < 1)]
        whole = v[v >= 1]
        if (not frac.empty and len(frac) / len(v) <= MAX_SHARE and len(whole) / len(v) >= 0.6 and whole.max() <= 100
                and (frac * 100).round(6).isin(set(whole.round(6))).mean() >= 0.9):
            before = out[col].copy().astype(float)
            hit = before.index[(before > 0) & (before < 1)]
            out[col] = before.where(~before.index.isin(hit), (before * 100).round(6))
            ex = [{"row": int(i), "before": float(before[i]), "after": float(out[col][i])} for i in hit[:EXAMPLES]]
            report.append(_entry(col, "percent_scale", "rescaled", len(hit),
                                 f"{len(whole):,} of {len(v):,} values in '{col}' are whole percentages; {len(hit):,} are fractions "
                                 f"between 0 and 1 whose x100 value already occurs in the column, multiplied by 100", ex))
            continue
        over = v[v > 1]
        if over.empty or len(over) / len(v) > MAX_SHARE or over.max() > 100 or (v[v <= 1].size / len(v)) < 0.75:
            continue
        before = out[col].copy().astype(float)
        out[col] = before.where(~(before > 1), before / 100)
        ex = [{"row": int(i), "before": float(before[i]), "after": float(out[col][i])} for i in over.index[:EXAMPLES]]
        report.append(_entry(col, "percent_scale", "rescaled", len(over),
                             f"{len(v) - len(over):,} of {len(v):,} values in '{col}' are fractions between 0 and 1; "
                             f"{len(over):,} are whole percentages (between {over.min():g} and {over.max():g}), divided by 100", ex))


# ── 3. magnitude outliers: 37773.00 among values like 377.73 ────────────────────────────────────────────────────
def _fix_magnitude(df, out, report):
    for col in df.columns:
        s = out[col]
        if not _is_number(s) or pd.api.types.is_integer_dtype(s):
            continue
        v = s.dropna().astype(float)
        if len(v) < 30 or (v <= 0).any():
            continue
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        far = v[v > q3 + 10 * iqr]
        if far.empty or len(far) / len(v) > 0.05:
            continue
        is_int = lambda x: abs(x - round(x)) < 1e-9
        rest = v.drop(far.index)
        if rest.map(is_int).mean() > 0.20:                  # integers are normal here, so "an integer" proves nothing
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = out[col].copy().astype(float)
        fixed = {}
        for i, x in far.items():
            for k in (100, 1000):
                if is_int(x) and lo <= x / k <= hi:
                    fixed[i] = x / k
                    break
        if not fixed:
            continue
        out[col] = before
        for i, x in fixed.items():
            out.loc[i, col] = x
        ex = [{"row": int(i), "before": float(before[i]), "after": float(x)} for i, x in list(fixed.items())[:EXAMPLES]]
        report.append(_entry(col, "magnitude_outlier", "rescaled", len(fixed),
                             f"{len(fixed):,} value(s) in '{col}' are whole numbers far above the rest, and exactly 100x or 1000x "
                             f"a normal value (a decimal shift); the other values keep their cents", ex))


# ── 4. impossible negatives ─────────────────────────────────────────────────────────────────────────────────────
_NONNEG_NAME = re.compile(r"(quantit|qty|count|units?_|_units|stock|on_?hand|inventory|age$|salary|weight|height|"
                          r"length|price|cost|duration|distance|days|hours|headcount)")


def _fix_impossible(df, out, report):
    for col in df.columns:
        s = out[col]
        if not _is_number(s) or not _NONNEG_NAME.search("_".join(_tokens(col)) if _tokens(col) else str(col).lower()):
            continue
        v = s.dropna().astype(float)
        neg = v[v < 0]
        if len(v) < MIN_VALUES or neg.empty or len(neg) / len(v) > 0.05:
            continue
        before = out[col].copy()
        out[col] = before.astype(float).where(~(before.astype(float) < 0))
        ex = [{"row": int(i), "before": float(before[i]), "after": None} for i in neg.index[:EXAMPLES]]
        report.append(_entry(col, "impossible_negative", "blanked", len(neg),
                             f"'{col}' cannot be negative; {len(neg):,} negative value(s) cannot be recovered, so they are left blank", ex))


# ── 5. dates in the future ──────────────────────────────────────────────────────────────────────────────────────
_FUTURE_OK = re.compile(r"(due|expir|renew|eta|deadline|schedul|planned|target|valid|until|next|forecast|delivery|follow)")


def _fix_future_dates(df, out, report, today):
    limit = pd.Timestamp(today) + pd.Timedelta(days=1)
    for col in df.columns:
        s = out[col]
        if not pd.api.types.is_datetime64_any_dtype(s) or _FUTURE_OK.search(str(col).lower()):
            continue
        v = s.dropna()
        fut = v[v > limit]
        if len(v) < MIN_VALUES or fut.empty or len(fut) / len(v) > 0.05:
            continue
        # a typo makes a few scattered future dates; a dataset whose calendar simply runs past today (a plan, a forecast, a
        # synthetic year) has a dense run of DIFFERENT future days. 10 or more distinct days filling at least 40% of the
        # span they cover is a calendar, not typos: abstain.
        days = fut.dt.normalize().drop_duplicates()
        span = (days.max() - days.min()).days + 1
        if len(days) >= 10 and len(days) >= 0.4 * span:
            continue
        before = s.copy()
        out[col] = s.where(~(s > limit))
        ex = [{"row": int(i), "before": str(before[i].date()), "after": None} for i in fut.index[:EXAMPLES]]
        report.append(_entry(col, "future_date", "blanked", len(fut),
                             f"{len(fut):,} date(s) in '{col}' are after today ({pd.Timestamp(today).date()}) and cannot be "
                             f"recovered, so they are left blank", ex))


# ── 6. an end date before its start date ────────────────────────────────────────────────────────────────────────
_START = {"start", "begin", "from", "order", "ordered", "created", "opened", "booked", "purchase", "purchased", "hired", "admitted", "issued", "placed"}
_END = {"end", "finish", "until", "ship", "shipped", "shipping", "delivery", "delivered", "closed", "completed", "resolved", "returned", "discharged", "fulfilled"}


def _fix_end_before_start(df, out, report):
    dcols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
    stems = collections.defaultdict(dict)
    for c in dcols:
        toks = _tokens(c)
        for role, words in (("start", _START), ("end", _END)):
            hit = [i for i, t in enumerate(toks) if t in words]
            if len(hit) == 1:
                stems[tuple(toks[:hit[0]] + ["#"] + toks[hit[0] + 1:])][role] = c
    for stem, pair in stems.items():
        if len(pair) != 2:
            continue
        s_col, e_col = pair["start"], pair["end"]
        both = out[s_col].notna() & out[e_col].notna()
        bad = both & (out[e_col] < out[s_col])
        # a data-entry error touches a small share of the TABLE; a majority of comparable pairs violating means these are
        # probably not a start and an end at all (or the columns are reversed), so do nothing
        if both.sum() < MIN_VALUES or not bad.any() or bad.sum() / both.sum() > 0.5 or bad.sum() / len(out) > 0.10:
            continue
        before = out[e_col].copy()
        out[e_col] = before.where(~bad)
        ex = [{"row": int(i), "before": f"{before[i].date()} (start {out[s_col][i].date()})", "after": None} for i in bad[bad].index[:EXAMPLES]]
        report.append(_entry(e_col, "end_before_start", "blanked", int(bad.sum()),
                             f"'{e_col}' is earlier than '{s_col}' in {int(bad.sum()):,} row(s); which date is wrong cannot be "
                             f"known, so the end date is left blank", ex))


# ── 7. mixed date formats ───────────────────────────────────────────────────────────────────────────────────────
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_ISO = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T].*)?$")
_DMY_NAME = re.compile(r"^(\d{1,2})[- ]([A-Za-z]{3,9})[- ,]+(\d{4})$")
_MDY_NAME = re.compile(r"^([A-Za-z]{3,9})\.? (\d{1,2}),? (\d{4})$")
_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_DOT = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")


def _ts(y, m, d):
    try:
        return pd.Timestamp(int(y), int(m), int(d))
    except (ValueError, OverflowError):
        return None


def _parse_one(t: str):
    """-> (kind, timestamp or None, slash parts). kind: iso | named | dot | slash | bad."""
    t = t.strip()
    if (m := _ISO.match(t)):
        ts = _ts(*m.groups())
        return ("iso", ts, None) if ts is not None else ("bad", None, None)
    if (m := _DMY_NAME.match(t)) and m.group(2)[:3].lower() in _MONTHS:
        ts = _ts(m.group(3), _MONTHS[m.group(2)[:3].lower()], m.group(1))
        return ("named", ts, None) if ts is not None else ("bad", None, None)
    if (m := _MDY_NAME.match(t)) and m.group(1)[:3].lower() in _MONTHS:
        ts = _ts(m.group(3), _MONTHS[m.group(1)[:3].lower()], m.group(2))
        return ("named", ts, None) if ts is not None else ("bad", None, None)
    if (m := _DOT.match(t)):
        ts = _ts(m.group(3), m.group(2), m.group(1))
        return ("dot", ts, None) if ts is not None else ("bad", None, None)
    if (m := _SLASH.match(t)):
        return "slash", None, (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return "bad", None, None


def _fix_mixed_dates(df, out, report):
    for col in df.columns:
        s = out[col]
        if not _is_text(s):
            continue
        vals = s.dropna().astype(str)
        if len(vals) < MIN_VALUES:
            continue
        parsed = vals.map(_parse_one)
        kinds = parsed.map(lambda x: x[0])
        if (kinds == "bad").any():                           # every filled cell must be a recognisable date
            continue
        families = set(kinds)
        slash = parsed[kinds == "slash"]
        day_first = sum(1 for _, _, (a, b, c) in slash if a > 12 and b <= 12)
        month_first = sum(1 for _, _, (a, b, c) in slash if b > 12 and a <= 12)
        if len(families) < 2 and not (day_first and month_first):
            continue                                         # one uniform format: the cleaner's job, not a mix
        order = "day" if day_first and not month_first else "month" if month_first and not day_first else None
        result, ambiguous = {}, 0
        for i, (kind, ts, parts) in parsed.items():
            if kind != "slash":
                result[i] = ts
                continue
            a, b, y = parts
            if a > 12 and b <= 12:
                result[i] = _ts(y, b, a)
            elif b > 12 and a <= 12:
                result[i] = _ts(y, a, b)
            elif a <= 12 and b <= 12 and order == "day":
                result[i] = _ts(y, b, a)
            elif a <= 12 and b <= 12 and order == "month":
                result[i] = _ts(y, a, b)
            else:
                result[i] = None                              # ambiguous and no order is proven: do not guess
                ambiguous += 1
        if sum(v is None for v in result.values()) > ambiguous:
            continue                                         # an impossible date (31 Feb): not a clean mix, abstain
        before = out[col].copy()
        new = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for i, ts in result.items():
            if ts is not None:
                new[i] = ts
        out[col] = new
        changed = [i for i in vals.index if kinds[i] != "iso"]
        ex = [{"row": int(i), "before": _short(before[i]), "after": str(new[i].date()) if pd.notna(new[i]) else None} for i in changed[:EXAMPLES]]
        fam = ", ".join(f"{n:,} {k}" for k, n in collections.Counter(kinds).most_common())
        note = f"; {ambiguous:,} ambiguous slash date(s) left blank" if ambiguous else ""
        report.append(_entry(col, "mixed_date_formats", "parsed", len(changed),
                             f"'{col}' mixes date formats ({fam}); each date was read with its own format, and a slash date was "
                             f"read day-first or month-first only where a part above 12 proves it{note}", ex))


# ── 8. swapped values between two text columns ──────────────────────────────────────────────────────────────────
def _fix_swapped(df, out, report):
    tcols = [c for c in df.columns if _is_text(out[c])][:12]
    for b in tcols:                                           # b: the small-domain column (region, status, category)
        vb = out[b].dropna().astype(str)
        if len(vb) < MIN_VALUES:
            continue
        counts = vb.value_counts()
        dom_b = set(counts[counts >= max(5, 0.02 * len(vb))].index)
        if not 2 <= len(dom_b) <= 20 or counts.size > 40:
            continue
        for a in tcols:                                       # a: the large-domain column (customer, item, name)
            if a == b or out[a].nunique() < 5 * len(dom_b):
                continue
            va, vb2 = out[a].astype(str), out[b].astype(str)
            a_has_dom = out[a].notna() & va.isin(dom_b)
            b_not_dom = out[b].notna() & ~vb2.isin(dom_b)
            hit = a_has_dom & b_not_dom
            n = int(hit.sum())
            if n == 0 or n / len(out) > 0.05 or (a_has_dom & ~hit).sum() > 0:   # a never legitimately holds a domain token
                continue
            before = out[[a, b]].copy()
            out.loc[hit, a], out.loc[hit, b] = before.loc[hit, b].values, before.loc[hit, a].values
            ex = [{"row": int(i), "before": f"{a}={_short(before[a][i])}, {b}={_short(before[b][i])}",
                   "after": f"{a}={_short(out[a][i])}, {b}={_short(out[b][i])}"} for i in hit[hit].index[:EXAMPLES]]
            report.append(_entry(f"{a} / {b}", "swapped_values", "swapped back", n,
                                 f"in {n:,} row(s) '{a}' holds a value that only ever belongs in '{b}' ({', '.join(sorted(dom_b)[:4])}) "
                                 f"and '{b}' holds a value that does not; the two cells were swapped back", ex))


# ── 9. near-duplicate rows ──────────────────────────────────────────────────────────────────────────────────────
_ID_WORDS = {"id", "code", "sku", "key", "number", "no", "num", "ref", "uuid", "guid", "isbn", "barcode"}


def _identifier_like(out: pd.DataFrame, c) -> bool:
    """Identifiers (ORD-000601 vs ORD-000194) are similar strings BY CONSTRUCTION, so fuzzy matching must never see them
    as one value spelled two ways: a unique-ish column, or one named like an id, is skipped."""
    if _ID_WORDS & set(_tokens(c)):
        return True
    # a column that is nearly all distinct AND mostly contains digits is a code (ORD-000601); a column of distinct NAMES
    # (no digits) is not, or a genuine staff table, where nearly every name is unique, could never be deduplicated
    return out[c].nunique() >= 0.9 * len(out) and out[c].astype(str).str.contains(r"\d").mean() > 0.5


def _fix_near_duplicate_rows(df, out, report):
    if len(out) < MIN_VALUES or len(out) > 200_000:
        return None
    drop, found = set(), []
    for c in [c for c in out.columns if _is_text(out[c]) and not _identifier_like(out, c)][:10]:
        rest = out.drop(columns=[c])
        if rest.shape[1] == 0:
            continue
        key = pd.util.hash_pandas_object(rest.astype(str), index=False)
        for _, idx in key.groupby(key).groups.items():
            idx = [i for i in idx if i not in drop]
            if len(idx) < 2:
                continue
            first = idx[0]
            for j in idx[1:]:
                a, b = str(out[c][first]), str(out[c][j])
                # a typo changes letters, never digits: two values with different digits are different values
                if a != b and re.sub(r"\D", "", a) == re.sub(r"\D", "", b) \
                        and difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.8:
                    drop.add(j)
                    found.append((c, first, j, a, b))
    if not drop or len(drop) / len(out) > MAX_SHARE:
        return None
    ex = [{"row": int(j), "before": f"{c}={_short(b)} (repeats row {int(f)} with {c}={_short(a)})", "after": "dropped"} for c, f, j, a, b in found[:EXAMPLES]]
    cols = sorted({c for c, *_ in found})
    report.append(_entry(", ".join(cols), "near_duplicate_rows", "dropped", len(drop),
                         f"{len(drop):,} row(s) repeat an earlier row in every column except a near-identical spelling in "
                         f"{', '.join(repr(c) for c in cols)}; the later copy was dropped", ex))
    return sorted(drop)


# ── the whole thing ─────────────────────────────────────────────────────────────────────────────────────────────
def validate_values(df: pd.DataFrame, today=None):
    """-> (repaired copy, report). report = {"entries": [...], "changed": cells changed, "rowsDropped": n}."""
    today = today or _dt.date.today()
    out = df.copy()
    entries: list = []
    try:                                  # first, so an impossible future date is not also read as "the other date is earlier"
        _fix_future_dates(df, out, entries, today)
    except Exception:  # noqa: BLE001
        pass
    for rule in (_fix_units, _fix_mixed_dates, _fix_percent_scale, _fix_magnitude, _fix_impossible, _fix_end_before_start, _fix_swapped):
        try:
            rule(df, out, entries)
        except Exception:  # noqa: BLE001  one rule failing must never break the rest, or the table
            continue
    rows_dropped = 0
    try:
        drop = _fix_near_duplicate_rows(df, out, entries)
        if drop:
            out = out.drop(index=drop)
            rows_dropped = len(drop)
    except Exception:  # noqa: BLE001
        pass
    changed = sum(e["count"] for e in entries if e["action"] != "dropped")
    return out, {"entries": entries, "changed": int(changed), "rowsDropped": int(rows_dropped)}
