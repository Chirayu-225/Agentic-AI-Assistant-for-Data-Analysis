"""
cleaning.py — Deterministic rule-based data cleaning pipeline.

Runs before the LLM pass. Handles all known, pattern-matchable
data quality issues without any LLM involvement.
"""

import re
import warnings

import pandas as pd
import numpy as np

from missing_tokens import AMBIGUOUS_TOKENS, ambiguous_is_marker, check_policy, kept_note


_SENTINELS = {"nan", "none", "null", "n/a", "na", "--", "-", "", "nat"}
# What stays "always missing" when the NA policy is "safe" and a column's "na" / "none" are real values.
_UNAMBIGUOUS_SENTINELS = _SENTINELS - AMBIGUOUS_TOKENS

# Currency symbols and codes, as prefix or suffix ("₹1,200", "Rs. 900",
# "INR 450", "1200 USD").
_CURRENCY = re.compile(
    r"(?i)(?:[\$£€₹¥₩₺₽]|\brs\.?|\binr\b|\busd\b|\beur\b|\bgbp\b|\baud\b|\bcad\b|\bsgd\b|\baed\b)"
)
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")

_DATE_LIKE = (
    r"^\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"          # 2026-03-05, 2026/03/05
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"               # 05/03/2026
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}"        # 05 Mar 2026
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4})"       # Mar 5, 2026
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\s*$"          # optional time
)

_TOTAL_LABEL = r"^(grand\s+)?(total|subtotal|sum)s?\s*:?$"


def _parse_money(series: pd.Series) -> tuple[pd.Series, bool]:
    """'₹1,20,000.50' -> 120000.5, 'Rs. 900' -> 900, '(1,200)' -> -1200.
    Returns (parsed values, NaN where unparseable; whether any value
    carried a currency marker)."""
    s = series.astype(str).str.strip()
    had_currency = bool(s.str.contains(_CURRENCY, na=False).any())
    negative = s.str.match(r"^\(.*\)$", na=False) | s.str.match(r"^-", na=False)
    core = (
        s.str.replace(_CURRENCY, "", regex=True)
        .str.replace(r"[()\s,]", "", regex=True)
        .str.lstrip("-")
    )
    ok = core.str.match(_NUMBER, na=False)
    values = pd.to_numeric(core.where(ok), errors="coerce")
    values = values.where(~negative, -values)
    return values, had_currency


_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})(?:[ T].*)?\s*$")


def _parse_dates(series: pd.Series) -> tuple[pd.Series, str]:
    """Parses a date column with ONE convention for the whole column.

    Numeric dates like 05/03/2026 are ambiguous: 5 March (day-first, the
    Indian/UK convention) or 3 May (month-first, US). Earlier versions let
    pandas guess value by value, so one column mixed conventions —
    "05/03/2026" became 3 May while "25/03/2026" became 25 March — and
    every monthly figure on day-first data was wrong (found in review).

    Now the column's convention is decided once, from all its values: any
    first part > 12 means day-first; any second part > 12 means
    month-first; both at once means the column itself is inconsistent and
    it is NOT parsed. If nothing decides it, day-first is assumed (the
    target users' convention) and the returned note says so. ISO
    (2026-03-05) and named-month dates are unambiguous and parsed as-is.

    Returns (parsed dates, note for the cleaning log — '' if nothing to say)."""
    text = series.astype(str).str.strip()
    parts = text.str.extract(_NUMERIC_DATE)
    numeric = parts[0].notna()
    note = ""
    dayfirst = True
    if numeric.any():
        first = pd.to_numeric(parts.loc[numeric, 0])
        second = pd.to_numeric(parts.loc[numeric, 1])
        day_first_evidence = bool((first > 12).any())
        month_first_evidence = bool((second > 12).any())
        if day_first_evidence and month_first_evidence:
            return pd.Series(pd.NaT, index=series.index), (
                "mixes day-first and month-first dates (e.g. both 25/03 and 03/25) — "
                "left as text; fix the source so one convention is used")
        if month_first_evidence:
            dayfirst = False
            note = "read as month-first (MM/DD/YYYY): some values have a day part above 12"
        elif day_first_evidence:
            note = "read as day-first (DD/MM/YYYY): some values have a day above 12"
        else:
            note = ("every date is ambiguous (day and month both ≤ 12) — ASSUMED day-first "
                    "(DD/MM/YYYY); if the file uses MM/DD/YYYY, these dates are wrong")

    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if numeric.any():
        num = parts.loc[numeric]
        a = pd.to_numeric(num[0]); b = pd.to_numeric(num[1]); y = pd.to_numeric(num[2])
        y = y.where(y >= 100, y + 2000)
        day, month = (a, b) if dayfirst else (b, a)
        out.loc[numeric] = pd.to_datetime(
            pd.DataFrame({"year": y, "month": month, "day": day}), errors="coerce")
    rest = ~numeric
    if rest.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                out.loc[rest] = pd.to_datetime(text[rest], errors="coerce", format="mixed")
            except (TypeError, ValueError):  # pandas < 2.0 has no format="mixed"
                out.loc[rest] = pd.to_datetime(text[rest], errors="coerce")
    return out, note


def _currency_labels(series: pd.Series) -> list[str]:
    """Distinct currencies written in a column, e.g. ['$ (USD)', '₹ (INR)']."""
    text = series.dropna().astype(str)
    found = []
    for pattern, label in _CURRENCY_NAMES:
        if text.str.contains(pattern, na=False).any():
            found.append(label)
    return found


def _keep_unparsed(df_c: pd.DataFrame, col: str, original: pd.Series, parsed: pd.Series,
                   present: pd.Series, what: str, log: list) -> None:
    """A non-blank cell that couldn't be converted ('pending', 'TBD') is not
    the same as an empty cell. Keep its original text in a side column and
    name it in the log, instead of silently turning it into a blank."""
    lost = present & parsed.isna()
    n = int(lost.sum())
    if not n:
        return
    side = f"{col}_original"
    k = 2
    while side in df_c.columns:
        side = f"{col}_original_{k}"
        k += 1
    df_c[side] = original.where(lost)
    examples = ", ".join(repr(v) for v in original[lost].astype(str).unique()[:4])
    log.append(f"⚠ {n} value(s) in '{col}' aren't {what} ({examples}) — blank in '{col}', "
               f"original text kept in '{side}'")


def _category_key(value: str) -> str:
    """'Mid Market', 'mid-market', ' MID-MARKET ' -> 'midmarket'."""
    return re.sub(r"[^0-9a-z]", "", str(value).lower())


def _merge_category_variants(col: pd.Series):
    """Merges values of a low-cardinality text column that differ only in
    case, spacing or punctuation ("North" / "north " / "NORTH"), plus dotted
    abbreviations that unambiguously match one other value ("N." -> "North"
    only if no other value starts with "n"). Each group takes its most
    common spelling. Returns (new column, short description) or None."""
    present = col.dropna().astype(str)
    present = present[~present.str.lower().isin(_SENTINELS)]
    n_unique = present.nunique()
    if n_unique < 2:
        return None
    # A category has few distinct values once case/spacing/punctuation are
    # ignored. Judged on those normalised keys (not raw spellings), so a
    # small table full of variants still qualifies, while free text and
    # IDs (nearly every value distinct) never do.
    keys = present.map(_category_key)
    n_keys = keys.nunique()
    if n_keys > 60 or n_keys > max(10, 0.5 * len(present)):
        return None
    # Values that are really numbers or amounts ("$1,000", "₹1,000",
    # "1.000") reduce to the same digits — merging them would silently
    # rewrite one currency or number as another. Never treat those as
    # spelling variants (found in review: a mixed ₹/$ column had '$1,000'
    # rewritten as '₹1,000').
    if (keys.str.fullmatch(r"\d+", na=False)).mean() >= 0.5:
        return None

    counts = present.value_counts()
    groups: dict[str, list[str]] = {}
    for value in counts.index:  # most common first
        key = _category_key(value)
        if key:
            groups.setdefault(key, []).append(value)

    # Dotted abbreviations: "N." joins the single full key starting with "n".
    for key in list(groups):
        if key not in groups:
            continue
        spellings = groups[key]
        if len(key) <= 3 and all(re.fullmatch(r"\s*[A-Za-z]{1,3}\.\s*", v) for v in spellings):
            targets = [k for k in groups if k != key and k.startswith(key) and len(k) > len(key)]
            if len(targets) == 1:
                groups[targets[0]].extend(groups.pop(key))

    mapping, merged_examples = {}, []
    for key, spellings in groups.items():
        canonical = max(spellings, key=lambda v: counts.get(v, 0))
        variants = [v for v in spellings if v != canonical]
        for v in spellings:
            mapping[v] = canonical
        if variants:
            merged_examples.append(f"{' / '.join(repr(v) for v in variants[:3])} → {canonical!r}")
    if not merged_examples:
        return None
    new_col = col.map(lambda v: mapping.get(str(v), v) if pd.notna(v) else v)
    return new_col, "; ".join(merged_examples[:4]) + (" …" if len(merged_examples) > 4 else "")


def _rows_agree(group: pd.DataFrame) -> bool:
    """True when every column has at most one distinct non-blank value."""
    for c in group.columns:
        vals = group[c].dropna()
        vals = vals[~vals.astype(str).str.lower().isin(_SENTINELS)]
        if vals.astype(str).nunique() > 1:
            return False
    return True



_CURRENCY_NAMES = [
    (re.compile(r"₹|\brs\.?|\binr\b", re.I), "₹ (INR)"),
    (re.compile(r"\$|\busd\b", re.I), "$ (USD)"),
    (re.compile(r"€|\beur\b", re.I), "€ (EUR)"),
    (re.compile(r"£|\bgbp\b", re.I), "£ (GBP)"),
    (re.compile(r"¥", re.I), "¥"),
]


def detect_currencies(raw_df: pd.DataFrame) -> dict:
    """{normalised column name: currency label} for columns whose raw text
    carries currency markers — e.g. {'amount': '₹ (INR)'}. Read from the
    RAW frame, because cleaning strips the symbols. Used so insight text
    states the data's real currency instead of assuming dollars."""
    found = {}
    # A column NAME can state the currency even when cells are bare numbers
    # ("amount_inr", "price_usd", "cost_in_rupees").
    name_tokens = {"inr": "₹ (INR)", "rupee": "₹ (INR)", "rupees": "₹ (INR)", "rs": "₹ (INR)",
                   "usd": "$ (USD)", "dollar": "$ (USD)", "dollars": "$ (USD)",
                   "eur": "€ (EUR)", "euro": "€ (EUR)", "euros": "€ (EUR)", "gbp": "£ (GBP)"}
    for col in raw_df.columns:
        toks = re.split(r"[^a-z0-9]+", str(col).lower())
        for t in toks:
            if t in name_tokens:
                found[str(col).strip().lower().replace(" ", "_")] = name_tokens[t]
                break
    for col in raw_df.columns:
        if str(col).strip().lower().replace(" ", "_") in found:
            continue
        series = raw_df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        text = series.dropna().astype(str).head(2000)
        if text.empty:
            continue
        for pattern, label in _CURRENCY_NAMES:
            if text.str.contains(pattern, na=False).mean() >= 0.2:
                found[str(col).strip().lower().replace(" ", "_")] = label
                break
    return found


def _dup_warning(n: int) -> str:
    return (f"⚠ {n} row(s) are exact copies of another row — KEPT, because identical rows are often genuine repeat "
            f"transactions. The audit trail lists which rows, and lets you remove them if they are accidental copies.")


def rule_based_clean(raw_df: pd.DataFrame, drop_exact_duplicates: bool = False,
                     na_policy: str = "legacy") -> tuple[pd.DataFrame, list[str]]:
    """
    Apply deterministic cleaning steps in sequence.
    Returns (cleaned_df, log_of_actions).

    na_policy: "legacy" (the default) is the original behaviour, unchanged: the strings "na" / "none" (any case) are
    sentinel nulls everywhere. "safe" is what the app asks for (cache_layer.cached_rule_clean): they are blanked only
    in a column where they are a missing-value marker (the other values are numbers or dates), and kept, with a log
    line, where they are real values such as North America in a TERRITORY column. See missing_tokens.py. The
    unambiguous sentinels ("N/A", "null", "--", ...) are blanked under both policies.

    drop_exact_duplicates: identical rows are KEPT by default and only
    reported. In transaction data, two identical rows are often two real
    sales (same product, quantity and price on the same day); deleting
    them silently removed ~30% of a hotel-bookings file and cut revenue in
    review tests. Pass True only when the user has confirmed the copies are
    accidental. Rows that repeat a unique record ID (same invoice_id twice)
    are still merged either way — those are the same record by definition.
    """
    check_policy(na_policy)
    safe_na = na_policy == "safe"
    df_c = raw_df.copy()
    log  = []

    # Guard against a header-only or otherwise empty upload — every step
    # below assumes at least one row exists (starting with df_c.iloc[0]),
    # so skip cleaning entirely rather than crash with an IndexError.
    if len(df_c) == 0:
        log.append("⚠ Dataset has no data rows — skipped cleaning steps")
        return df_c, log

    # "safe" NA policy: what the reader already decided (a marker "NA" it blanked) goes first in the log, and the
    # per-column decision "are 'na' / 'none' markers here, or real values?" is made once per column, lazily.
    if safe_na:
        log.extend(raw_df.attrs.get("reader_notes", []))   # structure repairs made while reading (stage0_bridge.py)
        log.extend(raw_df.attrs.get("na_notes", []))
    _ambiguous_are_markers: dict = {}

    def _tokens_are_markers(col) -> bool:
        if not safe_na:
            return True                                   # legacy: "na" / "none" are always sentinels
        decided = _ambiguous_are_markers.get(col)
        if decided is None:
            s = df_c[col]
            lowered = s[s.notna()].astype(str).str.strip().str.lower()
            # no ambiguous token in the column: the answer does not matter, and the legacy sentinel set applies
            decided = (not lowered.isin(AMBIGUOUS_TOKENS).any()) or ambiguous_is_marker(s)
            _ambiguous_are_markers[col] = decided
        return decided

    # ── 1. Promote row 0 to header if columns look auto-generated ─────────────
    first_row = df_c.iloc[0]
    cols_are_integers = all(str(c).strip().lstrip("-").isdigit() for c in df_c.columns)
    cols_are_unnamed  = all(str(c).startswith("Unnamed") for c in df_c.columns)
    row0_looks_like_header = (
        first_row.apply(
            lambda v: isinstance(v, str)
            and not str(v).replace(".", "", 1).replace("-", "", 1).isdigit()
        ).sum()
        >= len(df_c.columns) * 0.6
    )
    row_offset = 0                       # raw row labels -> the cleaned table's rows (audit_ledger.py reads attrs["source_rows"])
    if (cols_are_integers or cols_are_unnamed) and row0_looks_like_header:
        df_c.columns = [str(v).strip() for v in df_c.iloc[0]]
        df_c = df_c.iloc[1:].reset_index(drop=True)
        row_offset = 1
        log.append("✓ Promoted row 0 to column headers")

    # ── 2. Normalise column names ──────────────────────────────────────────────
    original_cols = df_c.columns.tolist()
    df_c.columns = [str(c).strip().lower().replace(" ", "_") for c in df_c.columns]
    if df_c.columns.tolist() != original_cols:
        log.append("✓ Normalised column names (lowercase + underscores)")

    # ── 2b. Force identifier-like columns to string, before anything below
    #        gets a chance to treat them as a continuous quantity ────────────
    # A postal/zip code, phone number, or account number is a LABEL, not a
    # measurement — averaging or median-filling one is meaningless (a
    # median zip code is not a real place). This has to run BEFORE the
    # numeric-conversion pass in step 5 and the median-fill in step 8, or
    # pandas' own read_csv dtype inference (which already turned an
    # all-digit zip column into int64 before this function ever saw it)
    # silently routes it into "just another number" for the rest of the
    # pipeline. Once cast to string here, it falls through to the ordinary
    # categorical 'Unknown' fill in step 8 like any other text column.
    id_col_pattern = re.compile(
        r"(^|_)(id|zip(code)?|post(al)?code|pin\s*code|phone|fax|ssn|ein"
        r"|account_?(no|num|number)|acct_?(no|num)|sku|isbn|upc|barcode)(_|$)",
        re.IGNORECASE,
    )
    id_like_cols = {col for col in df_c.columns if id_col_pattern.search(col)}
    for col in id_like_cols:
        if pd.api.types.is_numeric_dtype(df_c[col]):
            df_c[col] = df_c[col].apply(lambda v: str(int(v)) if pd.notna(v) and float(v).is_integer() else (str(v) if pd.notna(v) else v))
            log.append(f"✓ Treated '{col}' as an identifier (not a quantity) — kept as text so it won't be averaged or median-filled")

    # ── 3. Drop fully empty columns and rows ───────────────────────────────────
    before_cols = df_c.shape[1]
    df_c.dropna(axis=1, how="all", inplace=True)
    if df_c.shape[1] < before_cols:
        log.append(f"✓ Dropped {before_cols - df_c.shape[1]} fully-empty column(s)")

    before_rows = df_c.shape[0]
    df_c.dropna(axis=0, how="all", inplace=True)
    if df_c.shape[0] < before_rows:
        log.append(f"✓ Dropped {before_rows - df_c.shape[0]} fully-empty row(s)")

    # ── 4. Exact duplicate rows: reported, removed only on request ────────────
    _dup_log_at = None
    n_dupes = int(df_c.duplicated().sum())
    if n_dupes and drop_exact_duplicates:
        df_c.drop_duplicates(inplace=True)
        log.append(f"✓ Removed {n_dupes} exact duplicate row(s), as requested")
    elif n_dupes:
        _dup_log_at = len(log)
        log.append(_dup_warning(n_dupes))

    # ── 5. Per-column type cleaning ───────────────────────────────────────────
    def _is_string_col(series):
        return (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

    for col in df_c.columns:
        if not _is_string_col(df_c[col]):
            continue
        if col in id_like_cols:
            # Already handled in step 2b — skip the numeric-family
            # detectors below entirely, or e.g. an all-digit zip code
            # would get re-flagged as a number and averaged.
            continue

        series = df_c[col].astype(str).str.strip()
        # pandas 3 keeps a real missing value as NaN under astype(str) instead
        # of the text "nan", so the sentinel check alone no longer excludes it.
        # A blank must not count as a present value, or a column with more than
        # 20% blanks could never reach the 80% parse threshold below.
        sentinels = _SENTINELS if _tokens_are_markers(col) else _UNAMBIGUOUS_SENTINELS
        present = df_c[col].notna() & ~series.str.lower().isin(sentinels)
        n_present = int(present.sum())
        if n_present == 0:
            continue

        # 5a. Percentages — kept in the unit the file used (5% -> 5), so an
        #     answer about "average discount percentage" reads 3.5, not
        #     0.035. The column name and log make the unit explicit.
        pct_mask = series.str.endswith("%", na=False)
        if pct_mask.sum() > n_present * 0.4:
            parsed = pd.to_numeric(series.str.replace("%", "", regex=False).str.strip(), errors="coerce")
            if parsed.notna().sum() >= pct_mask.sum() * 0.8:
                df_c[col] = parsed
                log.append(f"✓ Converted percentages in '{col}' to numbers in percent points (5% → 5)")
                continue

        # 5b. Money and formatted numbers in one pass: currency symbols and
        #     codes (₹, Rs., INR, $, USD...), thousands/lakh separators,
        #     and accounting negatives "(1,200.00)". A column converts when
        #     at least 80% of its non-blank values parse, so a column that
        #     MIXES "₹1,200", "Rs. 900" and "450" (common in Indian exports)
        #     is still recognised as money.
        parsed, had_currency = _parse_money(series)
        if parsed[present].notna().sum() >= n_present * 0.8:
            currencies = _currency_labels(series[present]) if had_currency else []
            if len(currencies) > 1:
                # ₹1,000 + $1,000 is not 2,000 of anything. Leave the column
                # as text so nothing can sum it, and say why.
                log.append(f"⚠ '{col}' mixes currencies ({', '.join(currencies)}) — left as text so "
                           f"different currencies are never added together; split or convert it first")
                continue
            original = df_c[col].copy()
            df_c[col] = parsed.where(present)
            log.append(f"✓ Parsed {'currency' if had_currency else 'numeric'} values in '{col}'")
            _keep_unparsed(df_c, col, original, parsed, present, "numbers", log)
            continue

        # 5c. Dates — mixed formats in one column ("2026-03-05",
        #     "05 Mar 2026", "2026/03/05") are normal in merged exports.
        #     Only attempted when most values LOOK like dates, so free text
        #     and codes are never coerced.
        looks_like_date = series[present].str.contains(_DATE_LIKE, regex=True, na=False)
        if looks_like_date.sum() >= n_present * 0.6:
            parsed_date, date_note = _parse_dates(series)
            if parsed_date[present].notna().sum() >= n_present * 0.8:
                original = df_c[col].copy()
                df_c[col] = parsed_date.where(present)
                log.append(f"✓ Parsed dates in '{col}'" + (f": {date_note}" if date_note else ""))
                _keep_unparsed(df_c, col, original, parsed_date, present, "valid dates", log)
                continue
            if date_note.startswith("mixes"):
                log.append(f"⚠ '{col}' {date_note}")
                continue

    # ── 6. Strip whitespace from remaining strings (preserve real NaN) ─────────
    for col in df_c.columns:
        if _is_string_col(df_c[col]):
            mask = df_c[col].notna()
            df_c.loc[mask, col] = df_c.loc[mask, col].astype(str).str.strip()

    # ── 7. Replace sentinel nulls ─────────────────────────────────────────────
    sentinel_strings = {"nan", "none", "null", "n/a", "na", "--", "-", ""}
    sentinel_numbers = [-999, -9999, 999999, 9999999]

    for col in df_c.columns:
        if _is_string_col(df_c[col]):
            lower = df_c[col].str.lower().str.strip()
            if _tokens_are_markers(col):
                mask = lower.isin(sentinel_strings)
            else:
                mask = lower.isin(sentinel_strings - AMBIGUOUS_TOKENS)
                tokens = df_c[col][lower.isin(AMBIGUOUS_TOKENS)].str.strip().value_counts()
                for token in sorted(tokens.index):
                    log.append(kept_note(col, token, int(tokens[token])))
            if mask.any():
                df_c.loc[mask, col] = np.nan
                log.append(f"✓ Replaced {mask.sum()} sentinel string null(s) in '{col}'")
        elif pd.api.types.is_numeric_dtype(df_c[col]):
            mask = df_c[col].isin(sentinel_numbers)
            if mask.any():
                df_c.loc[mask, col] = np.nan
                log.append(f"✓ Replaced {mask.sum()} sentinel numeric null(s) in '{col}'")

    # ── 8. Total / summary rows ───────────────────────────────────────────────
    # A "TOTAL" or "Grand Total" line at the bottom of an export is not a
    # record: left in, it doubles every sum and adds one to every count.
    # Detected by an exact label match in the first two columns.
    total_mask = pd.Series(False, index=df_c.index)
    for label_col in list(df_c.columns[:2]):
        labels = df_c[label_col].astype(str).str.strip().str.lower()
        total_mask |= labels.str.match(_TOTAL_LABEL, na=False)
    if total_mask.any():
        df_c = df_c[~total_mask]
        log.append(f"✓ Excluded {int(total_mask.sum())} total/summary row(s) so they aren't counted as data")

    # ── 9. Merge spelling variants of the same category ───────────────────────
    for col in df_c.columns:
        if col in id_like_cols or not _is_string_col(df_c[col]):
            continue
        merged = _merge_category_variants(df_c[col])
        if merged is not None:
            df_c[col], examples = merged
            log.append(f"✓ Merged spelling variants in '{col}': {examples}")

    # ── 10. Duplicates, again, now that values are normalised ─────────────────
    # Same rule as step 4: reported, removed only when requested.
    if drop_exact_duplicates:
        before_rows = df_c.shape[0]
        df_c = df_c.drop_duplicates()
        if df_c.shape[0] < before_rows:
            log.append(f"✓ Removed {before_rows - df_c.shape[0]} duplicate row(s) that only differed in formatting")
    else:
        n_after = int(df_c.duplicated().sum())
        if n_after > n_dupes:
            log.append(f"⚠ {n_after - n_dupes} more row(s) became identical after formatting was "
                       f"standardised — KEPT; check whether they are accidental copies")

    # Rows that repeat a unique record ID and agree on every value both of
    # them have (one copy may have a blank cell) are the same record
    # submitted twice — keep the most complete copy.
    for col in id_like_cols:
        if col not in df_c.columns:
            continue
        ids = df_c[col]
        # Must look like a per-record ID (mostly distinct). A shared key
        # such as employee_id repeats by design and is skipped; even if it
        # weren't, rows are only merged when they agree on every value.
        if ids.nunique(dropna=True) < 0.75 * ids.notna().sum():
            continue
        dup_ids = ids[ids.duplicated(keep=False) & ids.notna()].unique()
        drop_idx = []
        for rid in dup_ids:
            group = df_c[ids == rid]
            if _rows_agree(group):
                keep = group.notna().sum(axis=1).idxmax()
                drop_idx.extend(i for i in group.index if i != keep)
        if drop_idx:
            df_c = df_c.drop(index=drop_idx)
            log.append(f"✓ Removed {len(drop_idx)} repeated record(s) with the same '{col}' (kept the most complete copy)")

    # ── 11. Missing values are reported, never invented ───────────────────────
    # Earlier versions filled blank numbers with the column median and blank
    # text with 'Unknown' on EVERY query. That fabricated values the user
    # never wrote, changed averages and counts, and made questions like
    # "how many invoices have no payment date?" impossible to answer. Blanks
    # now stay blank: pandas sums, means and counts skip them, and the log
    # says exactly where they are. (The opt-in Data Quality Agent can still
    # propose an imputation the user explicitly accepts.)
    side_cols = {c for c in df_c.columns if re.fullmatch(r".+_original(_\d+)?", str(c))
                 and str(c).rsplit("_original", 1)[0] in df_c.columns}
    for col in df_c.columns:
        if col in side_cols:
            continue  # blank by design: only filled where a value couldn't be converted
        null_count = int(df_c[col].isnull().sum())
        if null_count:
            log.append(f"⚠ '{col}' has {null_count} missing value(s) — left blank (calculations skip them; nothing was filled in)")

    if _dup_log_at is not None:          # the warning counted copies before the same-ID rule ran: say only what is still there
        n_left = int(df_c.duplicated().sum())
        log[_dup_log_at] = _dup_warning(n_left) if n_left else None
        log = [l for l in log if l is not None]
    try:                                 # which raw row each remaining row came from (the audit trail needs it; nothing else does)
        _src = (pd.Index(df_c.index).to_numpy() + row_offset) if pd.api.types.is_integer_dtype(df_c.index) else None
    except Exception:  # noqa: BLE001
        _src = None
    df_c = df_c.reset_index(drop=True)
    if _src is not None and len(_src) == len(df_c):
        from row_trace import SourceRows
        df_c.attrs["source_rows"] = SourceRows(_src)
        if row_offset:
            df_c.attrs["header_promoted"] = True
    if not log:
        log.append("✓ No issues detected — dataset looks clean")
    return df_c, log
