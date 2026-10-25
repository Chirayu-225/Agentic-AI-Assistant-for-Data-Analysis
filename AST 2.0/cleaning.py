"""
cleaning.py — Deterministic rule-based data cleaning pipeline.

Runs before the LLM pass. Handles all known, pattern-matchable
data quality issues without any LLM involvement.
"""

import re
import warnings

import pandas as pd
import numpy as np


_SENTINELS = {"nan", "none", "null", "n/a", "na", "--", "-", "", "nat"}

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


def _parse_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):  # pandas < 2.0 has no format="mixed"
            return pd.to_datetime(series, errors="coerce")


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
    n_keys = present.map(_category_key).nunique()
    if n_keys > 60 or n_keys > max(10, 0.5 * len(present)):
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


def rule_based_clean(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply deterministic cleaning steps in sequence.
    Returns (cleaned_df, log_of_actions).
    """
    df_c = raw_df.copy()
    log  = []

    # Guard against a header-only or otherwise empty upload — every step
    # below assumes at least one row exists (starting with df_c.iloc[0]),
    # so skip cleaning entirely rather than crash with an IndexError.
    if len(df_c) == 0:
        log.append("⚠ Dataset has no data rows — skipped cleaning steps")
        return df_c, log

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
    if (cols_are_integers or cols_are_unnamed) and row0_looks_like_header:
        df_c.columns = [str(v).strip() for v in df_c.iloc[0]]
        df_c = df_c.iloc[1:].reset_index(drop=True)
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

    # ── 4. Drop exact duplicate rows ──────────────────────────────────────────
    before_rows = df_c.shape[0]
    df_c.drop_duplicates(inplace=True)
    if df_c.shape[0] < before_rows:
        log.append(f"✓ Removed {before_rows - df_c.shape[0]} duplicate row(s)")

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
        present = ~series.str.lower().isin(_SENTINELS)
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
            df_c[col] = parsed.where(present)
            log.append(f"✓ Parsed {'currency' if had_currency else 'numeric'} values in '{col}'")
            continue

        # 5c. Dates — mixed formats in one column ("2026-03-05",
        #     "05 Mar 2026", "2026/03/05") are normal in merged exports.
        #     Only attempted when most values LOOK like dates, so free text
        #     and codes are never coerced.
        looks_like_date = series[present].str.contains(_DATE_LIKE, regex=True, na=False)
        if looks_like_date.sum() >= n_present * 0.6:
            parsed_date = _parse_dates(series)
            if parsed_date[present].notna().sum() >= n_present * 0.8:
                df_c[col] = parsed_date.where(present)
                log.append(f"✓ Parsed dates in '{col}'")
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
            mask  = lower.isin(sentinel_strings)
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
    before_rows = df_c.shape[0]
    df_c = df_c.drop_duplicates()
    if df_c.shape[0] < before_rows:
        log.append(f"✓ Removed {before_rows - df_c.shape[0]} duplicate row(s) that only differed in formatting")

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
    for col in df_c.columns:
        null_count = int(df_c[col].isnull().sum())
        if null_count:
            log.append(f"⚠ '{col}' has {null_count} missing value(s) — left blank (calculations skip them; nothing was filled in)")

    df_c = df_c.reset_index(drop=True)
    if not log:
        log.append("✓ No issues detected — dataset looks clean")
    return df_c, log
