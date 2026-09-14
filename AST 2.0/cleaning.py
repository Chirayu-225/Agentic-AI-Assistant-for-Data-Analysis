"""
cleaning.py — Deterministic rule-based data cleaning pipeline.

Runs before the LLM pass. Handles all known, pattern-matchable
data quality issues without any LLM involvement.
"""

import re
import warnings

import pandas as pd
import numpy as np


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
    currency_re = r"[\$£€₹¥₩₺₽]"

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
            # would get re-flagged as "thousands separators" here and
            # silently converted right back to a number.
            continue

        series = df_c[col].astype(str).str.strip()

        # 5a. Parenthetical negatives: (1000) → -1000
        paren_mask = series.str.match(r"^\([\d,\.]+\)$", na=False)
        if paren_mask.any():
            series = series.str.replace(r"\((.*?)\)", r"-\1", regex=True)
            log.append(f"✓ Converted parenthetical negatives in '{col}'")

        # 5b. Currency
        currency_mask = series.str.contains(currency_re, regex=True, na=False)
        if currency_mask.sum() > len(series) * 0.4:
            cleaned = (
                series.str.replace(currency_re, "", regex=True)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().sum() > currency_mask.sum() * 0.5:
                df_c[col] = parsed
                log.append(f"✓ Parsed currency values in '{col}'")
            continue

        # 5c. Percentages
        pct_mask = series.str.endswith("%", na=False)
        if pct_mask.sum() > len(series) * 0.4:
            cleaned = series.str.replace("%", "", regex=False).str.strip()
            parsed  = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().sum() > pct_mask.sum() * 0.5:
                df_c[col] = parsed / 100.0
                log.append(f"✓ Converted percentage values in '{col}' (÷100)")
            continue

        # 5d. Thousands separators
        comma_mask = series.str.match(r"^-?[\d,]+\.?\d*$", na=False)
        if comma_mask.sum() > len(series) * 0.4:
            cleaned = series.str.replace(",", "", regex=False)
            parsed  = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().sum() > comma_mask.sum() * 0.5:
                df_c[col] = parsed
                log.append(f"✓ Removed thousands separators in '{col}'")
            continue

        # 5e. Datetime
        # Free-text columns (a book title, a URL, a publisher blurb with
        # a date buried inside it) will never pass the 0.6 threshold
        # below, but pandas still pays the cost of trying — dateutil's
        # per-element fallback is slow on long strings and prints a
        # UserWarning for every single column it's tried on, which was
        # drowning real server logs in noise on datasets with a lot of
        # text columns. A cheap pre-check (average length under 30 chars
        # — comfortably longer than any real date/datetime string) skips
        # columns that were never going to parse as dates anyway.
        avg_len = series.str.len().mean()
        if pd.notna(avg_len) and avg_len <= 30:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed_date = pd.to_datetime(series, errors="coerce")
            if parsed_date.notna().sum() > len(series) * 0.6:
                df_c[col] = parsed_date
                log.append(f"✓ Parsed datetime in '{col}'")
                continue

        # 5f. Plain numeric strings
        parsed = pd.to_numeric(series, errors="coerce")
        non_null = series.replace("nan", pd.NA).dropna()
        if len(non_null) > 0 and parsed.notna().sum() / len(non_null) > 0.7:
            df_c[col] = parsed
            log.append(f"✓ Converted '{col}' to numeric")

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

    # ── 8. Fill remaining nulls — numeric: median, categorical: 'Unknown' ─────
    # First, identify totals/summary rows via the first column (the
    # label/category column in essentially every real-world table this
    # app sees — "Region", "Department", "Office", etc.). This matters
    # because a table extracted from a PDF report routinely has a
    # "Total" row where some cells are DELIBERATELY blank (a grand-total
    # row often only has the sum column filled in, not a per-column
    # breakdown) — that blank isn't missing data to guess at, it's
    # structural. Median-filling those cells with an average of the
    # OTHER rows silently fabricates a plausible-looking but completely
    # fictional number for a line that's supposed to be a sum, which is
    # a worse failure than just leaving it blank: nothing signals it's
    # fake. (Found via a real extracted PDF table during testing — this
    # wasn't a hypothetical edge case.)
    is_total_row = pd.Series(False, index=df_c.index)
    if len(df_c.columns) > 0:
        label_col = df_c[df_c.columns[0]].astype(str).str.strip().str.lower()
        is_total_row = label_col.str.match(r"^(grand\s+)?(total|sum|subtotal)s?$", na=False)

    for col in df_c.columns:
        null_count = df_c[col].isnull().sum()
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df_c[col]):
            fillable_mask = df_c[col].isnull() & ~is_total_row
            total_row_nulls = int((df_c[col].isnull() & is_total_row).sum())

            if fillable_mask.any():
                median_val = df_c.loc[~is_total_row, col].median()
                df_c.loc[fillable_mask, col] = median_val
                log.append(f"✓ Filled {int(fillable_mask.sum())} null(s) in '{col}' with median ({median_val:.2f})")

            if total_row_nulls > 0:
                log.append(
                    f"⚠ Left {total_row_nulls} null(s) in '{col}' as-is in what looks like a "
                    f"totals/summary row — filling those would fabricate a number, not describe one"
                )

        elif pd.api.types.is_datetime64_any_dtype(df_c[col]):
            # Don't guess dates — leave as NaT, but log it clearly
            log.append(f"⚠ '{col}' has {null_count} missing date(s) — left as-is (no safe default)")

        else:
            fillable_mask = df_c[col].isnull() & ~is_total_row
            if fillable_mask.any():
                df_c.loc[fillable_mask, col] = df_c.loc[fillable_mask, col].fillna("Unknown")
                log.append(f"✓ Filled {int(fillable_mask.sum())} null(s) in '{col}' with 'Unknown'")

    df_c = df_c.reset_index(drop=True)
    if not log:
        log.append("✓ No issues detected — dataset looks clean")
    return df_c, log
