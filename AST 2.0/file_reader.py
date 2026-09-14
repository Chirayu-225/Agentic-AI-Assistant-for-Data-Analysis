"""
file_reader.py — single shared entry point for turning uploaded file bytes
into a DataFrame, for every backend service that needs one.

WHY THIS EXISTS
    analysis_service.py, auto_analyze_service.py, cleaning_agent.py, and
    profile_service.py each had their own private `_read_csv(csv_bytes)`
    — identical copies of the same ~6 lines. That duplication is how a
    format gap like Excel support stayed invisible for so long: fixing it
    in one copy wouldn't have fixed the other three, and it's easy to
    forget one exists at all. One shared function, one place to extend.

FORMAT SUPPORT
    .csv / .txt  → pd.read_csv (utf-8, falling back to latin1 — the same
                   fallback the old per-service functions already had,
                   for files with e.g. Windows-1252 characters)
    .xlsx / .xls → pd.read_excel. If the workbook has exactly one sheet,
                   that sheet is used. If it has several — common for
                   dashboard-style workbooks that mix small pivot/summary
                   tabs with the real source data — the sheet with the
                   most rows is chosen automatically (see
                   _pick_best_excel_sheet's docstring for why row count,
                   not sheet order, is the right signal), and which
                   sheet was picked is printed server-side so it's never
                   a silent guess.

    Dispatch is by the filename's extension, not by sniffing file
    content — the frontend already has the filename on hand for every
    call site, and sniffing binary-vs-text is a much larger surface
    (magic bytes, encoding guesses) for marginal benefit here.
"""

from __future__ import annotations

import io
import warnings

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".txt", ".xlsx", ".xls")


def _pick_best_excel_sheet(xls: pd.ExcelFile) -> tuple[str, pd.DataFrame]:
    """A workbook built as a dashboard — pivot/summary tabs alongside the
    real source tables — is common in the wild, not an edge case: a
    coffee-sales workbook that prompted this function existing had
    'Dashboard', 'Country Barchart', and 'Total Sales' as small
    pre-computed summary tabs sitting right alongside the real data in
    'orders' (795 rows), 'customers' (1001 rows), and 'products' (49
    rows). Defaulting to sheet 0 — the naive choice — silently grabbed
    whichever tab happened to be first, which was a 7-row dashboard
    summary, not the dataset anyone actually wants analyzed.

    Instead: read every sheet, keep the one with the most rows. A
    summary/pivot tab is, by definition, smaller than the data it
    summarises, so the largest table in a workbook is in practice
    almost always the real underlying dataset.
    """
    best_name, best_df = None, None
    for name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=name)
        except Exception:
            continue
        if df.empty:
            continue
        if best_df is None or len(df) > len(best_df):
            best_name, best_df = name, df

    if best_df is None:
        # Every sheet was empty or unreadable — fall back to sheet 0
        # as-is so the caller gets a clear pandas-level error rather
        # than this function swallowing it into a confusing None.
        return xls.sheet_names[0], pd.read_excel(xls, sheet_name=0)
    return best_name, best_df


def read_tabular_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Raises ValueError for an unrecognised extension, or whatever
    pandas/openpyxl raises for a file that doesn't parse — callers
    already wrap this in their own try/except and turn it into a
    {"success": False, "error": ...} response, so no extra handling
    is added here."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in ("xlsx", "xls"):
        # openpyxl warns about workbook features it doesn't support
        # (slicers, some chart extensions) that have nothing to do with
        # the data being read — real noise, not a signal, so it's
        # suppressed the same way the equally-harmless datetime-parse
        # warning is in cleaning.py.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            if len(xls.sheet_names) == 1:
                return pd.read_excel(xls, sheet_name=0)

            sheet_name, df = _pick_best_excel_sheet(xls)
            print(
                f"[file_reader] '{filename}' has {len(xls.sheet_names)} sheets "
                f"{xls.sheet_names} — auto-selected '{sheet_name}' ({len(df)} rows) "
                f"as the largest data sheet.",
                flush=True,
            )
            return df

    if ext in ("csv", "txt"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")

    raise ValueError(
        f"Unsupported file type '.{ext}' — expected one of "
        f"{', '.join(SUPPORTED_EXTENSIONS)}."
    )
