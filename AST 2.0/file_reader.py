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

# Above this, route CSV/TXT files through DuckDB's own CSV engine
# (sandbox_limits.run_large_csv_sql_sandboxed) instead of pd.read_csv —
# see that function's module-level docstring for why. Deliberately CSV/
# TXT only: there's no equivalent out-of-core engine for Excel here, so
# XLSX/XLS still goes through pandas/openpyxl regardless of size — that
# stays a real, separate limitation (see README) rather than being
# silently papered over by this threshold.
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50MB


def should_use_large_csv_path(file_bytes: bytes, filename: str) -> bool:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in ("csv", "txt") and len(file_bytes) > LARGE_FILE_THRESHOLD_BYTES


# Shared cap on how many rows of a query RESULT get serialized back to the
# frontend, in one place so analysis_service.py (single-dataset queries)
# and multi_table_service.py (joins) can't drift out of sync — same
# rationale as SUPPORTED_EXTENSIONS/LARGE_FILE_THRESHOLD_BYTES above:
# duplicating this constant is how one copy gets fixed and the other
# doesn't. Uncapped, a query whose answer happens to be a 200K-row table
# (a plausible result of a loose filter or a join with a wide fan-out,
# now MORE likely now that the large-CSV path allows bigger inputs in the
# first place) gets serialized to JSON in full and shipped to the
# browser — this is a response-size problem, not a sandbox memory
# problem, so it's enforced here, after the sandboxed query has already
# returned a real pandas DataFrame, not inside the sandbox itself.
# 5,000 rows comfortably covers what a human actually reads in a table
# UI; anything past that is a summarization/aggregation problem for the
# user's query to solve, not something worth shipping as raw rows.
MAX_RESULT_ROWS = 5000


def _pick_best_excel_sheet(file_bytes: bytes, xls: pd.ExcelFile) -> tuple[str, pd.DataFrame]:
    """A workbook built as a dashboard — pivot/summary tabs alongside the
    real source tables — is common in the wild, not an edge case: a
    coffee-sales workbook that prompted this function existing had
    'Dashboard', 'Country Barchart', and 'Total Sales' as small
    pre-computed summary tabs sitting right alongside the real data in
    'orders' (795 rows), 'customers' (1001 rows), and 'products' (49
    rows). Defaulting to sheet 0 — the naive choice — silently grabbed
    whichever tab happened to be first, which was a 7-row dashboard
    summary, not the dataset anyone actually wants analyzed.

    Instead: find the sheet with the most rows, then read ONLY that one
    with pandas. A summary/pivot tab is, by definition, smaller than the
    data it summarises, so the largest table in a workbook is in
    practice almost always the real underlying dataset.

    REVISED — this used to find "most rows" by calling pd.read_excel on
    EVERY sheet and comparing len(df), which means fully parsing every
    losing sheet too, in full, before throwing that work away. On a
    real 5-sheet, 18MB workbook (hotel_revenue_historical_full-2.xlsx —
    filed as a live bug: /api/profile timing out at 35s), that cost 42
    real seconds — reading three multi-decade sheets of ~22k/79k/41k
    rows each just to learn that the 79k-row one wins, before ever
    reading it "for real." Sheet ROW COUNTS are available essentially
    for free from the xlsx file's own dimension metadata via openpyxl in
    read_only mode (~15ms for this same 5-sheet workbook, measured) —
    pandas is only asked to actually parse the ONE sheet that wins.
    Dropped the total cost for that file from 42s to ~24s (just the
    winning sheet's own real parse time — openpyxl/pandas' per-cell
    overhead on 79k rows is the genuinely unavoidable part; see
    file_reader.py's module docstring and the Next.js client's
    /profile timeout, bumped alongside this fix for exactly this cost).

    KNOWN TRADE-OFF: ws.max_row is a fast proxy for "how many data rows
    does this sheet have," not a byte-identical guarantee — a sheet with
    stray formatting on an otherwise-empty trailing row can occasionally
    report a max_row slightly higher than the real last data row. This
    could, in a rare case, change WHICH sheet wins vs. the old exhaustive
    method. It can't affect the DATA returned for whichever sheet does
    win, since that sheet is still read for real with pd.read_excel —
    only the selection heuristic changed, not the correctness of reading
    the sheet once selected.
    """
    import openpyxl

    best_name, best_row_count = None, -1
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        try:
            for name in wb.sheetnames:
                try:
                    ws = wb[name]
                    # max_row includes the header row, so subtract 1 for
                    # an actual data-row count; a header-only or fully
                    # empty sheet lands at <= 0, treated as empty below,
                    # same as the old df.empty check.
                    row_count = (ws.max_row or 0) - 1
                except Exception:
                    continue
                if row_count > best_row_count:
                    best_name, best_row_count = name, row_count
        finally:
            wb.close()
    except Exception:
        # The fast dimension scan itself failed (a workbook openpyxl's
        # read-only mode chokes on for some reason pd.read_excel
        # wouldn't) — fall through to the sheet-0 fallback below rather
        # than a half-finished, possibly-wrong pick.
        best_name = None

    if best_name is None or best_row_count <= 0:
        # Every sheet looked empty (or the scan above failed entirely) —
        # fall back to sheet 0 as-is so the caller gets a clear pandas-
        # level error rather than this function silently guessing.
        return xls.sheet_names[0], pd.read_excel(xls, sheet_name=0)

    return best_name, pd.read_excel(xls, sheet_name=best_name)


def _fast_excel_engine():
    """'calamine' when python-calamine is installed and pandas can use it."""
    try:
        import python_calamine  # noqa: F401
        major, minor = (int(x) for x in pd.__version__.split(".")[:2])
        if (major, minor) >= (2, 2):
            return "calamine"
    except Exception:
        pass
    return None


def _read_excel_with_engine(file_bytes: bytes, filename: str, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine) if engine else pd.ExcelFile(io.BytesIO(file_bytes))
        if len(xls.sheet_names) == 1:
            return pd.read_excel(xls, sheet_name=0)

        sheet_name, df = _pick_best_excel_sheet(file_bytes, xls)
        print(
            f"[file_reader] '{filename}' has {len(xls.sheet_names)} sheets "
            f"{xls.sheet_names} — auto-selected '{sheet_name}' ({len(df)} rows) "
            f"as the largest data sheet ({engine or 'openpyxl'}).",
            flush=True,
        )
        return df


def read_tabular_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/TXT/Excel file into a DataFrame, reusing a
    recent parse of the exact same bytes when there is one (cache_layer.
    frame_cache). Parsing is the dominant cost of most endpoints — ~24s
    for the large Excel sheet in the live timeout bug — and the same file
    used to be re-parsed by /profile, /clean, /clean/agent, /auto-analyze
    and every follow-up. The cache returns a COPY, keyed on a digest of
    the bytes, so it can't go stale or leak mutations between requests.
    Errors are never cached.
    """
    import cache_layer
    return cache_layer.cached_parse(file_bytes, filename, _read_tabular_file_uncached)


def _read_tabular_file_uncached(file_bytes: bytes, filename: str) -> pd.DataFrame:
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
        # PERFORMANCE: openpyxl parses cell-by-cell in pure Python (~15-60s for
        # an 80k-row sheet). python-calamine (a Rust reader, pandas >= 2.2) does
        # the same job 5-20x faster. Used when installed; any failure falls back
        # to openpyxl, so behaviour is identical and nothing new is required.
        engines = [e for e in (_fast_excel_engine(), None) if e is not None] + [None]
        last_exc = None
        for engine in engines:
            try:
                return _read_excel_with_engine(file_bytes, filename, engine)
            except Exception as e:  # noqa: BLE001 — retried with the next engine
                last_exc = e
                if engine is not None:
                    print(f"[file_reader] {engine} failed ({e}); falling back to openpyxl", flush=True)
                    continue
                raise
        raise last_exc  # pragma: no cover

    if ext in ("csv", "txt"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")

    raise ValueError(
        f"Unsupported file type '.{ext}' — expected one of "
        f"{', '.join(SUPPORTED_EXTENSIONS)}."
    )
