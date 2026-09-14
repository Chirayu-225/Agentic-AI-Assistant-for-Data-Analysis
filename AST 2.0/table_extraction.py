"""
table_extraction.py — detects and extracts tables from PDFs, converting
each into a real pandas DataFrame + CSV bytes.

WHY THIS EXISTS, AND WHY IT'S NOT PART OF RAG
    RAG (rag_engine.py) answers questions by retrieving TEXT chunks by
    semantic similarity and asking an LLM to read them. That's the wrong
    tool for "what was Q3 revenue in Europe" — the LLM would be reading
    a table's text representation and eyeballing numbers out of it,
    which is a worse hallucination risk than what the CSV pipeline
    already solved, not a fix for it.

    Instead: extract each table into an actual DataFrame here, hand it
    off as a real file (see index_document's table-export step), and
    let it go through the EXACT SAME analysis_service.py /
    auto_analyze_service.py / cleaning_agent.py pipeline already built
    and tested for CSV uploads — real sandboxed computation, not
    LLM-guessed numbers. The "no hallucination" guarantee for tabular
    data inside a PDF is inherited for free, not re-implemented.

    Document narrative text still goes through RAG exactly as before.
    This module only ever touches whatever pdfplumber recognises as a
    table.

NOISE FILTERING
    pdfplumber's table detector works off line/rule geometry and can
    false-positive on incidental layout structure (a 2-column list, a
    header/footer with aligned text) that isn't really tabular data.
    _is_real_table() requires: at least 2 data rows (beyond the header),
    at least 2 columns, and most cells actually populated — a sparse,
    mostly-empty grid is far more likely a layout artifact than a real
    table someone would want to analyze.
"""

from __future__ import annotations

import io

import pandas as pd


def _is_real_table(rows: list[list]) -> bool:
    if len(rows) < 3:  # header + at least 2 data rows
        return False
    col_count = len(rows[0]) if rows[0] else 0
    if col_count < 2:
        return False

    total_cells = sum(len(r) for r in rows)
    filled_cells = sum(1 for r in rows for cell in r if cell and str(cell).strip())
    if total_cells == 0 or filled_cells / total_cells < 0.6:
        return False
    return True


def _rows_to_dataframe(rows: list[list]) -> pd.DataFrame | None:
    header, *data_rows = rows
    header = [str(h).strip() if h else f"column_{i}" for i, h in enumerate(header)]
    # Deduplicate blank/repeated header names — pandas rejects duplicate
    # column labels outright, and a merged-cell PDF table often produces
    # exactly this.
    seen: dict[str, int] = {}
    deduped = []
    for h in header:
        if h in seen:
            seen[h] += 1
            deduped.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            deduped.append(h)

    try:
        df = pd.DataFrame(data_rows, columns=deduped)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def extract_tables_from_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Returns a list of, for each real table found:
        {
            "title": human-readable label ("Table 1 (page 2)"),
            "page": 1-based page number,
            "row_count": int,
            "col_count": int,
            "preview": first 5 rows as list[dict], for a quick look
                        without downloading the CSV,
            "csv_bytes": the full table as CSV bytes, ready to hand to
                          the exact same file_reader.py / analysis
                          pipeline a regular CSV upload goes through,
        }
    Never raises for a PDF with no tables — returns an empty list, since
    "no structured data in this document" is a completely normal,
    expected outcome, not an error.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    results = []
    table_num = 0
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                raw_tables = page.extract_tables()
            except Exception:
                continue

            for raw_table in raw_tables:
                if not raw_table or not _is_real_table(raw_table):
                    continue

                df = _rows_to_dataframe(raw_table)
                if df is None:
                    continue

                table_num += 1
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)

                results.append({
                    "title": f"Table {table_num} (page {page_index + 1})",
                    "page": page_index + 1,
                    "row_count": len(df),
                    "col_count": len(df.columns),
                    "preview": df.head(5).to_dict(orient="records"),
                    "csv_bytes": csv_buf.getvalue().encode("utf-8"),
                })

    return results
