"""
clean_adapt.py — turns "what a cleaner did to a file" into scoreable predictions.

A cleaner returns a table; the scorer wants PREDICTIONS in source coordinates:
    {line, col_index, column, action, value, reason}
with action one of fix | blank | quarantine | ignore | dropped_unreported
(plus ask / flag / keep for pipelines that can say so). This module is the
pure-Python core of that translation, so it can be tested without pandas.

Rules the adapter keeps, because the baseline is only fair if it does:
  * It reports what HAPPENED, not what the cleaner meant. A cell that changed
    is a "fix" with the new value; a cell that became empty is a "blank".
  * Coordinates are SOURCE coordinates. A cleaner that re-indexes after
    dropping rows is mapped back to the physical line numbers of the file
    before anything is scored, so a drop early in the file cannot shift every
    later prediction.
  * A removed row is "quarantine" only when the cleaner's own log accounts for
    it (the logged count covers the rows actually removed). Otherwise it is
    "dropped_unreported": a silent drop, shown as such in the baseline.
  * A blank line the file reader skipped is not a cleaner action: it is
    reported as "ignore" at its line.
  * Boolean literals compare case-insensitively ("true" == "True").
  * Numbers compare by value ("1234.50" == 1234.5) but leading zeros stay
    significant ("042425" != "42425"), so an ID mangled into an integer is a
    visible change, not an invisible one.
  * A cleaner that cannot say "ask" or "flag" declares that in `capabilities`;
    the scorer then reports those labels as "action not supported", not as misses.
"""

from __future__ import annotations

import csv
import io
import re

_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
_SIDE = re.compile(r".+_original(?:_\d+)?$")  # side columns some cleaners add next to a converted column
DROP_PATTERNS = [r"Removed (\d+) exact duplicate", r"Removed (\d+) repeated record", r"Removed (\d+) duplicate row",
                 r"Excluded (\d+) total/summary", r"Dropped (\d+) fully-empty row"]


def norm_col(name) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _canon(s: str) -> str:
    t = s.strip()
    if _NUM.match(t):
        return t.rstrip("0").rstrip(".") if "." in t else t
    return s


def same_value(a, b) -> bool:
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if a.strip().lower() in ("true", "false") and a.strip().lower() == b.strip().lower():
        return True  # a bool read as text and written back is a representation change, not an edit
    return a == b or _canon(a) == _canon(b)


def parse_raw(data: bytes) -> dict:
    """The file as a line-oriented reader sees it: header line, header fields,
    records [(line, fields)] (blank lines skipped, as pandas skips them), and
    the blank lines themselves."""
    rd = csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace")))
    header_line, header, records, blanks = None, [], [], []
    for row in rd:
        line = rd.line_num
        if not row:
            blanks.append(line)
        elif header_line is None:
            header_line, header = line, row
        else:
            records.append((line, row))
    return {"header_line": header_line, "header": header, "records": records, "blank_lines": blanks}


def reported_drops(log) -> tuple:
    """(number of removed rows the log accounts for, matching log lines)."""
    total, why = 0, []
    for entry in log:
        for pat in DROP_PATTERNS:
            m = re.search(pat, str(entry))
            if m:
                total += int(m.group(1))
                why.append(str(entry).strip()[:120])
                break
    return total, why


def _pred(line, ci, name, action, value=None, reason=""):
    return {"line": line, "col_index": ci, "column": name, "action": action, "value": value, "reason": reason}


def failure(case_id: str, error: str, capabilities) -> dict:
    return {"case_id": case_id, "status": "error", "error": str(error)[:300], "capabilities": sorted(capabilities),
            "structure": {}, "predictions": []}


def build_predictions(case_id, raw, in_cols, out_cols, out_rows, out_lines, log, capabilities) -> dict:
    """raw: parse_raw(); in_cols: the cleaner's input column names (one per raw
    header field, in order); out_cols/out_rows: its output (values already as
    text, "" for missing); out_lines: the SOURCE line of every output row."""
    if len(out_rows) != len(out_lines):
        raise ValueError("every output row needs exactly one source line")
    in_idx = {}
    for i, c in enumerate(in_cols):
        in_idx.setdefault(norm_col(c), i)
    col_map = [None if _SIDE.match(str(c)) else in_idx.get(norm_col(c)) for c in out_cols]
    header = raw["header"]
    fields_by_line = dict(raw["records"])
    preds = [_pred(ln, None, None, "ignore", reason="the reader skipped a blank line") for ln in raw["blank_lines"]]

    kept = set(out_lines)
    dropped = [ln for ln, _ in raw["records"] if ln not in kept]
    if dropped:
        n_rep, why = reported_drops(log)
        if n_rep >= len(dropped):
            action, reason = "quarantine", "; ".join(why)
        else:
            action, reason = "dropped_unreported", f"{len(dropped)} row(s) removed; the log accounts for {n_rep}"
        preds += [_pred(ln, None, None, action, reason=reason) for ln in dropped]

    for ln, row in zip(out_lines, out_rows):
        fields = fields_by_line[ln]
        for j, ci in enumerate(col_map):
            if ci is None:
                continue
            before = fields[ci] if ci < len(fields) else ""
            after = row[j]
            if not same_value(before, after):
                name = header[ci] if ci < len(header) else str(out_cols[j])
                preds.append(_pred(ln, ci + 1, name, "blank" if after == "" else "fix", None if after == "" else after))
    visible = [str(c) for c, ci in zip(out_cols, col_map) if ci is not None]
    return {"case_id": case_id, "status": "ok", "error": "", "capabilities": sorted(capabilities),
            "structure": {"header_line": raw["header_line"], "columns": visible}, "predictions": preds}
