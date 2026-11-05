"""
clean_adapter_stage0.py — the Stage 0 pipeline as the scorer sees it:
    clean_read.read()  ->  an all-text table  ->  the SAME cleaning.rule_based_clean the baseline used.

Only the reader changed, so any difference from the baseline is the reader's doing. Three adapter rules:
  * Legacy parity on missing values: pandas' read_csv treated a fixed list of strings ("N/A", "NaN", "null",
    "#N/A", ...) as missing before the cleaner ever saw the data. The reader keeps raw strings, so the same list
    is applied here, to keep the comparison fair. (It is a pandas default, not a Stage 0 decision; Stage 1
    typing will own it. It matters for the frozen split, whose sentinels are not all in that list.)
  * Identifier columns the reader hinted at (digit strings of one fixed width, some with a leading zero) are
    handed back as TEXT after cleaning, so a column like "06971" cannot be turned into 6971.0.
  * Everything the reader decided at row level is reported as a prediction at its source line:
    excluded rows -> quarantine, skipped blank lines -> ignore, doubtful rows -> flag.
"""

from __future__ import annotations

from clean_adapt import _pred, build_predictions, failure, norm_col
from clean_adapter_legacy import cell_text, clean_with_provenance
from clean_read import read

CAPABILITIES = {"fix", "blank", "quarantine", "ignore", "keep", "flag"}
PANDAS_NA = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>",
             "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}


def clean_table(data: bytes) -> dict:
    """The Stage 0 table for `data`: read -> raw-string frame -> rule_based_clean -> identifier columns restored.
    Returns {"res": reader result, "names", "rows", "out": cleaned DataFrame or None, "lines": source line of every
    cleaned row, "log", "error": "" or why there is no table}. Shared by the scorer adapter and the experiment."""
    import numpy as np
    import pandas as pd

    res = read(data)
    out = {"res": res, "names": [c["name"] for c in res["columns"]], "rows": res["rows"], "out": None, "lines": [],
           "log": [], "error": ""}
    if res["status"] != "ok":
        out["error"] = f"unreadable: {res['reason']}"
        return out
    rows, names = res["rows"], out["names"]
    if not rows:
        return out
    df = pd.DataFrame([[np.nan if v in PANDAS_NA else v for v in f] for _, f in rows], columns=names, dtype=object)
    try:
        cleaned, lines, log = clean_with_provenance(df, [ln for ln, _ in rows])
    except Exception as e:  # noqa: BLE001
        out["error"] = f"cleaner failed: {type(e).__name__}: {e}"
        return out
    by_line = dict(rows)
    for name in res["hints"]["identifier_columns"]:
        key, j = norm_col(name), names.index(name)
        if key in cleaned.columns:
            cleaned[key] = pd.Series([by_line[ln][j].strip() if by_line[ln][j].strip().isdigit() else v
                                      for ln, v in zip(lines, cleaned[key])], index=cleaned.index, dtype=object)
    out.update(out=cleaned, lines=lines, log=log)
    return out


def run(case_id: str, data: bytes, companions: dict) -> dict:
    t = clean_table(data)
    res, names, rows = t["res"], t["names"], t["rows"]
    gate = {"asked": bool(res["asks"]), "kinds": [a["kind"] for a in res["asks"]], "asks": res["asks"],
            "header": res["structure"].get("header", {})}
    if t["error"]:
        d = failure(case_id, t["error"], CAPABILITIES)
        d["gate"] = gate
        return d
    st = res["structure"]
    structure = {"header_line": st.get("header_line"), "columns": list(names)}
    raw = {"header_line": st.get("header_line"), "header": list(st.get("header_fields", []))[: len(names)] or names,
           "records": rows, "blank_lines": []}
    preds = [_pred(ln, None, None, "ignore", reason="blank line skipped by the reader") for ln in res["blanks"]]
    preds += [_pred(x["line"], None, None, "quarantine", reason=x["reason"]) for x in res["excluded"]]
    preds += [_pred(x["line"], None, None, "flag", reason=x["reason"]) for x in res["flags"] if x["reason"] == "possible_shift"]
    if t["out"] is None:
        return {"case_id": case_id, "status": "ok", "error": "", "capabilities": sorted(CAPABILITIES), "structure": structure,
                "predictions": preds, "gate": gate}
    out = t["out"]
    text_rows = [[cell_text(v) for v in r] for r in out.itertuples(index=False, name=None)]
    doc = build_predictions(case_id, raw, names, [str(c) for c in out.columns], text_rows, t["lines"], t["log"], CAPABILITIES)
    doc["predictions"] += preds
    doc["structure"] = structure
    doc["gate"] = gate
    return doc


def cleaned_csv_bytes(data: bytes):
    """-> (csv bytes of the Stage 0 table, "") or (None, why). Used by the experiment's third arm."""
    t = clean_table(data)
    if t["error"]:
        return None, t["error"]
    if t["out"] is None:
        return None, "the file has a header but no data rows"
    return t["out"].to_csv(index=False).encode("utf-8"), ""
