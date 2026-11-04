"""
clean_adapter_legacy.py — runs the EXISTING cleaner (file_reader + cleaning.rule_based_clean)
on a benchmark file and reports what it did as predictions (see clean_adapt.py).

Thin pandas glue only; the translation rules live in clean_adapt.py and are tested there.

What it can say: fix, blank, quarantine (a drop the cleaner's own log accounts for),
ignore (a blank line the reader skipped), keep. It CANNOT ask or flag at a location (its
warnings carry no coordinates), so it declares those unsupported.

Row provenance: rule_based_clean ends with reset_index(drop=True), which erases which
source line each surviving row came from. The input index is set to the physical line
numbers, and reset_index is wrapped for the duration of the call to read them back.
"""

from __future__ import annotations

from clean_adapt import build_predictions, failure, norm_col, parse_raw

CAPABILITIES = {"fix", "blank", "quarantine", "ignore", "keep"}


def cell_text(v) -> str:
    """A cleaned cell as text: missing -> "", bool -> true/false, a date at midnight -> its ISO day."""
    import numpy as np
    import pandas as pd

    if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat() if v == v.normalize() else v.isoformat(sep=" ")
    return str(v)


def clean_with_provenance(df, lines):
    """rule_based_clean on `df` (one row per entry of `lines`, the SOURCE line of each). Returns
    (cleaned frame, source line of every cleaned row, the cleaner's log). Raises on any failure.
    rule_based_clean ends with reset_index(drop=True), which erases provenance, so the index is set
    to the line numbers and reset_index is wrapped for the duration of the call to read them back."""
    import pandas as pd

    import cleaning

    df.index = pd.Index(list(lines))
    seen = []
    original = pd.DataFrame.reset_index

    def spy(self, *a, **k):
        if k.get("drop"):
            seen.append(list(self.index))
        return original(self, *a, **k)

    pd.DataFrame.reset_index = spy
    try:
        out, log = cleaning.rule_based_clean(df)
    finally:
        pd.DataFrame.reset_index = original
    if any("Promoted row 0" in str(x) for x in log):
        raise RuntimeError("the cleaner re-headed the table; source lines can no longer be tracked")
    out_lines = seen[-1] if seen else []
    if len(out) and len(out_lines) != len(out):
        raise RuntimeError("cannot recover source lines for the cleaned rows")
    return out, (out_lines if len(out) else []), log


def run(case_id: str, data: bytes, companions: dict) -> dict:
    import file_reader

    raw = parse_raw(data)
    try:
        df = file_reader._read_tabular_file_uncached(data, "data.csv")  # bypass the parse cache
    except Exception as e:  # noqa: BLE001 — a parse failure IS the result for this case
        return failure(case_id, f"reader failed: {type(e).__name__}: {e}", CAPABILITIES)
    if len(df) != len(raw["records"]):
        return failure(case_id, f"cannot align rows: reader gave {len(df)}, file has {len(raw['records'])}", CAPABILITIES)
    in_cols = [norm_col(c) for c in df.columns]
    try:
        out, lines, log = clean_with_provenance(df, [ln for ln, _ in raw["records"]])
    except Exception as e:  # noqa: BLE001
        return failure(case_id, f"cleaner failed: {type(e).__name__}: {e}", CAPABILITIES)
    rows = [[cell_text(v) for v in r] for r in out.itertuples(index=False, name=None)]
    return build_predictions(case_id, raw, in_cols, [str(c) for c in out.columns], rows, lines, log, CAPABILITIES)
