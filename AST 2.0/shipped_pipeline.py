"""
shipped_pipeline.py — the app's deterministic pipeline as ONE function, so it can be measured (exp_shipped.py) and reused
by the data-quality agent:

    bytes -> the app's "safe" read (structure bridge + real NA values) -> the rule cleaner -> [the fill step]
          -> value_validators (repairs and blanks, all reported)

No model, no network. `fill=True` adds the agent's fill step (fill_missing.py) before the validators, so a blank the
validators create is never filled back in.
"""
from __future__ import annotations

import io

import pandas as pd


def shipped_table(data: bytes, filename: str = "data.csv", *, fill: bool = False, today=None):
    """-> (table, {"cleaning": [...], "fill": report or None, "validators": report})."""
    import cleaning
    import file_reader
    import value_validators

    df = file_reader.read_tabular_file(data, filename)    # THE app entry point: structure bridge + real NA values, exactly as served
    cleaned, log = cleaning.rule_based_clean(df, na_policy="safe")
    fill_report = None
    if fill:
        from fill_missing import fill_missing
        cleaned, fill_report = fill_missing(cleaned)
    fixed, vreport = value_validators.validate_values(cleaned, today=today)
    return fixed, {"cleaning": log, "fill": fill_report, "validators": vreport}


def table_csv_bytes(df: pd.DataFrame) -> bytes:
    """The table as CSV text (the form the experiment's solvers take). Side columns the cleaner keeps (<col>_original)
    are dropped; dates are written as ISO dates."""
    keep = [c for c in df.columns if not str(c).endswith("_original")]
    buf = io.StringIO()
    df[keep].to_csv(buf, index=False, date_format="%Y-%m-%d")
    return buf.getvalue().encode("utf-8")


def shipped_csv_bytes(data: bytes, *, fill: bool = False) -> bytes:
    table, _report = shipped_table(data, fill=fill)
    return table_csv_bytes(table)
