"""scale_review.py - the "x1000" question.

A whole-number column (a quantity, a count) where a few values are exactly a thousand times what the rest of the
column looks like, e.g. most quantities are 1-20 and a handful are 5000, 12000. Nothing in the file proves whether
that is a typo (an extra three zeros, a unit slip) or a real bulk order, so nothing is changed on our own: the user is
asked, and the change (divide by 1000) is made only on a click.

A value is a suspect only when ALL hold:
  - the column is whole numbers, 30+ filled cells, and not an id/code/phone/zip-like column;
  - the value is a multiple of 1000 and >= 1000;
  - the value divided by 1000 lands inside the range of the other values (so the "fixed" value would look normal);
  - the value is more than 5x the largest ordinary value (so it really stands out);
  - the suspects are few (at most 5% of the filled cells).
No model is involved.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_FILLED = 30
MAX_SHARE = 0.05
FACTOR = 1000
_SKIP_NAME = re.compile(r"(^|_)(id|code|key|sku|zip|postal|phone|pin|number|no|num)$|^(id|zip|phone)", re.I)


def _suspect_mask(s: pd.Series) -> tuple[pd.Series, float | None]:
    """(mask of suspect cells, typical value) for one column; an all-False mask when the column does not qualify."""
    none = pd.Series(False, index=s.index)
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return none, None
    v = s.dropna()
    if len(v) < MIN_FILLED:
        return none, None
    vals = v.to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)) or np.any(vals != np.round(vals)) or vals.min() < 0:
        return none, None
    if v.nunique() / len(v) > 0.9:                      # nearly all different: an identifier, not a quantity
        return none, None
    cand = (v >= FACTOR) & (v % FACTOR == 0)
    n_cand = int(cand.sum())
    if n_cand == 0 or n_cand > MAX_SHARE * len(v):
        return none, None
    rest = v[~cand]
    if len(rest) < MIN_FILLED // 2:
        return none, None
    top = float(rest.max())
    ok = cand & ((v / FACTOR) <= top) & (v > 5 * top)
    if not ok.any():
        return none, None
    mask = none.copy()
    mask.loc[ok[ok].index] = True
    return mask, float(rest.median())


def find_scale_suspects(df: pd.DataFrame) -> list[dict]:
    """One entry per column with suspects: {column, count, typical, examples:[{row, value, suggested}]}. `row` is the
    row number as a spreadsheet shows it (header = row 1), taken from the table's original row labels."""
    out = []
    for col in df.columns:
        if _SKIP_NAME.search(str(col)):
            continue
        mask, typical = _suspect_mask(df[col])
        n = int(mask.sum())
        if not n:
            continue
        ex = []
        for idx in list(mask[mask].index)[:3]:
            val = float(df.at[idx, col]) if not isinstance(df.at[idx, col], pd.Series) else float(df.loc[idx, col].iloc[0])
            try:
                row = int(idx) + 2
            except (TypeError, ValueError):
                row = None
            ex.append({"row": row, "value": int(val), "suggested": int(val // FACTOR)})
        out.append({"column": str(col), "count": n, "typical": int(typical) if typical is not None else None,
                    "examples": ex})
    return out


def apply_scale_fix(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Divide the suspect values of the NAMED columns by 1000 (suspects are re-derived here from the table itself, never
    taken from the caller). Returns (new table, [{column, count}])."""
    new = df.copy()
    done = []
    for col in columns:
        if col not in new.columns:
            continue
        mask, _ = _suspect_mask(new[col])
        n = int(mask.sum())
        if not n:
            continue
        if pd.api.types.is_integer_dtype(new[col]) or str(new[col].dtype) in ("Int64", "Int32"):
            new.loc[mask, col] = (new.loc[mask, col] // FACTOR).astype(new[col].dtype)
        else:
            new.loc[mask, col] = new.loc[mask, col] / FACTOR
        done.append({"column": col, "count": n})
    return new, done
