"""
fill_missing.py — fill missing values from evidence inside the table itself. No model is involved, so nothing is invented.

  1. INFERRED  a blank is filled from other rows when ANOTHER COLUMN proves the value: for example every row of one
               customer has the same postal code, so a blank postal code on that customer's row takes it. The relationship
               must hold for at least 99% of the rows that have both values, must be shown by at least 5 repeated groups,
               and a cell is filled only when at least 3 known rows of its group agree, none disagree and at most 25% of the group is blank. Computed from the original table only,
               so one fill never becomes the "evidence" for another.
  2. DERIVED   a number another pair of columns determines exactly (amount = quantity x unit_price) is computed (fill_derived).
  2b. MEDIAN   only when the caller asks (median=True): numbers with no evidence get their column's median. OFF by default.
  3. TEXT      text with no evidence stays blank (reported as "no evidence"). Only if the caller asks for it
               (text_placeholder=True, the "Unknown" button) does it get the word "Unknown".
  never        dates, true/false columns, empty columns and identifier-like columns (every value different) are never
               filled: a guessed date or ID is a made-up fact.

Every fill is reported (column, how many cells, which rule, which value or which key column) and the input table is never
modified. Used by fill_service.run_fill behind an explicit button.
"""
import pandas as pd
from pandas.api import types as pdt

TEXT_FILL = "Unknown"
MIN_ROWS_FOR_ID_CHECK = 20       # fewer rows than this: "every value different" proves nothing
MIN_PURITY = 0.99                # share of rows with both values that must follow the key -> value relationship
MIN_REPEATED_GROUPS = 5          # the relationship must be shown by at least this many keys that occur twice or more
MAX_UNIQUE_SHARE = 0.5           # a key must repeat: more than half of its values being different is an identifier
MAX_KEYS_PER_COLUMN = 3
MAX_CANDIDATE_KEYS = 12
MAX_ROWS_FOR_INFERENCE = 300_000
MIN_AGREEING_ROWS = 3            # a group fills its blanks only when at least this many KNOWN rows agree
MAX_BLANK_SHARE = 0.25           # ...and only when at most this share of the group is blank
MIN_DERIVED_ROWS = 20            # a formula between columns must be shown by at least this many rows
MIN_DERIVED_PURITY = 0.97        # ...and hold on at least this share of them
DERIVED_REL_TOL = 0.005          # relative tolerance when checking a formula (rounding)


def _is_text(s: pd.Series) -> bool:
    return bool(pdt.is_object_dtype(s) or pdt.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype))


def _is_fillable_kind(s: pd.Series) -> bool:
    return not (pdt.is_bool_dtype(s) or pdt.is_datetime64_any_dtype(s) or pdt.is_timedelta64_dtype(s))


def _key_columns(df: pd.DataFrame, target: int) -> list:
    """Positions of columns that can act as a key for `target`: they repeat, and are not dates, flags or floats."""
    cands = []
    for j in range(len(df.columns)):
        if j == target:
            continue
        s = df.iloc[:, j]
        if pdt.is_bool_dtype(s) or pdt.is_datetime64_any_dtype(s) or pdt.is_float_dtype(s):
            continue
        present = int(s.notna().sum())
        nun = int(s.nunique(dropna=True))
        if nun < 2 or present == 0 or nun > MAX_UNIQUE_SHARE * present:
            continue
        cands.append((nun, j))
    return [j for _, j in sorted(cands, reverse=True)[:MAX_CANDIDATE_KEYS]]   # most specific repeating key first


def _evidence(key: pd.Series, target: pd.Series, missing: pd.Series):
    """For one key column: (mapping key->value for the keys whose known values all agree, rows it can fill, purity, repeated
    groups). `key`, `target`, `missing` are positional (RangeIndex)."""
    known = ~missing
    sub = pd.DataFrame({"k": key[known].astype(object), "v": target[known].astype(object)}).dropna()
    if sub.empty or sub["v"].nunique() < 2:          # one distinct known value proves nothing (vacuous evidence)
        return None
    counts = sub.groupby(["k", "v"], sort=False).size()
    per_key_total = counts.groupby(level=0).sum()
    per_key_max = counts.groupby(level=0).max()
    per_key_values = counts.groupby(level=0).size()
    purity = float(per_key_max.sum()) / float(per_key_total.sum())
    # blanks per key, to cap the blank share of each group
    blanks = key[missing].astype(object).dropna().value_counts()
    group_size = per_key_total.add(blanks.reindex(per_key_total.index).fillna(0), fill_value=0)
    ok = (per_key_values == 1) & (per_key_total >= MIN_AGREEING_ROWS) & \
         (blanks.reindex(per_key_total.index).fillna(0) <= MAX_BLANK_SHARE * group_size)
    unanimous = per_key_values[ok].index
    repeated = int(len(unanimous))
    first = counts.reset_index().drop_duplicates("k").set_index("k")["v"]
    mapping = first.loc[first.index.isin(unanimous)]
    return mapping, purity, repeated


def _infer(df: pd.DataFrame, out: pd.DataFrame, t: int):
    """Fill blanks of column `t` in `out` from evidence in `df`. -> list of report entries (one per key column used)."""
    target = df.iloc[:, t].reset_index(drop=True)
    missing = target.isna()
    remaining = out.iloc[:, t].reset_index(drop=True).isna()
    scored = []
    for j in _key_columns(df, t):
        key = df.iloc[:, j].reset_index(drop=True)
        ev = _evidence(key, target, missing)
        if ev is None:
            continue
        mapping, purity, repeated = ev
        if purity < MIN_PURITY or repeated < MIN_REPEATED_GROUPS or mapping.empty:
            continue
        can = int((remaining & key.astype(object).map(mapping).notna()).sum())
        if can:
            scored.append((can, purity, j, mapping))
    entries = []
    new = out.iloc[:, t].reset_index(drop=True)
    for can, purity, j, mapping in sorted(scored, key=lambda x: (-x[0], -x[1]))[:MAX_KEYS_PER_COLUMN]:
        key = df.iloc[:, j].reset_index(drop=True)
        mapped = key.astype(object).map(mapping)
        mask = new.isna() & mapped.notna()
        if not mask.any():
            continue
        filled = new.astype(object).where(~mask, mapped)
        try:
            filled = filled.astype(new.dtype)
        except (TypeError, ValueError):
            pass
        new = filled
        entries.append({"column": str(df.columns[t]), "count": int(mask.sum()), "method": "inferred",
                        "value": f"from the other rows with the same {df.columns[j]}", "from": str(df.columns[j])})
    if entries:
        new.index = out.index
        out.isetitem(t, new)
    return entries


def _numeric_cols(df: pd.DataFrame):
    return [j for j in range(len(df.columns)) if pdt.is_numeric_dtype(df.iloc[:, j]) and not pdt.is_bool_dtype(df.iloc[:, j])]


def _close(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).abs() <= DERIVED_REL_TOL * b.abs().clip(lower=1e-9) + 1e-9


def fill_derived(df: pd.DataFrame):
    """Fill numeric blanks that another pair of columns determines EXACTLY: target = A * B or target = A / B. The formula
    must hold (within rounding) on at least 97% of at least 20 rows that have all three values, and a blank is filled only
    where both other values are present. Computed from `df` only. -> (new_df, entries)."""
    out = df.copy()
    entries = []
    num = _numeric_cols(df)
    for t in num:
        miss = df.iloc[:, t].isna()
        if not miss.any() or miss.all():
            continue
        best = None
        for a in num:
            for b in num:
                if t in (a, b) or a == b:
                    continue
                for op in ("mul", "div"):
                    if op == "mul" and a > b:
                        continue
                    A, B, T = (df.iloc[:, c].astype(float) for c in (a, b, t))
                    have = A.notna() & B.notna() & T.notna()
                    if op == "div":
                        have &= B != 0
                    n = int(have.sum())
                    if n < MIN_DERIVED_ROWS:
                        continue
                    pred = A * B if op == "mul" else A / B.where(B != 0)
                    purity = float(_close(pred[have], T[have]).mean())
                    if purity < MIN_DERIVED_PURITY:
                        continue
                    can = miss & A.notna() & B.notna() & (B != 0 if op == "div" else True)
                    cnt = int(can.sum())
                    if cnt and (best is None or (purity, cnt) > (best[0], best[1])):
                        best = (purity, cnt, a, b, op, pred, can)
        if best is None:
            continue
        purity, cnt, a, b, op, pred, can = best
        new = out.iloc[:, t].astype(float).where(~can, pred.round(6))
        if pdt.is_integer_dtype(df.iloc[:, t]) or str(df.iloc[:, t].dtype) in ("Int64", "Int32"):
            r = new[can]
            if not ((r - r.round()).abs() < 1e-6).all():
                continue                                  # a whole-number column cannot hold a fraction: leave blank
            new = new.round().astype("Int64") if str(df.iloc[:, t].dtype).startswith("Int") else new.round()
        out.isetitem(t, new)
        sym = "x" if op == "mul" else "/"
        entries.append({"column": str(df.columns[t]), "count": cnt, "method": "derived",
                        "value": f"{df.columns[a]} {sym} {df.columns[b]} (holds on {purity:.0%} of the other rows)",
                        "from": f"{df.columns[a]}, {df.columns[b]}"})
    return out, entries


def fill_missing(df: pd.DataFrame, infer: bool = True, text_placeholder: bool = False, placeholder_columns=None,
                 median: bool = False):
    """-> (new_df, report). report = {"filled": [...], "skipped": [...], "total_filled": int, "remaining_text": int}.
    `df` is never modified. text_placeholder labels every text blank with no evidence "Unknown"; placeholder_columns does
    the same for just those column names (the user's per-column answer)."""
    only = {str(c) for c in (placeholder_columns or ())}
    out = df.copy()
    filled, skipped = [], []
    allow_infer = infer and len(df) <= MAX_ROWS_FOR_INFERENCE
    remaining_text = 0
    for i, col in enumerate(df.columns):               # by position: duplicate column names must not collide
        s0 = df.iloc[:, i]
        n0 = int(s0.isna().sum())
        if n0 == 0:
            continue
        name = str(col)

        def skip(reason, count, kind="never", numeric=False):
            entry = {"column": name, "count": count, "reason": reason, "kind": kind}
            if numeric:
                entry["numeric"] = True          # a number cannot be "Unknown": the app shows this as a note, not a question
            skipped.append(entry)

        if n0 == len(s0):
            skip("the whole column is empty", n0)
            continue
        if not _is_fillable_kind(s0):
            skip("dates and true/false values are never guessed", n0)
            continue
        present = s0.dropna()
        if _is_text(s0) and len(present) >= MIN_ROWS_FOR_ID_CHECK and present.nunique() == len(present):
            skip("every value is different (looks like an identifier)", n0)
            continue
        if not (pdt.is_numeric_dtype(s0) or _is_text(s0)):
            skip("this kind of column is not filled", n0)
            continue

        if allow_infer:
            filled.extend(_infer(df, out, i))
        s = out.iloc[:, i]
        left = int(s.isna().sum())
        if left == 0:
            continue
        if pdt.is_numeric_dtype(s):
            if not median:                               # a median is a guess: only when the caller asks for it
                skip("nothing in the file tells us the value", left, kind="no_evidence", numeric=True)
                continue
            med = s.median()
            try:
                new = s.fillna(med)
            except (TypeError, ValueError):             # a whole-number nullable column cannot hold a fractional median
                new = s.astype("Float64").fillna(med)
            out.isetitem(i, new)
            filled.append({"column": name, "count": left, "method": "median", "value": round(float(med), 4)})
        elif text_placeholder or name in only:
            if isinstance(s.dtype, pd.CategoricalDtype) and TEXT_FILL not in s.cat.categories:
                s = s.cat.add_categories([TEXT_FILL])
            out.isetitem(i, s.fillna(TEXT_FILL))
            filled.append({"column": name, "count": left, "method": "text", "value": TEXT_FILL})
        else:
            skip("nothing in the file tells us the value", left, kind="no_evidence")
            remaining_text += left
    return out, {"filled": filled, "skipped": skipped, "total_filled": int(sum(f["count"] for f in filled)),
                 "remaining_text": remaining_text}
