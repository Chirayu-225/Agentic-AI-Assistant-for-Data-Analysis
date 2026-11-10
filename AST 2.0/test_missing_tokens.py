"""
test_missing_tokens.py — missing_tokens.py: the shared marker-or-value rule and the two-pass restore.

pandas turns the literal text "NA" and "None" into missing values when it reads a CSV. For a column such as
TERRITORY (EMEA / NA / APAC / Japan, NA = North America) that erases real values. missing_tokens decides, per column,
whether an ambiguous token ("na" / "none", any case) is a missing-value MARKER or a real VALUE, and puts real values
back after the normal read. Every other pandas NA token ("N/A", "null", "#N/A", ...) stays missing.

Works on pandas 2.x and 3.x. No LLM, no network. Run: python test_missing_tokens.py
"""
import csv
import io
import random
import sys

import pandas as pd

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def csv_text(columns: dict) -> str:
    """Columns of strings -> CSV text (csv module quoting, so embedded newlines and commas are quoted)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(columns))
    for row in zip(*columns.values()):
        w.writerow(row)
    return buf.getvalue()


def read_both(text: str):
    """-> (the normal first read, a reread function exactly like the reader's second pass)."""
    df = pd.read_csv(io.StringIO(text))

    def reread(cols):
        return pd.read_csv(io.StringIO(text), usecols=cols, dtype=str, na_filter=False)
    return df, reread


def values(df, col):
    """A column as plain Python values with None for missing, for exact comparisons."""
    return [None if pd.isna(v) else v for v in df[col].tolist()]


def main():
    import missing_tokens as mt

    # ---- constants -------------------------------------------------------------------------------------------------
    try:
        from pandas._libs.parsers import STR_NA_VALUES as installed
        check("PANDAS_NA_TOKENS equals the installed pandas' own default NA list",
              set(mt.PANDAS_NA_TOKENS) == set(installed), f"extra={set(mt.PANDAS_NA_TOKENS) - set(installed)} "
              f"missing={set(installed) - set(mt.PANDAS_NA_TOKENS)}")
    except ImportError:
        print("[SKIPPED] this pandas does not expose STR_NA_VALUES")
    check("the restorable tokens are exactly 'NA' and 'None', both pandas defaults",
          tuple(mt.RESTORABLE_TOKENS) == ("NA", "None") and set(mt.RESTORABLE_TOKENS) <= set(mt.PANDAS_NA_TOKENS) | {"None"})
    check("ambiguous tokens are 'na' and 'none' (compared case-insensitively)", set(mt.AMBIGUOUS_TOKENS) == {"na", "none"})
    check("the app policy is 'safe'", mt.APP_NA_POLICY == "safe")

    # ---- the shared rule: ambiguous_is_marker ------------------------------------------------------------------
    S = pd.Series
    check("numeric values around NA tokens: marker", mt.ambiguous_is_marker(S(["1", "2", "3.5", "NA", "-4", "NA"])) is True)
    check("real numbers (not strings) work too", mt.ambiguous_is_marker(S([1, 2.5, None, 4])) is True)
    check("dates (with digits): marker", mt.ambiguous_is_marker(S(["2024-01-05", "2024-02-11", "05/03/2024"] * 10)) is True)
    check("month names without digits: value, not marker", mt.ambiguous_is_marker(S(["May", "June", "March", "Sat", "Mon"] * 10)) is False)
    check("plain words: value", mt.ambiguous_is_marker(S(["EMEA", "APAC", "Japan"] * 10)) is False)
    check("empty remainder keeps the token", mt.ambiguous_is_marker(S([])) is False)
    check("only ambiguous tokens and blanks (empty remainder) keeps the token",
          mt.ambiguous_is_marker(S(["NA", "na", "None", "NONE", None, "N/A", "", "null"])) is False)
    check("exactly 95% numeric is enough", mt.ambiguous_is_marker(S(["1"] * 19 + ["x"])) is True)
    check("90% numeric is not", mt.ambiguous_is_marker(S(["1"] * 18 + ["x", "y"])) is False)
    check("94% numeric is not", mt.ambiguous_is_marker(S(["1"] * 94 + ["x"] * 6)) is False)
    check("alphanumeric codes are values", mt.ambiguous_is_marker(S(["A1", "B2", "C3", "D4"] * 10)) is False)
    check("other pandas NA tokens and sentinels are ignored in the remainder",
          mt.ambiguous_is_marker(S(["1", "2", "N/A", "null", "#N/A", "-", "--", "nan", "NULL", "3"])) is True)
    check("surrounding whitespace is trimmed", mt.ambiguous_is_marker(S([" 1 ", "2 ", " 3", "NA"])) is True)
    big = S([str(i) for i in range(4800)] + ["abc"] * 200)          # 96% numeric, more than the 2000 sample
    check("a large column is judged on a fixed-seed sample: same answer every time",
          mt.ambiguous_is_marker(big) is True and mt.ambiguous_is_marker(big) is True and mt.ambiguous_is_marker(big.sample(frac=1, random_state=1)) is True)
    before = big.copy()
    mt.ambiguous_is_marker(big)
    check("the input series is not modified", big.equals(before))

    # ---- candidate columns -----------------------------------------------------------------------------------------
    df = pd.DataFrame({
        "text_nan": ["a", None], "text_full": ["a", "b"], "num_nan": [1.0, None], "ints": [1, 2],
        "bools": [True, False], "dates_nat": pd.to_datetime(["2024-01-01", None]),
        "deltas_nat": pd.to_timedelta(["1 day", None]), "all_nan": [float("nan")] * 2,
    })
    check("candidates: text with missing values, plus an entirely-missing column; never numeric/bool/datetime/timedelta",
          mt.candidate_columns(df) == ["text_nan", "all_nan"], str(mt.candidate_columns(df)))

    # ---- restore: the cases -----------------------------------------------------------------------------------------
    def restore(columns):
        text = csv_text(columns)
        first, reread = read_both(text)
        return first, mt.restore_ambiguous_tokens(first, reread)

    rng = random.Random(0)

    # a territory-like column with 38% NA keeps them
    terr = ["NA"] * 38 + ["EMEA"] * 40 + ["APAC"] * 22
    rng.shuffle(terr)
    first, res = restore({"order": [str(i) for i in range(100)], "territory": terr})
    check("territory-like 38% NA: the first read really lost them", int(first["territory"].isna().sum()) == 38)
    check("territory-like 38% NA: all 38 NA come back as the text NA", values(res.df, "territory") == terr and res.fallback_reason is None, str(res.fallback_reason))
    check("territory-like: kept count is reported", res.kept.get(("territory", "NA")) == 38, str(res.kept))
    check("territory-like: other columns are untouched", res.df["order"].tolist() == list(range(100)))

    # EMEA 1407, NA 1074, APAC 221, Japan 121
    terr = ["EMEA"] * 1407 + ["NA"] * 1074 + ["APAC"] * 221 + ["Japan"] * 121
    rng.shuffle(terr)
    first, res = restore({"ORDERNUMBER": [str(10000 + i) for i in range(2823)], "TERRITORY": terr})
    check("EMEA/NA/APAC/Japan file: pandas' own read shows 1,074 missing", int(first["TERRITORY"].isna().sum()) == 1074)
    check("EMEA/NA/APAC/Japan file: all 1,074 NA are kept, nothing missing",
          int(res.df["TERRITORY"].isna().sum()) == 0 and int((res.df["TERRITORY"] == "NA").sum()) == 1074)

    # a country column with 3 NA rows (Namibia) keeps them
    country = ["USA", "France", "Spain", "Italy"] * 25 + ["NA"] * 3
    first, res = restore({"country": country})
    check("country column with 3 NA rows keeps them", values(res.df, "country") == country)

    # a numeric column with NA tokens is untouched: dtype and missing values identical to the first read
    num = [str(i) for i in range(50)] + ["NA"] * 5
    first, res = restore({"qty": num, "k": ["a"] * 55})
    check("numeric column with NA tokens: untouched (dtype and missing identical)",
          res.df["qty"].dtype == first["qty"].dtype and res.df["qty"].isna().sum() == 5
          and res.df["qty"].equals(first["qty"]) and pd.api.types.is_numeric_dtype(res.df["qty"]))

    # a postal column: 97% numeric strings, 3% alphanumeric, a few NA -> NA blanked, alphanumeric kept, text dtype
    postal = [f"{10000 + i}" for i in range(194)] + ["K1A 0B1", "SW1A 1AA", "M5V 2T6", "H3Z 2Y7", "V6B 4N9", "T2P 3M5"] + ["NA"] * 5
    rng.shuffle(postal)
    first, res = restore({"postal": postal})
    out = values(res.df, "postal")
    check("postal 97% numeric: the NA are blanked (marker)", out.count(None) == 5 and "NA" not in out)
    check("postal: alphanumeric values kept", all(v in out for v in ("K1A 0B1", "SW1A 1AA", "M5V 2T6")))
    check("postal: still a text column", not pd.api.types.is_numeric_dtype(res.df["postal"]))
    check("postal: markers are counted for the log", res.markers.get(("postal", "NA")) == 5, str(res.markers))

    # a date-strings column with NA blanks the NA
    dates = [f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(100)] + ["NA"] * 4
    first, res = restore({"when": dates})
    check("date strings with NA: the NA are blanked", int(res.df["when"].isna().sum()) == 4 and "NA" not in values(res.df, "when"))

    # a column of month names / weekday names keeps the NA
    months = ["May", "June", "March", "Sat", "Mon"] * 12 + ["NA"] * 10
    first, res = restore({"label": months})
    check("month/weekday names with NA: the NA are KEPT", values(res.df, "label") == months)

    # a status column with "None" at 25% keeps it
    status = ["active", "closed"] * 38 + ["None"] * 24
    status = status[:100]
    rng.shuffle(status)
    first, res = restore({"status": status})
    check("status with 'None' at 25%: kept", values(res.df, "status") == status, str(res.fallback_reason))

    # N/A, null, #N/A (and the rest of pandas' list) are ALWAYS missing, even in a column that keeps NA
    mixed = ["EMEA", "APAC"] * 20 + ["N/A", "null", "#N/A", "NULL", "n/a", "nan", "NaN", "<NA>", "-NaN", ""] + ["NA"] * 10
    first, res = restore({"region": mixed})
    out = values(res.df, "region")
    check("always-missing tokens stay missing (10 of them)", sum(v is None for v in out) == 10, str(sum(v is None for v in out)))
    check("... while NA in the same column is kept", out.count("NA") == 10)

    # an all-NA column keeps its token; an empty remainder (only ambiguous tokens) keeps them
    first, res = restore({"x": ["NA"] * 30, "k": ["a"] * 30})
    check("an all-NA column keeps the token", values(res.df, "x") == ["NA"] * 30, str(res.fallback_reason))
    amb = ["NA", "na", "None", "NONE", "NA", "none"] * 5
    first, res = restore({"y": amb})
    check("an empty remainder (only ambiguous tokens) keeps them all", values(res.df, "y") == amb)

    # leading-zero identifiers: a numeric-looking column is never touched; a text id column keeps NA as a value
    first, res = restore({"code": ["007", "012", "NA", "045"] * 5})
    check("leading-zero numeric-looking IDs: untouched", res.df["code"].equals(first["code"]))
    first, res = restore({"sku": ["A007", "B012", "NA", "C045"] * 5})
    check("text IDs: the NA is kept as a value", values(res.df, "sku") == ["A007", "B012", "NA", "C045"] * 5)

    # a quoted multi-line field, embedded commas and quotes
    note = ["line one\nline two, with comma", 'she said "hi"', "NA", "plain"] * 6
    first, res = restore({"note": note, "n": [str(i) for i in range(24)]})
    check("quoted multi-line fields: NA kept, text intact", values(res.df, "note") == note, str(res.fallback_reason))

    # ---- nothing to do ----------------------------------------------------------------------------------------------
    first, reread = read_both(csv_text({"a": ["x", "y"], "b": ["1", "2"]}))
    calls = []
    res = mt.restore_ambiguous_tokens(first, lambda cols: calls.append(cols))
    check("no column with missing values: no second read at all, same frame", res.df is first and not calls)

    # ---- fail closed ------------------------------------------------------------------------------------------------
    text = csv_text({"territory": ["NA", "EMEA", "NA", "APAC"] * 10, "n": [str(i) for i in range(40)]})
    first, real = read_both(text)

    def fails_closed(label, reread):
        res = mt.restore_ambiguous_tokens(first, reread)
        check(f"fail closed - {label}: the first-read frame is returned unchanged", res.df is first, str(res.fallback_reason))
        check(f"fail closed - {label}: the reason is recorded", isinstance(res.fallback_reason, str) and res.fallback_reason != "")

    fails_closed("fewer rows", lambda cols: real(cols).iloc[:-1])
    fails_closed("more rows", lambda cols: pd.concat([real(cols), real(cols).iloc[:1]], ignore_index=True))
    fails_closed("different column names", lambda cols: real(cols).rename(columns={"territory": "other"}))
    fails_closed("the reread raises", lambda cols: (_ for _ in ()).throw(RuntimeError("boom")))
    fails_closed("a missing cell holds a token that is not an NA token",
                 lambda cols: real(cols).assign(territory=lambda d: d["territory"].replace({"NA": "Narnia"})))
    fails_closed("shifted by one row (missing cells no longer line up)",
                 lambda cols: pd.concat([real(cols).iloc[1:], real(cols).iloc[:1]], ignore_index=True))

    alt = csv_text({"v": ["a", "NA", "b", "NA", "c", "NA", "d", "NA", "e", "NA", "f", "NA"]})
    first2, real2 = read_both(alt)
    shifted2 = lambda cols: pd.concat([real2(cols).iloc[2:], real2(cols).iloc[:2]], ignore_index=True)
    check("setup: the shifted reread still holds NA exactly where the first read had missing cells",
          all(str(shifted2(["v"])["v"].iloc[i]) == "NA" for i in range(len(first2)) if pd.isna(first2["v"].iloc[i])))
    res = mt.restore_ambiguous_tokens(first2, shifted2)
    check("the stronger check catches a shifted reread whose NA cells still line up (non-missing cells differ)",
          res.df is first2 and bool(res.fallback_reason), str(res.fallback_reason))

    # the input frame is never changed by a successful restore either
    first3, real3 = read_both(csv_text({"t": ["NA", "x"] * 10}))
    snapshot = first3.copy()
    res = mt.restore_ambiguous_tokens(first3, real3)
    check("a successful restore returns a new frame and leaves the input as it was",
          first3.equals(snapshot) and res.df is not first3 and values(res.df, "t") == ["NA", "x"] * 10)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
