"""
test_cleaning_pandas3.py — rule_based_clean on text columns that contain REAL
missing values, under pandas 3's str dtype.

The bug this locks in: pandas 3 keeps a missing value as NaN under
astype(str) instead of turning it into the text "nan", so the sentinel check
alone no longer excluded it from the "present" mask. A blank was then counted
as a present-but-unparseable value, and a currency / number / date column
with more than 20% blanks (1 of 4, 1 of 3) failed the 80%-parse threshold and
stayed text. It also produced a spurious "<col>_original" side column.

Also locks the wording of the month-first / day-first date notes (a second
number above 12 can only be a DAY, not a month).

No LLM, no sandbox, no network. Run: python test_cleaning_pandas3.py
"""
import sys

import pandas as pd

from cleaning import _parse_dates, rule_based_clean

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def clean(values):
    """Clean a one-value-column frame. A second column keeps the rows with a
    blank from being dropped as 'fully empty' (a single-column frame would
    hide the bug)."""
    df = pd.DataFrame({"k": [f"row{i}" for i in range(len(values))], "v": values})
    out, log = rule_based_clean(df)
    return out, log


def is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def is_date(s):
    return pd.api.types.is_datetime64_any_dtype(s)


def with_blank(kind_values, n, blank_at=1):
    """n values, the one at blank_at replaced by a real missing value."""
    vals = [kind_values(i) for i in range(n)]
    vals[blank_at] = None
    return vals


def main():
    print("=" * 70)
    print("PART 1 — the original case: $1200, $2500, <blank>, $1800")
    print("=" * 70)
    out, log = clean(["$1200", "$2500", None, "$1800"])
    check("sales column becomes numeric", is_numeric(out["v"]), str(out["v"].dtype))
    check("values are right and the blank stays blank",
          out["v"].iloc[0] == 1200 and out["v"].iloc[1] == 2500 and pd.isna(out["v"].iloc[2]) and out["v"].iloc[3] == 1800,
          str(out["v"].tolist()))
    check("no spurious 'v_original' side column for a plain blank", "v_original" not in out.columns, str(list(out.columns)))

    print("\n" + "=" * 70)
    print("PART 2 — currency / plain number / date columns, 1 blank out of 4, 3 and 10")
    print("=" * 70)
    kinds = {
        "currency": (lambda i: f"${100 + i * 10}.50", is_numeric),
        "plain number": (lambda i: f"{10 + i}.25", is_numeric),
        "date": (lambda i: f"2024-01-{10 + i:02d}", is_date),
    }
    for kind, (make, ok) in kinds.items():
        for n in (4, 3, 10):
            out, log = clean(with_blank(make, n))
            check(f"{kind}: 1 blank out of {n} is still converted", ok(out["v"]), str(out["v"].dtype))
            check(f"{kind}: 1 blank out of {n} — no '_original' column", "v_original" not in out.columns, str(list(out.columns)))
            check(f"{kind}: 1 blank out of {n} — the blank stays blank", int(out["v"].isna().sum()) == 1, str(out["v"].tolist()))
    # More than 20% blank, as long as >= 80% of the NON-blank values parse
    out, log = clean(["$1"] * 7 + [None] * 3)
    check("currency: 3 blanks out of 10 (30% blank) is still converted", is_numeric(out["v"]), str(out["v"].dtype))

    print("\n" + "=" * 70)
    print("PART 3 — text categories with blanks must stay text")
    print("=" * 70)
    out, log = clean(["north", "south", None, "north"])
    check("real text categories stay text", not is_numeric(out["v"]) and not is_date(out["v"]), str(out["v"].dtype))
    check("category values are unchanged and the blank stays blank",
          out["v"].iloc[0] == "north" and out["v"].iloc[1] == "south" and pd.isna(out["v"].iloc[2]) and out["v"].iloc[3] == "north",
          str(out["v"].tolist()))
    check("no '_original' column for text categories", "v_original" not in out.columns, str(list(out.columns)))

    print("\n" + "=" * 70)
    print("PART 4 — an explicit 'N/A' sentinel still works; a column with no blanks is unaffected")
    print("=" * 70)
    out, log = clean(["$1200", "$2500", "N/A", "$1800"])
    check("explicit 'N/A' is treated as blank, column converts", is_numeric(out["v"]) and int(out["v"].isna().sum()) == 1, str(out["v"].tolist()))
    check("explicit 'N/A' — no '_original' column", "v_original" not in out.columns, str(list(out.columns)))
    out, log = clean(["$1,234.50", "$2,000.00", "$15.25", "$99.99"])
    check("no blanks: '$1,234.50' column converts", is_numeric(out["v"]) and out["v"].iloc[0] == 1234.5, str(out["v"].tolist()))

    print("\n" + "=" * 70)
    print("PART 5 — a real unparseable value is still reported (not swallowed by the fix)")
    print("=" * 70)
    out, log = clean(["$10", "$20", "pending", "$30", "$40"])
    check("'pending' keeps its text in a side column", "v_original" in out.columns, str(list(out.columns)))
    check("column still converts (4 of 5 non-blank values parse)", is_numeric(out["v"]), str(out["v"].dtype))

    print("\n" + "=" * 70)
    print("PART 6 — wording of the date-convention notes")
    print("=" * 70)
    _, month_first_note = _parse_dates(pd.Series(["03/25/2026", "01/02/2026"]))
    check("month-first note says a DAY part above 12", "day part above 12" in month_first_note, month_first_note)
    check("month-first note no longer says 'month part'", "month part" not in month_first_note, month_first_note)
    check("month-first note still says it was read as month-first", "read as month-first" in month_first_note, month_first_note)
    _, day_first_note = _parse_dates(pd.Series(["25/03/2026", "01/02/2026"]))
    check("day-first note says a day above 12", "a day above 12" in day_first_note and "read as day-first" in day_first_note, day_first_note)
    check("day-first note does not mention a month part", "month part" not in day_first_note, day_first_note)

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
        sys.exit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
