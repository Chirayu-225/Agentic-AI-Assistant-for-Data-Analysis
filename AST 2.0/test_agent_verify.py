"""
test_agent_verify.py — agent_verify.verify_fix: what a data-quality fix script may and may not do to the table.

Adversarial fixes that MUST be caught (blank a state, fill territory with "Unknown", rename Planes to Aircraft, drop a
row, reorder rows, round a float column, rename a column, "007" -> 7, lowercase a product-code column, a flag column
that reuses an existing name) and fixes that MUST pass (whitespace tidying, case normalisation in a low-cardinality
column, a new boolean flag column with nulls, "1" -> 1.0, a date reformatted to the same date). Plus a mixed fix, the
global tripwire (a simulated verifier bug), escaping, exact counts, speed and "inputs never mutated".

No LLM, no sandbox, no network. Run: python test_agent_verify.py
"""
import sys
import time

import numpy as np
import pandas as pd

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


STATES = ["CA", "NY", "TX", "FL", "WA", "OR", "NV", "AZ", "CO", "UT", "MA", "NJ", "PA", "OH", "MI"]
LINES = ["Planes", "Ships", "Trains", "Cars", "Trucks"]


def base(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "state": [STATES[i % len(STATES)] for i in range(n)],
        "territory": [None if i % 10 == 0 else ["EMEA", "NA", "APAC"][i % 3] for i in range(n)],
        "productline": [LINES[i % len(LINES)] for i in range(n)],
        "productcode": [f"S{10 + i // 7}_{1000 + i}" for i in range(n)],          # high cardinality
        "amount": np.round(rng.random(n) * 1000, 4),
        "customer_id": [f"{i:03d}" for i in range(n)],                              # identifiers with leading zeros
        "qty_text": [str(1 + i % 9) for i in range(n)],
        "when": [["05 Jan 2024", "17 Feb 2024", "03 Mar 2024"][i % 3] for i in range(n)],
    })


def violations_of(r, kind=None, column=None):
    return [v for v in r["violations"] if (kind is None or v["kind"] == kind) and (column is None or v["column"] == column)]


def main():
    import agent_verify as av
    verify = av.verify_fix

    before = base()
    snapshot = before.copy()

    # ── a fix that changes nothing ──────────────────────────────────────────────────────────────────────────────
    r = verify(before, before.copy())
    check("identical tables: ok, nothing found, nothing kept", r["ok"] and not r["violations"] and not r["kept_changes"]
          and not r["reverted_columns"] and not r["new_columns"] and not r["fault"] and not r["structural"])
    check("identical tables: repaired equals the original", r["repaired"].equals(before))
    check("inputs are never mutated", before.equals(snapshot))
    check("repaired is a new object, not the input", r["repaired"] is not before)

    # ── MUST BE CAUGHT ─────────────────────────────────────────────────────────────────────────────────────────
    after = before.copy(); after.loc[[3, 4, 9, 12, 20, 21, 22], "state"] = None
    r = verify(before, after)
    v = violations_of(r, "blanked", "state")
    check("blank a state: caught as 'blanked' with the exact count (7)", len(v) == 1 and v[0]["count"] == 7, str(r["violations"]))
    check("blank a state: the column is reverted, the rest is untouched", r["reverted_columns"] == ["state"]
          and r["repaired"]["state"].equals(before["state"]) and r["ok"] is True)
    check("blank a state: the example is a value that would have been lost", v[0]["example"] in STATES)

    after = before.copy(); after["territory"] = after["territory"].fillna("Unknown")
    r = verify(before, after)
    v = violations_of(r, "filled", "territory")
    check("fill territory with 'Unknown': caught as 'filled' (20)", len(v) == 1 and v[0]["count"] == 20 and v[0]["example"] == "Unknown", str(r["violations"]))
    check("fill territory: reverted, still missing in the result", r["repaired"]["territory"].isna().sum() == 20)

    after = before.copy(); after["productline"] = after["productline"].replace({"Planes": "Aircraft"})
    r = verify(before, after)
    v = violations_of(r, "changed", "productline")
    check("rename Planes to Aircraft: caught as 'changed' (40)", len(v) == 1 and v[0]["count"] == 40 and "Planes" in v[0]["example"], str(r["violations"]))
    check("rename Planes: reverted", r["repaired"]["productline"].equals(before["productline"]))

    after = before.drop(index=3)
    r = verify(before, after)
    check("drop a row: structural, the whole fix is discarded", r["structural"] and not r["ok"] and r["repaired"].equals(before)
          and violations_of(r, "structure"), str(r["violations"]))

    after = before.sort_values("amount").reset_index(drop=True)
    r = verify(before, after)
    check("reorder rows (index renumbered): structural", r["structural"] and not r["ok"] and r["repaired"].equals(before), str(r["violations"]))
    after = before.sort_values("amount")
    r = verify(before, after)
    check("reorder rows (index kept): structural", r["structural"] and not r["ok"])
    after = before.iloc[::-1].reset_index(drop=True)
    r = verify(before, after)
    check("reverse the rows: structural", r["structural"] and not r["ok"])

    after = before.copy(); after["amount"] = after["amount"].round(1)
    r = verify(before, after)
    check("round a float column: caught as 'changed'", violations_of(r, "changed", "amount") and r["repaired"]["amount"].equals(before["amount"]), str(r["violations"]))

    after = before.rename(columns={"state": "State"})
    r = verify(before, after)
    check("rename a column: structural (a rename is a drop plus an add)", r["structural"] and not r["ok"] and r["repaired"].equals(before), str(r["violations"]))
    after = before.drop(columns=["qty_text"])
    check("drop a column: structural", verify(before, after)["structural"])

    after = before.copy(); after["customer_id"] = after["customer_id"].astype(int)
    r = verify(before, after)
    check("'007' -> 7: caught as 'changed' (an identifier, not a number)", violations_of(r, "changed", "customer_id")
          and r["repaired"]["customer_id"].equals(before["customer_id"]), str(r["violations"]))
    after = before.copy(); after["customer_id"] = after["customer_id"].str.lstrip("0")
    check("'007' -> '7' (as text): caught as well", violations_of(verify(before, after), "changed", "customer_id"))

    after = before.copy(); after["productcode"] = after["productcode"].str.lower()
    r = verify(before, after)
    check("lowercase a product-code column: caught as 'changed' (case counts as a change in high-cardinality text)",
          violations_of(r, "changed", "productcode") and r["repaired"]["productcode"].equals(before["productcode"]), str(r["violations"]))

    after = before.copy(); after["state"] = [True] * len(after)
    r = verify(before, after)
    check("a flag column written over an existing name: caught, the original column comes back",
          violations_of(r, "changed", "state") and r["repaired"]["state"].equals(before["state"]))
    after = pd.concat([before, pd.Series([True] * len(before), name="state")], axis=1)
    r = verify(before, after)
    check("a flag column that duplicates an existing name: 'name_clash', dropped, the original untouched",
          violations_of(r, "name_clash", "state") and list(r["repaired"].columns) == list(before.columns)
          and r["repaired"]["state"].equals(before["state"]) and "state" in r["reverted_columns"], str(r["violations"]))

    # ── MUST PASS ──────────────────────────────────────────────────────────────────────────────────────────────
    b2 = before.copy(); b2["state"] = [" " + s + " " if i % 4 == 0 else s for i, s in enumerate(b2["state"])]
    after = b2.copy(); after["state"] = after["state"].str.strip()
    r = verify(b2, after)
    check("whitespace tidying passes and is counted (50 cells)", r["ok"] and not r["violations"]
          and r["kept_changes"] == [{"column": "state", "count": 50, "kind": "tidied"}], str(r["kept_changes"]))
    check("whitespace tidying: the tidied column is in the result", r["repaired"]["state"].equals(after["state"]))
    b3 = before.copy(); b3["productline"] = [LINES[i % 5] + ("  " if i % 5 == 0 else "") for i in range(len(b3))]
    after = b3.copy(); after["productline"] = after["productline"].str.replace(r"\s+", " ", regex=True)
    check("internal whitespace collapsing passes", verify(b3, after)["ok"] and not verify(b3, after)["violations"])

    b4 = before.copy(); b4["state"] = [s.lower() if i % 3 == 0 else s for i, s in enumerate(b4["state"])]
    after = b4.copy(); after["state"] = after["state"].str.upper()
    r = verify(b4, after)
    check("case normalisation in a low-cardinality column passes", r["ok"] and not r["violations"] and r["kept_changes"]
          and r["kept_changes"][0]["kind"] == "tidied", str(r["violations"]))

    after = before.copy(); after["amount_is_outlier"] = [None if i % 5 == 0 else bool(i % 2) for i in range(len(after))]
    r = verify(before, after)
    check("a new boolean flag column (with nulls) passes and is not counted by the tripwire",
          r["ok"] and not r["fault"] and r["new_columns"] == ["amount_is_outlier"]
          and {"column": "amount_is_outlier", "count": len(before), "kind": "added"} in r["kept_changes"]
          and "amount_is_outlier" in r["repaired"].columns and r["repaired"]["amount_is_outlier"].isna().sum() == 40, str(r["violations"]))
    check("... and the original columns come first, in their original order",
          list(r["repaired"].columns) == list(before.columns) + ["amount_is_outlier"])

    after = before.copy(); after["qty_text"] = pd.to_numeric(after["qty_text"]).astype(float)
    r = verify(before, after)
    check("'1' -> 1.0 passes (same number)", r["ok"] and not r["violations"] and r["kept_changes"][0]["column"] == "qty_text", str(r["violations"]))
    after = before.copy(); after["qty_text"] = (pd.to_numeric(after["qty_text"]) + 1).astype(float)
    check("'1' -> 2.0 does not", violations_of(verify(before, after), "changed", "qty_text"))

    after = before.copy(); after["when"] = pd.to_datetime(after["when"], format="%d %b %Y").dt.strftime("%Y-%m-%d")
    r = verify(before, after)
    check("a date reformatted to the same date passes", r["ok"] and not r["violations"] and r["kept_changes"][0]["column"] == "when", str(r["violations"]))
    after = before.copy(); after["when"] = pd.to_datetime(after["when"], format="%d %b %Y")
    check("a date column parsed into real datetimes (same dates) passes", verify(before, after)["ok"] and not verify(before, after)["violations"])
    after = before.copy(); after["when"] = (pd.to_datetime(after["when"], format="%d %b %Y") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    check("a date moved by a day does not", violations_of(verify(before, after), "changed", "when"))
    b5 = pd.DataFrame({"label": ["May", "June", "March", "Sat"] * 30})
    after = b5.copy(); after["label"] = ["06", "07", "03", "09"] * 30
    check("month names are not dates (no digit in the before text)", violations_of(verify(b5, after), "changed", "label"))

    # nulls equal nulls
    b6 = pd.DataFrame({"x": ["a", None, "b", np.nan], "y": [1.0, np.nan, None, 4.0]})
    r = verify(b6, b6.copy())
    check("None and NaN on both sides are equal: nothing found", r["ok"] and not r["violations"])

    # ── a mixed fix keeps the good and reverts the bad ───────────────────────────────────────────────────────────
    after = b3.copy()
    after["productline"] = after["productline"].str.replace(r"\s+", " ", regex=True)          # good
    after.loc[[1, 2, 3], "state"] = None                                                       # bad
    after["amount_is_outlier"] = after["amount"] > 900                                         # good
    r = verify(b3, after)
    check("mixed fix: ok overall, only the bad column reverted", r["ok"] and r["reverted_columns"] == ["state"], str(r["reverted_columns"]))
    check("mixed fix: the good changes are kept (tidied column + new column)",
          {k["column"] for k in r["kept_changes"]} == {"productline", "amount_is_outlier"}, str(r["kept_changes"]))
    check("mixed fix: the table is the original plus the good changes",
          r["repaired"]["state"].equals(b3["state"]) and r["repaired"]["productline"].equals(after["productline"])
          and "amount_is_outlier" in r["repaired"].columns)
    check("mixed fix: nulls over the original columns did not rise",
          int(r["repaired"][list(b3.columns)].isna().sum().sum()) <= int(b3.isna().sum().sum()))

    # ── the global tripwire: a simulated verifier bug ────────────────────────────────────────────────────────────
    real = av._compare_column
    av._compare_column = lambda b, a, n: {"filled": 0, "blanked": 0, "changed": 0, "tidied": 0, "examples": {}}
    try:
        after = before.copy(); after.loc[[3, 4], "state"] = None
        r = verify(before, after)
    finally:
        av._compare_column = real
    check("tripwire: a verifier bug that lets a blank through discards the whole fix with fault=true",
          r["fault"] is True and r["ok"] is False and r["repaired"].equals(before) and not r["structural"], str({k: r[k] for k in ("ok", "fault")}))
    after = before.copy(); after["amount_is_outlier"] = [None] * len(after)
    check("tripwire: nulls in a NEW column never trigger it", verify(before, after)["fault"] is False)

    # ── escaping, counts, speed ────────────────────────────────────────────────────────────────────────────────
    b7 = pd.DataFrame({"note": ["line one\nline two\t" + "x" * 100 + "\x07"] + ["ok"] * 30})
    after = b7.copy(); after.loc[0, "note"] = None
    r = verify(b7, after)
    ex = r["violations"][0]["example"]
    check("examples are escaped (no raw newline, tab or control character) and cut to 60 characters",
          "\n" not in ex and "\t" not in ex and "\x07" not in ex and len(ex) <= 60 and "\\n" in ex, repr(ex))

    big = pd.DataFrame({
        "state": [STATES[i % 15] for i in range(100_000)],
        "code": [f"P{i:06d}" for i in range(100_000)],
        "amount": np.arange(100_000, dtype=float) / 7,
        "when": ["2024-01-05"] * 100_000,
        "note": ["  padded  " if i % 3 == 0 else "plain" for i in range(100_000)],
    })
    after = big.copy(); after["note"] = after["note"].str.strip(); after["state"] = after["state"].str.lower()
    after.loc[:999, "amount"] = after.loc[:999, "amount"] + 1
    t = time.time()
    r = verify(big, after)
    took = time.time() - t
    check(f"100,000 rows: fast ({took:.2f}s)", took < 5.0, f"{took:.2f}s")
    check("100,000 rows: exact counts (1,000 changed amounts; padded notes tidied)",
          violations_of(r, "changed", "amount")[0]["count"] == 1000
          and {k["column"]: k["count"] for k in r["kept_changes"]}.get("note") == 33334, str(r["kept_changes"]))

    # ── the text helpers ───────────────────────────────────────────────────────────────────────────────────────
    check("describe: blanked", av.describe_violation({"kind": "blanked", "column": "state", "count": 1267, "example": "CA"})
          == "state: 1,267 existing values would have been blanked")
    check("describe: filled", av.describe_violation({"kind": "filled", "column": "territory", "count": 2, "example": "x"})
          == "territory: 2 missing values would have been filled in")
    check("describe: changed", av.describe_violation({"kind": "changed", "column": "p", "count": 5, "example": "x"})
          == "p: 5 existing values would have been changed")
    mk = lambda **kw: {"kept_changes": [], "reverted_columns": [], "structural": False, "fault": False, **kw}
    check("message: applied", av.build_message(mk(kept_changes=[1, 2, 3])) == "Applied 3 safe changes, used for your questions")
    check("message: applied (one)", av.build_message(mk(kept_changes=[1])) == "Applied 1 safe change, used for your questions")
    check("message: applied and reverted", av.build_message(mk(kept_changes=[1, 2], reverted_columns=["a"]))
          == "Applied 2 safe changes; reverted 1 column that would have changed your values")
    check("message: nothing changed (reverted)", av.build_message(mk(reverted_columns=["a", "b"])).startswith("No data was changed."))
    check("message: nothing changed (structural / fault / nothing)", all(av.build_message(mk(**kw)).startswith("No data was changed.")
          for kw in ({"structural": True}, {"fault": True}, {})))

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
