"""
test_encoding_artifacts.py — detect_encoding.find_encoding_artifacts: wrong-charset text (C1 control characters and
UTF-8-read-as-cp1252 mojibake) is flagged, one card per column; genuine accented text is not.

Pure function, no LLM, no sandbox, no network. Run: python test_encoding_artifacts.py
"""
import glob
import os
import sys
import time

import pandas as pd

from detect_encoding import EXAMPLE_MAX, REASON, find_encoding_artifacts

FAIL = []

# 19 artifacts: C1 control characters and the mojibake signatures (Ã¤ Ã© Ã¶ Ã¼, Â°, Â£, â€)
BAD = ["Berguvsv\x84gen 8", "MÃ¤rz", "donâ€™t", "CafÃ©", "Â°C", "Kr\x92m", "naÃ¯ve", "Ã¶l", "â€œquotedâ€\x9d",
       "Stra\x9ee", "ZÃ¼rich", "Â£5", "Ã¤", "\x85dots", "ÃŸ", "Ã©tÃ©", "Ã±o", "â€", "oâ€“o"]
CLEAN_ACCENTS = ["é", "ä", "ö", "ü", "å", "ñ", "ç", "SÃO PAULO", "pâté", "5€", "Ångström", "naïve",
                 "“curly” ‘quotes’ – dash…", "Zürich", "Café", "München", "São Paulo", "Ñandú", "Â"]


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def sample_csvs():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_data")
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        parts = os.path.relpath(p, root).lower().split(os.sep)
        if "clean_bench" in parts or "exp" in parts or any("frozen" in x for x in parts):
            continue   # benchmark and experiment data: not sample files, and never read here
        out.append(p)
    return out


def main():
    print("=" * 70)
    print("PART 1 — 19 bad values among 2,000 normal ones")
    print("=" * 70)
    assert len(BAD) == 19
    normal = [f"Street {i} Road" for i in range(2000)]
    df = pd.DataFrame({"addr": normal + BAD, "n": range(2019)})
    cards = find_encoding_artifacts(df)
    check("exactly one card", len(cards) == 1, str(cards))
    c = cards[0] if cards else {}
    check("the card names the column and the reason", c.get("column") == "addr" and c.get("reason") == REASON == "encoding_artifact", str(c))
    check("count is 19", c.get("count") == 19, str(c))
    check("of is the 2,019 non-blank values", c.get("of") == 2019, str(c))
    check("kinds lists both, in order", c.get("kinds") == ["c1_control", "mojibake"], str(c))
    check("the example is the first hit, with the control character shown as \\x84",
          c.get("example") == "Berguvsv\\x84gen 8", repr(c.get("example")))

    print("\n" + "=" * 70)
    print("PART 2 — kinds")
    print("=" * 70)
    only_c1 = find_encoding_artifacts(pd.DataFrame({"c": ["ok", "a\x84b", "fine"]}))
    check("a control character alone is kinds ['c1_control']", only_c1 and only_c1[0]["kinds"] == ["c1_control"], str(only_c1))
    only_moji = find_encoding_artifacts(pd.DataFrame({"c": ["ok", "MÃ¤rz", "fine"]}))
    check("mojibake alone is kinds ['mojibake']", only_moji and only_moji[0]["kinds"] == ["mojibake"], str(only_moji))
    for label, value in [("Ã¤", "Ã¤"), ("Ã©", "Ã©"), ("Ã¶", "Ã¶"), ("Ã¼", "Ã¼"), ("Â°", "Â°"), ("Â£", "Â£"), ("â€", "â€™"), ("\\x85", "a\x85b")]:
        got = find_encoding_artifacts(pd.DataFrame({"c": [value]}))
        check(f"signature {label} is flagged", len(got) == 1 and got[0]["count"] == 1, str(got))

    print("\n" + "=" * 70)
    print("PART 3 — genuine accented text is NOT flagged")
    print("=" * 70)
    check("clean accented column (str dtype)", find_encoding_artifacts(pd.DataFrame({"city": CLEAN_ACCENTS})) == [])
    check("clean accented column (object dtype)",
          find_encoding_artifacts(pd.DataFrame({"city": pd.Series(CLEAN_ACCENTS, dtype=object)})) == [])
    check("the clean column's dtype really is str under this pandas, or object when forced",
          str(pd.DataFrame({"x": CLEAN_ACCENTS})["x"].dtype) in ("str", "object", "string"))

    print("\n" + "=" * 70)
    print("PART 4 — one card per column, other columns skipped")
    print("=" * 70)
    mixed = pd.DataFrame({
        "good": ["a", "b", "c"],
        "bad1": ["x", "MÃ¤rz", "y\x84"],            # both kinds in ONE column -> still one card
        "bad2": ["Â°C", "ok", "ok"],
        "num": [1, 2, 3],
        "flt": [1.5, 2.5, 3.5],
        "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    cards = find_encoding_artifacts(mixed)
    check("one card per bad column, in column order", [c["column"] for c in cards] == ["bad1", "bad2"], str(cards))
    check("both kinds in one column are one card", cards[0]["count"] == 2 and cards[0]["kinds"] == ["c1_control", "mojibake"], str(cards[0]))
    check("numeric and date columns are skipped (even though str() of them is clean)", all(c["column"] not in ("num", "flt", "when") for c in cards))
    num_with_c1 = pd.DataFrame({"n": [1, 2, 3]})
    check("a numeric column is skipped", find_encoding_artifacts(num_with_c1) == [])
    dup = pd.DataFrame([["ok", "a\x84"]], columns=["same", "same"])
    check("two columns with the same name do not crash and the bad one is found", len(find_encoding_artifacts(dup)) == 1)

    print("\n" + "=" * 70)
    print("PART 5 — blanks, empty frames, a single artifact")
    print("=" * 70)
    blank = pd.DataFrame({"c": [None, "", "   ", "fine", None]})
    check("blank-safe: blanks are skipped, nothing flagged", find_encoding_artifacts(blank) == [])
    blank_bad = find_encoding_artifacts(pd.DataFrame({"c": [None, "", "   ", "Ã¤", "ok", float("nan")]}))
    check("'of' counts only non-blank values (2 here: 'Ã¤' and 'ok')",
          blank_bad and blank_bad[0]["of"] == 2 and blank_bad[0]["count"] == 1, str(blank_bad))
    check("an empty frame gives []", find_encoding_artifacts(pd.DataFrame()) == [])
    check("a frame with columns and no rows gives []", find_encoding_artifacts(pd.DataFrame({"c": pd.Series([], dtype=object)})) == [])
    check("an all-blank column gives []", find_encoding_artifacts(pd.DataFrame({"c": [None, None]})) == [])
    one = find_encoding_artifacts(pd.DataFrame({"c": ["fine"] * 500 + ["one\x84bad"]}))
    check("a single artifact is still flagged", len(one) == 1 and one[0]["count"] == 1 and one[0]["of"] == 501, str(one))

    print("\n" + "=" * 70)
    print("PART 6 — the example is escaped and at most 60 characters")
    print("=" * 70)
    ex = find_encoding_artifacts(pd.DataFrame({"c": ["a\x84b"]}))[0]["example"]
    check("a control character shows as \\x84 (no raw control character)", ex == "a\\x84b" and not any(0x80 <= ord(ch) <= 0x9f for ch in ex), repr(ex))
    ex = find_encoding_artifacts(pd.DataFrame({"c": ["tab\there\x84"]}))[0]["example"]
    check("a tab is shown as \\t", ex == "tab\\there\\x84", repr(ex))
    long_value = "A" * 100 + "\x84" + "B" * 100
    ex = find_encoding_artifacts(pd.DataFrame({"c": [long_value]}))[0]["example"]
    check("a long value is cut to at most 60 characters", len(ex) <= EXAMPLE_MAX == 60, f"len={len(ex)}")
    ex = find_encoding_artifacts(pd.DataFrame({"c": ["x" * 58 + "\x84"]}))[0]["example"]
    check("the cut applies after escaping, so the card never exceeds 60", len(ex) == 60, f"len={len(ex)}")
    ex = find_encoding_artifacts(pd.DataFrame({"c": ["MÃ¤rz"]}))[0]["example"]
    check("mojibake examples stay readable (printable characters are kept)", ex == "MÃ¤rz", repr(ex))
    first = find_encoding_artifacts(pd.DataFrame({"c": ["ok", "first\x84", "second\x85"]}))[0]["example"]
    check("the example is the FIRST hit", first == "first\\x84", repr(first))

    print("\n" + "=" * 70)
    print("PART 7 — speed")
    print("=" * 70)
    big = pd.DataFrame({"t": ["Street Road number %d" % (i % 997) for i in range(100_000)], "u": ["Café"] * 100_000})
    t0 = time.time()
    got = find_encoding_artifacts(big)
    secs = time.time() - t0
    check("100,000 rows x 2 columns in a few seconds", secs < 5.0 and got == [], f"{secs:.2f}s cards={got}")
    print(f"    ({secs:.2f}s)")

    print("\n" + "=" * 70)
    print("PART 8 — the repo's sample CSVs give zero cards")
    print("=" * 70)
    paths = sample_csvs()
    check("sample CSVs were found", len(paths) >= 5, str(len(paths)))
    for p in paths:
        name = os.path.relpath(p, os.path.dirname(os.path.abspath(__file__)))
        try:
            frame = pd.read_csv(p)
        except UnicodeDecodeError:
            frame = pd.read_csv(p, encoding="latin1")
        got = find_encoding_artifacts(frame)
        check(f"zero cards: {name}", got == [], str(got))

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
        sys.exit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
