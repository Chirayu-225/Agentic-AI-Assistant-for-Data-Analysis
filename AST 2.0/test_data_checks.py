"""
test_data_checks.py — data_checks.check_dataframe: the format check (drift / variance / consistent / skipped) and the
encoding cards, merged under one cap and one ranking.

Hand-made, deterministic fixtures: no random numbers, no files, no rates. Pure functions, no LLM, no sandbox, no network.
Run: python test_data_checks.py
"""
import ast
import copy
import os
import subprocess
import sys

import pandas as pd

import cleaning                      # the TEST may import it; data_checks itself may not
import data_checks as dc
from data_checks import check_dataframe, signature

FAIL = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────────────────────────
def one(values, name="v"):
    return pd.DataFrame({name: values})


def rec(result, name):
    return next(r for r in result["columns"] if r["column"] == name)


def cards_for(result, name, reason=None):
    return [c for c in result["cards"] if c["column"] == name and (reason is None or c["reason"] == reason)]


def drift_values(minority, n=200, first_minority=10):
    """n values: 'R-001'-style (signature A-999), with `minority` lower-case 'r001'-style cells (signature a999)."""
    out = [f"R-{i:03d}" for i in range(n)]
    for k in range(minority):
        out[first_minority + k] = f"r{first_minority + k:03d}"
    return out


def variance_values(n=200):
    return [(f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" if i % 2 == 0 else f"{1 + i % 12:02d}/{1 + i % 28:02d}/2024")
            for i in range(n)]


NAMES = ["Anna", "Ben", "Chloe", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal"]


def name_values(n=200, garbled=0):
    out = [f"{NAMES[i % 10]} {NAMES[(i // 10) % 10]}" for i in range(n)]
    for k in range(garbled):
        out[5 + k] = "Ã¤" + out[5 + k]
    return out


def phone_templates():
    seps = [("-", "-"), (" ", " "), (".", "."), ("/", "-"), ("-", " "), (" ", "-"), (".", "-")]
    prefixes = ["", "+9 ", "+99 ", "9"]
    templates = [f"{p}999{a}999{b}9999" for p in prefixes for a, b in seps]
    templates.append("(999) 999-9999")
    return templates


def fill_template(template, k):
    j = -1
    out = []
    for ch in template:
        if ch == "9":
            j += 1
            out.append(str((k * 7 + j * 3 + 1) % 10))
        else:
            out.append(ch)
    return "".join(out)


def phone_values():
    templates = phone_templates()
    counts = [840, 700, 560, 420] + [12] * 5 + [11] * 20          # 30/25/20/15% then 25 shapes over the last 10%
    base = []
    for template, count in zip(templates, counts):
        base.extend(fill_template(template, k) for k in range(count))
    assert len(templates) == 29 and len(set(templates)) == 29 and len(base) == 2800
    return [base[(i * 37) % 2800] for i in range(2800)]            # 37 is coprime with 2800: a fixed shuffle


def zip_values(n=1000, plus4=30, first=200):
    out = [f"{10000 + (i * 37) % 89999:05d}" for i in range(n)]
    for k in range(plus4):
        out[first + k] = f"{out[first + k]}-{1000 + k:04d}"
    return out


def main():
    print("=" * 70)
    print("PART 1 — the signature")
    print("=" * 70)
    check("digits, upper and lower case letters, everything else kept",
          signature("(555) 867-5309") == "(999) 999-9999" and signature("AB-12") == "AA-99" and signature("Ab1 x") == "Aa9 a")
    check("trimmed first", signature("  12 \t") == "99")
    check("non-ASCII letters follow case; non-ASCII digits are kept as they are",
          signature("Été") == "Aaa" and signature("٣") == "٣")
    check("numbers go through str()", signature(5) == "9" and signature(12.5) == "99.9")

    print("\n" + "=" * 70)
    print("PART 2 — the spec's eleven fixtures")
    print("=" * 70)
    # (1) legitimate two-format column, 50/50
    r = check_dataframe(one(variance_values(200), "date"))
    c = rec(r, "date")
    check("(1) 50/50 two formats: variance", c["verdict"] == "variance" and c["n_signatures"] == 2 and c["dominant_share"] == 0.5, str(c))
    check("(1) exactly ONE informational card, no minority cards",
          len(r["cards"]) == 1 and r["cards"][0]["reason"] == "format_variance" and r["cards"][0]["informational"] is True
          and not cards_for(r, "date", "format_drift"), str(r["cards"]))
    check("(1) variance message", c["message"] == "date mixes 2 formats; the most common covers 50%. Counts of inconsistent "
                                                  "values here may be formatting differences, not errors.", c["message"])

    # (2) 97/3 drift
    r = check_dataframe(one(drift_values(minority=30, n=1000, first_minority=100), "ref"))
    c = rec(r, "ref")
    check("(2) 97/3: drift", c["verdict"] == "drift" and c["dominant_signature"] == "A-999" and c["dominant_share"] == 0.97 and c["minority_cells"] == 30, str(c))
    cards = cards_for(r, "ref")
    check("(2) one card, count 30, the right signature and example",
          len(cards) == 1 and cards[0]["reason"] == "format_drift" and cards[0]["count"] == 30
          and cards[0]["signature"] == "a999" and cards[0]["example"] == "r100" and cards[0]["of"] == 1000, str(cards))
    check("(2) message carries the total, the example and the caveat",
          cards[0]["message"] == "ref: 97% of values use the format A-999; 30 values use another format (e.g. r100). "
                                 "These may be legitimate variants.", cards[0]["message"])
    check("(2) the column record carries the same message", c["message"] == cards[0]["message"])

    # (3) free-text names and notes
    notes = [("call back in %d days" % i) if i % 5 == 0 else "left a message" for i in range(300)]
    r = check_dataframe(pd.DataFrame({"name": name_values(300), "notes": notes}))
    check("(3) names: skipped, no digits", rec(r, "name")["verdict"] == "skipped" and rec(r, "name")["skip_reason"] == "less_than_half_have_digits", str(rec(r, "name")))
    check("(3) notes with digits in only 20%: skipped, same reason", rec(r, "notes")["skip_reason"] == "less_than_half_have_digits")
    check("(3) and no cards", r["cards"] == [], str(r["cards"]))

    # (4) the phone example
    r = check_dataframe(one(phone_values(), "phone"))
    c = rec(r, "phone")
    check("(4) 2,800 rows in 29 shapes: variance", c["verdict"] == "variance" and c["n_checked"] == 2800, str(c)[:200])
    check("(4) n_signatures is 29", c["n_signatures"] == 29, str(c["n_signatures"]))
    check("(4) the top four are 30/25/20/15%", [t["count"] for t in c["top_signatures"][:4]] == [840, 700, 560, 420]
          and len(c["top_signatures"]) == 5, str(c["top_signatures"]))
    check("(4) no per-row flags: exactly one informational card",
          len(r["cards"]) == 1 and r["cards"][0]["reason"] == "format_variance" and r["cards"][0]["count"] == 0, str(r["cards"])[:200])

    # (5) varying-length integer order ids, as ints and as text
    ids = [7 * i + 1 for i in range(300)]
    for label, values in (("int", ids), ("text", [str(x) for x in ids])):
        r = check_dataframe(one(values, "order_id"))
        check(f"(5) order ids ({label}): skipped as plain numbers",
              rec(r, "order_id")["verdict"] == "skipped" and rec(r, "order_id")["skip_reason"] == "plain_numbers" and r["cards"] == [],
              str(rec(r, "order_id")))

    # (6) prices with decimals
    for label, values in (("text", [f"{10 + i * 0.75:.2f}" for i in range(200)]), ("float", [10 + i * 0.75 for i in range(200)])):
        r = check_dataframe(one(values, "price"))
        check(f"(6) prices ({label}): skipped as plain numbers",
              rec(r, "price")["skip_reason"] == "plain_numbers" and r["cards"] == [], str(rec(r, "price")))

    # (7) ZIP column, 3% ZIP+4
    r = check_dataframe(one(zip_values(), "zip"))
    c = rec(r, "zip")
    check("(7) ZIP with 3% ZIP+4: drift", c["verdict"] == "drift" and c["dominant_signature"] == "99999" and c["minority_cells"] == 30, str(c)[:200])
    cards = cards_for(r, "zip")
    check("(7) one card for 99999-9999, count 30", len(cards) == 1 and cards[0]["signature"] == "99999-9999" and cards[0]["count"] == 30, str(cards)[:200])

    # (8) a 40-row table
    r = check_dataframe(one(drift_values(minority=2, n=40), "small"))
    check("(8) 40 rows: skipped, nothing flagged", rec(r, "small")["skip_reason"] == "fewer_than_100_values" and r["cards"] == [], str(rec(r, "small")))
    check("(8) n_rows is reported", r["n_rows"] == 40)

    # (9) an all-null column and a sentinel-only column
    sentinels = ["N/A", "null", "--", "nan", " NONE ", "-", "", "NaT", "na"]
    r = check_dataframe(pd.DataFrame({"empty": [None] * 200, "sent": [sentinels[i % 9] for i in range(200)]}))
    check("(9) all-null column: skipped, 0 values", rec(r, "empty")["skip_reason"] == "fewer_than_100_values" and rec(r, "empty")["n_checked"] == 0)
    check("(9) sentinel-only column: skipped, 0 values", rec(r, "sent")["skip_reason"] == "fewer_than_100_values" and rec(r, "sent")["n_checked"] == 0)
    check("(9) no cards", r["cards"] == [])

    # (10) run twice
    big = pd.DataFrame({"zip": zip_values(), "ref": drift_values(30, 1000, 100), "when": variance_values(1000), "name": name_values(1000, garbled=7)})
    a, b = check_dataframe(big), check_dataframe(big)
    check("(10) two runs give equal output, same order", a == b and [c["column"] for c in a["cards"]] == [c["column"] for c in b["cards"]])

    # (11) padding does not matter
    base = zip_values()
    pads = [" ", "\t", "  \t", ""]
    padded = [pads[i % 4] + v + pads[(i + 1) % 4] for i, v in enumerate(base)]
    check("(11) leading/trailing spaces and tabs: identical signatures, counts and cards",
          check_dataframe(one(base, "zip")) == check_dataframe(one(padded, "zip")))

    print("\n" + "=" * 70)
    print("PART 3 — the extras")
    print("=" * 70)
    mixed_blanks = [f"A-{i:03d}" for i in range(150)] + ["N/A", "n/a", " NaN ", "NULL", "none", "NA", "--", "-", "", "NaT", " nat\t", None, float("nan")]
    r = check_dataframe(one(mixed_blanks, "code"))
    c = rec(r, "code")
    check("sentinels and missing values are excluded from the counts", c["n_checked"] == 150 and c["n_signatures"] == 1 and c["verdict"] == "consistent", str(c))
    check("the sentinel set equals cleaning._SENTINELS", dc._SENTINELS == cleaning._SENTINELS == {"nan", "none", "null", "n/a", "na", "--", "-", "", "nat"},
          f"{dc._SENTINELS} vs {cleaning._SENTINELS}")

    r = check_dataframe(one([f"2024-01-{1 + i % 28:02d}" for i in range(150)], "d"))
    check("a fully uniform column is 'consistent' with no card and the neutral message",
          rec(r, "d")["verdict"] == "consistent" and r["cards"] == [] and rec(r, "d")["minority_cells"] == 0
          and rec(r, "d")["message"] == "d: all values use the format 9999-99-99.", str(rec(r, "d")))

    # 5-per-file cap and ranking
    counts = [5, 12, 9, 12, 19, 7, 3]
    wide = pd.DataFrame({f"c{i}": drift_values(m) for i, m in enumerate(counts)})
    r = check_dataframe(wide)
    check("every column is drift", all(rec(r, f"c{i}")["verdict"] == "drift" for i in range(7)))
    check("at most 5 cards per file", len(r["cards"]) == 5, str(len(r["cards"])))
    check("ranked by cells affected, ties by column position", [(c["column"], c["count"]) for c in r["cards"]]
          == [("c4", 19), ("c1", 12), ("c3", 12), ("c2", 9), ("c5", 7)], str([(c["column"], c["count"]) for c in r["cards"]]))
    check("the column records are not capped", len(r["columns"]) == 7)

    # 2-per-column cap, count ties broken by signature
    vals = [f"R-{i:03d}" for i in range(200)]
    for k in range(5):
        vals[10 + k] = f"RAB{k}"                 # signature AAA9   (5)
        vals[20 + k] = f"rab{k}"                 # signature aaa9   (5)
    for k in range(4):
        vals[30 + k] = f"RA-{k}"                 # signature AA-9   (4)
    r = check_dataframe(one(vals, "x"))
    c = rec(r, "x")
    check("three minority signatures: drift, total minority is 14", c["verdict"] == "drift" and c["minority_cells"] == 14, str(c)[:200])
    check("only the 2 largest become cards, equal counts ordered by signature string",
          [(x["signature"], x["count"]) for x in cards_for(r, "x")] == [("AAA9", 5), ("aaa9", 5)], str(cards_for(r, "x")))
    check("each card carries the TOTAL minority count in its message", all("14 values use another format" in x["message"] for x in cards_for(r, "x")))
    check("the record still lists up to 5 signatures (dominant + 3 minority)", len(c["top_signatures"]) == 4)

    # input not mutated
    df = pd.DataFrame({"zip": [(" " + v + "\t") for v in zip_values()], "n": list(range(1000)), "f": [i * 0.5 for i in range(1000)], "t": name_values(1000, garbled=3)})
    snapshot = df.copy(deep=True)
    columns_before, dtypes_before = list(df.columns), [str(t) for t in df.dtypes]
    check_dataframe(df)
    pd.testing.assert_frame_equal(df, snapshot)
    check("the input frame is not modified (values, columns, dtypes)", list(df.columns) == columns_before and [str(t) for t in df.dtypes] == dtypes_before)

    # imports
    tree = ast.parse(open(os.path.join(HERE, "data_checks.py"), encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    check("data_checks imports only re, collections, pandas, detect_encoding", imported <= {"__future__", "re", "collections", "pandas", "detect_encoding"}, str(imported))
    probe = ("import sys; sys.path.insert(0, %r); import data_checks; "
             "bad = [m for m in ('cleaning','analysis_service','llm_provider','sandbox','sandbox_limits','sandbox_proxy','query_engine','streamlit','groq') if m in sys.modules]; "
             "print('LOADED', bad)") % HERE
    out = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=120).stdout
    check("importing data_checks loads none of cleaning / analysis_service / llm_provider / the sandbox", "LOADED []" in out, out[-200:])

    print("\n" + "=" * 70)
    print("PART 4 — integration with the encoding cards")
    print("=" * 70)
    r = check_dataframe(pd.DataFrame({"name": name_values(300, garbled=7)}))
    enc = cards_for(r, "name", "encoding_artifact")
    check("encoding cards appear in check_dataframe's output", len(enc) == 1 and enc[0]["count"] == 7 and enc[0]["of"] == 300 and enc[0]["kinds"] == ["mojibake"], str(r["cards"]))
    check("the encoding card has the common keys and a message",
          set(("reason", "column", "column_index", "signature", "count", "of", "example", "message")) <= set(enc[0]) and enc[0]["signature"] is None
          and enc[0]["message"].startswith("name: 7 of 300 values contain encoding artifacts"), str(enc[0]))
    check("encoding message wording (mojibake example)",
          enc[0]["message"] == "name: 7 of 300 values contain encoding artifacts (e.g. Ã¤Farid Anna).", enc[0]["message"])
    check("a column the format check skipped can still carry an encoding card", rec(r, "name")["verdict"] == "skipped")

    mix = pd.DataFrame({
        "e1": name_values(200, garbled=25),
        "d1": drift_values(19),
        "e2": name_values(200, garbled=12),
        "d2": drift_values(12),
        "v1": variance_values(200),
        "d3": drift_values(9),
        "e3": name_values(200, garbled=3),
    })
    r = check_dataframe(mix)
    got = [(c["column"], c["reason"], c["count"]) for c in r["cards"]]
    check("cap and ranking across both kinds: 5 cards, by cells affected, ties by column position",
          got == [("e1", "encoding_artifact", 25), ("d1", "format_drift", 19), ("e2", "encoding_artifact", 12), ("d2", "format_drift", 12),
                  ("d3", "format_drift", 9)], str(got))
    check("the variance card (0 cells) and the 3-cell encoding card fall off the cap", not cards_for(r, "v1") and not cards_for(r, "e3"))
    check("encoding cards do not use up the 2-per-column format limit",
          len([c for c in check_dataframe(mix.iloc[:, [0]])["cards"]]) == 1)

    # one column with BOTH drift and encoding artifacts
    both = [f"Zone-{i % 90:02d}" for i in range(300)]
    for k in range(15):
        both[10 + k] = f"Zone{k:02d}"                  # signature Aaaa99    (15)
    for k in range(10):
        both[40 + k] = f"Zone-\x84{k}"                 # signature Aaaa-\x849 (10), and a C1 control character
    r = check_dataframe(one(both, "label"))
    reasons = sorted(c["reason"] for c in cards_for(r, "label"))
    check("one column with drift AND artifacts yields cards of both reasons",
          reasons == ["encoding_artifact", "format_drift", "format_drift"], str(reasons))
    check("the encoding card there counts the 10 garbled cells",
          cards_for(r, "label", "encoding_artifact")[0]["count"] == 10 and cards_for(r, "label", "encoding_artifact")[0]["kinds"] == ["c1_control"])
    check("the column's format record is still a drift verdict", rec(r, "label")["verdict"] == "drift" and rec(r, "label")["minority_cells"] == 25, str(rec(r, "label"))[:200])
    check("encoding message wording (control-character example is escaped)",
          cards_for(r, "label", "encoding_artifact")[0]["message"] == "label: 10 of 300 values contain encoding artifacts (e.g. Zone-\\x840).",
          cards_for(r, "label", "encoding_artifact")[0]["message"])

    # at most 2 variance cards per file; a drift card and an encoding card still appear
    six = {f"m{k}": variance_values(200) for k in range(1, 7)}
    only_variance = check_dataframe(pd.DataFrame(six))
    check("6 mixed-format columns give exactly 2 variance cards (the first two columns)",
          [(c["column"], c["reason"]) for c in only_variance["cards"]] == [("m1", "format_variance"), ("m2", "format_variance")],
          str([(c["column"], c["reason"]) for c in only_variance["cards"]]))
    check("all 6 columns still get a variance record", all(rec(only_variance, f"m{k}")["verdict"] == "variance" for k in range(1, 7)))
    r = check_dataframe(pd.DataFrame({**six, "d": drift_values(19), "e": name_values(200, garbled=4)}))
    reasons = [c["reason"] for c in r["cards"]]
    check("with 6 variance columns, a drift card and an encoding card still appear",
          reasons.count("format_variance") == 2 and "format_drift" in reasons and "encoding_artifact" in reasons, str(reasons))
    check("the findings rank before the informational cards, and the cap of 5 still holds",
          reasons == ["format_drift", "encoding_artifact", "format_variance", "format_variance"] and len(r["cards"]) <= 5, str(reasons))

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
        sys.exit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
