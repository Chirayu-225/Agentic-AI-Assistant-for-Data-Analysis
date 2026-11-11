"""
test_fill_missing.py — fill_missing (the rules) and fill_service.run_fill (the working table).

Evidence first: a blank is filled from other rows only when another column PROVES the value (the relationship holds for
99% of the rows that have both values, is shown by at least 5 repeated groups, and every known row of the group agrees).
Numbers with no evidence get their column's median. Text with no evidence stays blank unless "Unknown" is asked for.
Dates, true/false columns, empty columns and identifier-like columns are never filled. No model is involved.
Run: python test_fill_missing.py
"""
import io
import os
import time

import numpy as np
import pandas as pd

FAIL = []
URL = "http://test/files/data.csv"


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def customers(n_customers=40, rows_each=4):
    cust = [f"C{i:02d}" for i in range(n_customers)]
    df = pd.DataFrame({"customer": [c for _ in range(rows_each) for c in cust]})
    df["postal"] = df["customer"].map({c: f"P{i:03d}" for i, c in enumerate(cust)})
    return df


def main():
    from fill_missing import fill_missing

    # ── numbers: the median of the column's own values ──
    price = pd.DataFrame({"price": [10.0, 20.0, None, 40.0] * 10})
    out, rep = fill_missing(price, median=True)
    check("numbers with no evidence: the column median", rep["filled"][0]["method"] == "median" and rep["filled"][0]["value"] == 20.0 and out["price"].isna().sum() == 0 and out.loc[2, "price"] == 20.0)
    frac = pd.DataFrame({"q": pd.array([1, 2, None, None] * 10, dtype="Int64")})
    out, rep = fill_missing(frac, median=True)
    check("nullable whole-number column with a fractional median: filled, not crashed", out["q"].isna().sum() == 0 and float(out.loc[2, "q"]) == 1.5, str(out["q"].dtype))
    whole = pd.DataFrame({"q": pd.array([2, 4, None, 6] * 10, dtype="Int64")})
    check("whole median stays whole", float(fill_missing(whole, median=True)[0].loc[2, "q"]) == 4.0)

    # ── text with no evidence: blank by default, "Unknown" only on request ──
    region = pd.DataFrame({"region": ["EMEA", "NA", None, "NA"] * 10})
    out, rep = fill_missing(region)
    check("text with no evidence stays blank by default and says why", out["region"].isna().sum() == 10 and rep["skipped"][0]["kind"] == "no_evidence" and rep["remaining_text"] == 10 and rep["total_filled"] == 0)
    out, rep = fill_missing(region, text_placeholder=True)
    check("with the Unknown button: Unknown, and a real 'NA' value is untouched", out.loc[2, "region"] == "Unknown" and (out["region"] == "NA").sum() == 20 and rep["filled"][0]["value"] == "Unknown" and rep["remaining_text"] == 0)
    cat = pd.DataFrame({"cat": pd.Categorical(["a", "b", None, "a"] * 10)})
    check("category column with the Unknown button: Unknown added as a category", fill_missing(cat, text_placeholder=True)[0].loc[2, "cat"] == "Unknown")

    # ── never filled ──
    never = pd.DataFrame({
        "when": pd.to_datetime(["2024-01-01", None, "2024-03-01", "2024-04-01"] * 10),
        "flag": pd.array([True, False, None, True] * 10, dtype="boolean"),
        "emptycol": [None] * 40,
        "uid": [f"id{i}" if i % 10 else None for i in range(40)],
    })
    out, rep = fill_missing(never, text_placeholder=True)
    sk = {s["column"] for s in rep["skipped"]}
    check("dates, true/false, empty and identifier-like columns are never filled, even with Unknown",
          sk == {"when", "flag", "emptycol", "uid"} and rep["total_filled"] == 0 and out.isna().sum().sum() == never.isna().sum().sum())

    # ── evidence from other rows ──
    df = customers()
    truth = df["postal"].copy()
    df.loc[[3, 45, 90], "postal"] = None                       # rows whose customer has other rows with a known postal
    c07 = df.index[df["customer"] == "C07"]
    df.loc[c07, "postal"] = None                                # a customer with NO known row: no evidence
    before = df.copy(deep=True)
    out, rep = fill_missing(df)
    inf = [f for f in rep["filled"] if f["method"] == "inferred"]
    check("evidence: blanks whose group is proven are filled from the other rows", len(inf) == 1 and inf[0]["count"] == 3 and inf[0]["from"] == "customer", str(rep["filled"]))
    check("evidence: the filled values are exactly the true ones", all(out.loc[i, "postal"] == truth[i] for i in (3, 45, 90)))
    check("evidence: a customer with no known row stays blank (no evidence, not guessed)", out.loc[c07, "postal"].isna().all() and rep["remaining_text"] == 4)
    check("evidence: known values are never overwritten", (out["postal"].notna() & before["postal"].notna() & (out["postal"] != before["postal"])).sum() == 0)
    check("evidence: the input table is not modified", df.equals(before))

    noisy = customers()
    for c in range(10):                                         # 10 customers have one row with a different postal: purity 93.75%
        noisy.loc[noisy.index[noisy["customer"] == f"C{c:02d}"][0], "postal"] = "X999"
    noisy.loc[[41, 85], "postal"] = None
    out, rep = fill_missing(noisy)
    check("a relationship that holds for less than 99% of rows is not evidence", rep["total_filled"] == 0 and out["postal"].isna().sum() == 2)

    few = pd.DataFrame({"customer": ["A"] * 6 + ["B"] * 6 + ["C"] * 6 + ["D"] * 6 + ["e", "f", "g", "h"],
                        "postal": ["pa"] * 6 + ["pb"] * 6 + ["pc"] * 6 + ["pd"] * 5 + [None] + ["pe", "pf", "pg", "ph"]})
    out, rep = fill_missing(few)
    check("a relationship shown by fewer than 5 repeated groups is not evidence", rep["total_filled"] == 0)

    mixed = customers()
    mixed.loc[mixed.index[mixed["customer"] == "C05"][:2], "postal"] = ["P005", "P777"]      # C05 disagrees with itself
    mixed.loc[mixed.index[mixed["customer"] == "C05"][2], "postal"] = None                    # ... and has a blank
    mixed.loc[mixed.index[mixed["customer"] == "C06"][0], "postal"] = None                    # an agreeing customer with a blank
    out, rep = fill_missing(mixed)
    c05_blank = mixed.index[mixed["customer"] == "C05"][2]
    c06_blank = mixed.index[mixed["customer"] == "C06"][0]
    check("every known row of the group must agree: the self-contradicting customer is not filled, the agreeing one is",
          pd.isna(out.loc[c05_blank, "postal"]) and out.loc[c06_blank, "postal"] == "P006")

    uniq = pd.DataFrame({"order": [f"O{i:03d}" for i in range(60)], "postal": [f"P{i:03d}" for i in range(60)]})
    uniq.loc[[5, 17], "postal"] = None
    check("an identifier-like key (every value different) is never used as evidence", fill_missing(uniq)[1]["total_filled"] == 0)

    holes = customers()
    holes.loc[[3, 4], "postal"] = None
    holes.loc[[3, 4], "customer"] = None                         # the blank rows have no key either
    check("a blank with a blank key is not filled", fill_missing(holes)[0].loc[[3, 4], "postal"].isna().all())

    # numbers: evidence beats the median
    prod = pd.DataFrame({"product": [f"p{i:02d}" for i in range(30)] * 4})
    prod["unit_price"] = prod["product"].map({f"p{i:02d}": 5.0 + i for i in range(30)})
    prod.loc[[2, 33, 64], "unit_price"] = None
    prod.loc[prod.index[prod["product"] == "p09"], "unit_price"] = None
    out, rep = fill_missing(prod, median=True)
    by = {f["method"]: f for f in rep["filled"]}
    check("numbers: evidence beats the median (exact values from the same product)", by["inferred"]["count"] == 3 and all(out.loc[i, "unit_price"] == 5.0 + (i % 30) for i in (2, 33, 64)))
    check("numbers: with no evidence at all the median is used", by["median"]["count"] == 4)

    # ── general guarantees ──
    check("same shape, order and index; total_filled equals the nulls removed",
          list(out.columns) == list(prod.columns) and out.index.equals(prod.index) and prod.isna().sum().sum() - out.isna().sum().sum() == rep["total_filled"])
    check("every fill reports column, count, method and value", all({"column", "count", "method", "value"} <= set(f) for f in rep["filled"]))
    dup = pd.DataFrame([[1.0, None], [3.0, 5.0], [None, 7.0]], columns=["x", "x"])
    dout, _ = fill_missing(dup, median=True)
    check("duplicate column names are each filled by their own rule", dout.iloc[:, 0].tolist() == [1.0, 3.0, 2.0] and dout.iloc[:, 1].tolist() == [6.0, 5.0, 7.0], str(dout.values.tolist()))
    check("nothing to fill: empty report", fill_missing(pd.DataFrame({"a": range(5)}))[1] == {"filled": [], "skipped": [], "total_filled": 0, "remaining_text": 0})
    src = open("fill_missing.py", encoding="utf-8").read().lower()
    check("no model, provider or network code in the module", not any(w in src for w in ("llm", "groq", "requests", "openai")))

    big = pd.DataFrame({f"c{j}": np.random.RandomState(j).randint(0, 40, 30000).astype(str) for j in range(16)})
    big["id"] = np.arange(30000)
    for j in range(0, 16, 3):
        big.loc[big.sample(300, random_state=j).index, f"c{j}"] = None
    t0 = time.time()
    fill_missing(big)
    check("30,000 rows x 17 columns with blanks fills in a few seconds", time.time() - t0 < 8, f"{time.time() - t0:.1f}s")

    # ── run_fill: the working table ──
    import cache_layer
    import fill_service
    import session_cache
    os.environ.pop("CACHE_ENABLED", None)
    cust = [f"C{i:02d}" for i in range(30)]
    n = 120
    src_df = pd.DataFrame({
        "order": range(1, n + 1),
        "customer": [cust[i % 30] for i in range(n)],
        "postal": [f"P{(i % 30):03d}" for i in range(n)],
        "amount": [round(10 + i * 1.5, 2) for i in range(n)],
        "note": [["red", "blue", "green", "pink", "grey", "gold", "teal"][i % 7] for i in range(n)],
        "ordered": ["2024-01-%02d" % (1 + i % 28) for i in range(n)],
    })
    src_df.loc[[5, 40, 77], "postal"] = None                    # provable from the same customer's other rows
    src_df.loc[src_df["customer"] == "C11", "postal"] = None    # C11: nothing known
    src_df.loc[[3, 9, 22, 31, 50], "amount"] = None             # no relationship: median
    src_df.loc[[2, 8, 14, 26], "note"] = None                   # no relationship: no evidence
    src_df.loc[[7, 19], "ordered"] = None                       # a date: never guessed
    buf = io.BytesIO(); src_df.to_csv(buf, index=False); data = buf.getvalue()
    cache_layer.clear_all(); session_cache.clear()
    r = fill_service.run_fill(data, "data.csv", "s1", URL, median=True)
    meth = {(f["column"], f["method"]): f["count"] for f in r["filled"]}
    check("run_fill: postal filled by evidence, amount by median", meth.get(("postal", "inferred")) == 3 and meth.get(("amount", "median")) == 5, str(r["filled"]))
    check("run_fill: text with no evidence and dates stay blank by default",
          r["remainingText"] >= 4 and any(s["column"] == "ordered" for s in r["skipped"]) and not any(f["method"] == "text" for f in r["filled"]))
    check("run_fill: after = before minus what was filled", r["before"]["nullValues"] - r["after"]["nullValues"] == sum(f["count"] for f in r["filled"]))
    check("run_fill: the message says how each kind was filled", "from other rows" in r["message"] and "median" in r["message"])
    hit = session_cache.get("s1", URL)
    check("run_fill: the next question finds the filled table", hit is not None and int(hit.cleaned_df["amount"].isna().sum()) == 0)
    check("run_fill: another session does not see it", session_cache.get("other", URL) is None)
    r2 = fill_service.run_fill(data, "data.csv", "s1", URL)
    check("run_fill again: nothing left to fill from the data, and it says so", r2["filled"] == [] and "no evidence" in r2["message"])
    r3 = fill_service.run_fill(data, "data.csv", "s1", URL, unknown=True)
    left = session_cache.get("s1", URL).cleaned_df
    check("run_fill with the Unknown button: the remaining text blanks get Unknown, nothing else changes",
          r3["remainingText"] == 0 and int(left["note"].isna().sum()) == 0 and (left["note"] == "Unknown").sum() == 4 and left["ordered"].isna().sum() == 2)

    cache_layer.clear_all(); session_cache.clear()
    agent_table = pd.DataFrame({"amount": [1.0, None, 3.0], "amount_is_big": [False, False, True]})
    session_cache.put("s2", URL, "data.csv", data, agent_table)
    r4 = fill_service.run_fill(data, "data.csv", "s2", URL, median=True)
    check("run_fill starts from the agent's table when one exists (its flag column survives)",
          "amount_is_big" in session_cache.get("s2", URL).cleaned_df.columns and r4["after"]["columns"] == 2 and r4["filled"][0]["value"] == 2.0)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
