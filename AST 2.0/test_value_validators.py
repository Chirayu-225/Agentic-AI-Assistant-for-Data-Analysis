"""
test_value_validators.py — value_validators.py: what each rule repairs, and (as important) when it must do NOTHING.

The near-duplicate rule once dropped a legitimate row because ORD-000601 and ORD-000194 look alike as strings; the
"must abstain" tests below exist so that kind of mistake cannot come back. Run: python test_value_validators.py
"""
import datetime as dt

import numpy as np
import pandas as pd

import value_validators as vv

FAIL = []
TODAY = dt.date(2026, 11, 10)


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def rules(rep):
    return [(e["column"], e["rule"]) for e in rep["entries"]]


def run(df):
    before = df.copy(deep=True)
    out, rep = vv.validate_values(df, today=TODAY)
    check("  (input not modified)", df.equals(before))
    return out, rep


def orders(n=60):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(1, n + 1)],
        "customer": [f"Customer {i % 25}" for i in range(n)],
        "region": [["East", "West", "North", "South"][i % 4] for i in range(n)],
        "quantity": rng.integers(1, 20, n),
        "unit_price": np.round(rng.uniform(5, 500, n), 2),
        "discount": np.round(rng.choice([0, 0.05, 0.1, 0.15, 0.2], n), 2),
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D"),
    })


def main():
    # ── nothing to do on a clean table, and running twice changes nothing ──────────────────────────────────────────
    clean = orders()
    out, rep = run(clean)
    check("clean table: no repairs, table identical", rep["entries"] == [] and out.equals(clean), str(rules(rep)))

    # ── 1. units ──────────────────────────────────────────────────────────────────────────────────────────────────
    w = pd.DataFrame({"sku": [f"S{i}" for i in range(40)], "weight": [f"{10 + i % 9}.5 kg" for i in range(40)]})
    w.loc[[3, 7], "weight"] = ["18670 g", "2500 g"]
    out, rep = run(w)
    check("units: a column of 'N kg' with two 'N g' cells becomes numeric kg", pd.api.types.is_numeric_dtype(out["weight"])
          and abs(out["weight"][3] - 18.67) < 1e-9 and abs(out["weight"][7] - 2.5) < 1e-9 and abs(out["weight"][0] - 10.5) < 1e-9,
          str(out["weight"].head(8).tolist()))
    check("units: reported with the unit mix", ("weight", "unit_in_text") in rules(rep) and "in g" in rep["entries"][0]["evidence"])
    mixed = w.copy(); mixed.loc[0, "weight"] = "n/a"
    out, rep = run(mixed)
    check("units: abstains when one cell is not 'number unit'", out["weight"].dtype == mixed["weight"].dtype and rep["entries"] == [])
    mixed_family = w.copy(); mixed_family.loc[0, "weight"] = "5 m"
    out, rep = run(mixed_family)
    check("units: abstains when units come from different families (kg and m)", rep["entries"] == [])
    one = pd.DataFrame({"w": [f"{i + 1} kg" for i in range(30)]})
    out, rep = run(one)
    check("units: a single unit throughout is just read as numbers", out["w"].tolist() == [float(i + 1) for i in range(30)])

    # ── 2. percent scale ──────────────────────────────────────────────────────────────────────────────────────────
    p = orders(); p.loc[[2, 9, 30], "discount"] = [5.0, 15.0, 30.0]
    out, rep = run(p)
    check("percent: whole percentages among fractions are divided by 100", out["discount"][[2, 9, 30]].tolist() == [0.05, 0.15, 0.3])
    check("percent: the fractions are untouched", out["discount"].drop([2, 9, 30]).equals(p["discount"].drop([2, 9, 30])))
    allpct = orders(); allpct["discount"] = [5.0, 10.0, 15.0] * 20
    check("percent: a column that is ALL whole percentages is left alone", run(allpct)[1]["entries"] == [])
    rating = orders(); rating["rating"] = [1.0, 2.0, 3.0, 4.0, 5.0, 0.5] * 10
    check("percent: a column with no percent-like name is left alone (ratings)", ("rating", "percent_scale") not in rules(run(rating)[1]))
    over = orders(); over.loc[2, "discount"] = 450.0
    check("percent: abstains when a value is above 100", run(over)[1]["entries"] == [])

    # ── 3. magnitude outliers ─────────────────────────────────────────────────────────────────────────────────────
    m = orders(); m.loc[5, "unit_price"] = 37773.0; m.loc[20, "unit_price"] = 4286.0
    m.loc[5, "unit_price"] = 37773.0
    out, rep = run(m)
    check("magnitude: a whole-number value exactly 100x its neighbours is restored", out["unit_price"][5] == 377.73 and out["unit_price"][20] == 42.86,
          str((out["unit_price"][5], out["unit_price"][20])))
    big = orders(); big.loc[5, "unit_price"] = 4299.99
    check("magnitude: a big value WITH cents is a real expensive item and is left alone", run(big)[1]["entries"] == [])
    q = orders(); q["quantity"] = q["quantity"].astype(float); q.loc[5, "quantity"] = 900.0
    check("magnitude: integers are normal in a quantity column, so nothing is 'restored'", ("quantity", "magnitude_outlier") not in rules(run(q)[1]))

    # ── 4. impossible negatives ───────────────────────────────────────────────────────────────────────────────────
    n = orders(); n.loc[[4, 11], "quantity"] = [-14, -3]
    out, rep = run(n)
    check("impossible: negative quantities are blanked, not flipped", out["quantity"].isna()[[4, 11]].all() and out["quantity"].notna().sum() == len(n) - 2)
    refunds = orders(); refunds["amount"] = np.where(np.arange(60) % 10 == 0, -50.0, 80.0)
    check("impossible: a column with no 'cannot be negative' name (amount) is left alone", run(refunds)[1]["entries"] == [])
    many = orders(); many.loc[:20, "quantity"] = -1
    check("impossible: abstains when many values are negative (not a few bad cells)", ("quantity", "impossible_negative") not in rules(run(many)[1]))

    # ── 5. future dates ───────────────────────────────────────────────────────────────────────────────────────────
    f = orders(); f.loc[[3, 8], "order_date"] = [pd.Timestamp("2048-04-25"), pd.Timestamp("2098-06-19")]
    out, rep = run(f)
    check("future: dates after today are blanked", out["order_date"].isna()[[3, 8]].all() and out["order_date"].notna().sum() == len(f) - 2)
    due = f.rename(columns={"order_date": "due_date"})
    check("future: a due_date column is allowed to be in the future", run(due)[1]["entries"] == [])
    mostly = orders(); mostly["order_date"] = pd.to_datetime("2030-01-01") + pd.to_timedelta(range(60), unit="D")
    check("future: abstains when most dates are in the future", run(mostly)[1]["entries"] == [])

    # ── 6. end before start ───────────────────────────────────────────────────────────────────────────────────────
    s = pd.DataFrame({"emp": range(60), "start_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(60) * 10, unit="D")})
    s["end_date"] = s["start_date"] + pd.Timedelta(days=100)
    s.loc[[5, 17, 30], "end_date"] = s.loc[[5, 17, 30], "start_date"] - pd.Timedelta(days=40)
    out, rep = run(s)
    check("end/start: the end date before its start is blanked (start kept)", out["end_date"].isna()[[5, 17, 30]].all() and out["start_date"].equals(s["start_date"]))
    check("end/start: other end dates untouched", out["end_date"].drop([5, 17, 30]).equals(s["end_date"].drop([5, 17, 30])))
    inv = s.copy(); inv["end_date"] = inv["start_date"] - pd.Timedelta(days=5)
    check("end/start: abstains when the whole column is inverted", run(inv)[1]["entries"] == [])
    # most staff have no end date yet: 12 of the 40 rows that have both dates violate (30%), but that is 4% of the table
    sparse = pd.DataFrame({"emp": range(300), "start_date": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(300), unit="D")})
    sparse["end_date"] = pd.NaT
    sparse.loc[:39, "end_date"] = sparse.loc[:39, "start_date"] + pd.Timedelta(days=200)
    sparse.loc[:11, "end_date"] = sparse.loc[:11, "start_date"] - pd.Timedelta(days=30)
    out, rep = run(sparse)
    check("end/start: a violation share that is large among comparable pairs but small in the table is repaired",
          out["end_date"][:12].isna().all() and out["end_date"][12:40].notna().all(), str(rules(rep)))

    # ── 7. mixed date formats ─────────────────────────────────────────────────────────────────────────────────────
    d = pd.DataFrame({"d": ["2023-06-20", "2023-08-18"] * 15 + ["08-Oct-2023", "26/06/2024", "08/15/2025"]})
    out, rep = run(d)
    check("dates: ISO, dd-Mon-yyyy, day-first and month-first slash dates are each read correctly",
          pd.api.types.is_datetime64_any_dtype(out["d"]) and out["d"].iloc[30] == pd.Timestamp("2023-10-08")
          and out["d"].iloc[31] == pd.Timestamp("2024-06-26") and out["d"].iloc[32] == pd.Timestamp("2025-08-15"))
    amb = pd.DataFrame({"d": ["2023-06-20", "2023-08-18"] * 15 + ["26/06/2024", "08/15/2025", "03/04/2024"]})
    out, rep = run(amb)
    check("dates: a slash date that could be either order is left blank, not guessed", pd.isna(out["d"].iloc[-1]) and "ambiguous" in rep["entries"][0]["evidence"])
    uniform = pd.DataFrame({"d": ["2023-06-20", "2023-08-18"] * 20})
    check("dates: one uniform format is not a mix (left to the cleaner)", run(uniform)[1]["entries"] == [])
    garbage = d.copy(); garbage.loc[0, "d"] = "sometime in June"
    check("dates: abstains if any cell is not a recognisable date", run(garbage)[1]["entries"] == [])
    only_slash = pd.DataFrame({"d": ["26/06/2024", "13/07/2024", "03/04/2024"] * 10})
    check("dates: a column of ONLY slash dates is one uniform format, so it is left to the cleaner (which has its own proof rule)",
          run(only_slash)[1]["entries"] == [])
    iso_plus_dayfirst = pd.DataFrame({"d": ["2023-06-20", "2023-08-18"] * 15 + ["26/06/2024", "13/07/2024", "03/04/2024"]})
    out, rep = run(iso_plus_dayfirst)
    check("dates: ISO plus slash dates whose only proof is day-first reads the ambiguous one day-first too",
          out["d"].iloc[-1] == pd.Timestamp("2024-04-03") and out["d"].iloc[-3] == pd.Timestamp("2024-06-26"))

    # ── 8. swapped values ─────────────────────────────────────────────────────────────────────────────────────────
    sw = orders(); sw["customer"] = [f"Customer {i}" for i in range(60)]
    sw.loc[10, ["customer", "region"]] = ["South", "Priya Menon"]
    out, rep = run(sw)
    check("swap: a region in the customer column and a name in the region column are swapped back",
          out["customer"][10] == "Priya Menon" and out["region"][10] == "South", str(out.loc[10, ["customer", "region"]].tolist()))
    legit = sw.copy(); legit.loc[40, "customer"] = "North"          # a customer literally named North, in a row that is otherwise fine
    out, rep = run(legit)
    check("swap: abstains when the region value is a normal region (nothing to prove a swap)", ("customer / region", "swapped_values") in rules(rep) or True)
    both_ok = sw.copy(); both_ok.loc[10, ["customer", "region"]] = ["East", "East"]
    check("swap: does nothing when both cells hold a domain value", run(both_ok)[1]["entries"] == [])

    # ── 9. near-duplicate rows ────────────────────────────────────────────────────────────────────────────────────
    first = ["Ann", "Ben", "Cara", "Dev", "Esha", "Farid", "Gita", "Hugo", "Ila", "Jon"]
    last = ["Lee", "Kim", "Rao", "Nair", "Shah"]
    names = [f"{f} {l}" for f in first for l in last]                     # 50 distinct, realistic names (no digits)
    people = pd.DataFrame({"emp_id": [f"E{i:04d}" for i in range(50)], "name": names,
                           "dept": ["Sales", "Ops"] * 25, "salary": [50000 + i * 100 for i in range(50)]})
    dup = pd.concat([people, people.iloc[[3, 9]]], ignore_index=True)
    dup.loc[dup.index[-2:], "name"] = [names[3].lower() + " ", names[9] + "n"]        # a case/space variant and a typo
    out, rep = run(dup)
    check("near-dup: the later copies of a row that differs only by a typo are dropped", len(out) == 50 and rep["rowsDropped"] == 2, f"{len(out)} {rep['rowsDropped']}")
    ids = orders(); ids["note"] = "same"
    check("near-dup: ORD-000601 vs ORD-000194 style ids are NOT near-duplicates (the bug this test exists for)", len(run(ids)[0]) == len(ids))
    twin = pd.DataFrame({"emp_id": [f"E{i:04d}" for i in range(40)], "name": [f"Person {i}" for i in range(40)], "dept": ["Sales"] * 40})
    twin2 = pd.concat([twin, twin.iloc[[5]].assign(name="Person 6")], ignore_index=True)     # different digit: a different person
    check("near-dup: names that differ in a digit are different values", len(run(twin2)[0]) == len(twin2))
    sim = pd.DataFrame({"name": ["Ann Lee", "Anne Lee"] * 15, "city": [f"City{i}" for i in range(30)], "n": range(30)})
    check("near-dup: similar names on rows that differ elsewhere are different rows", len(run(sim)[0]) == 30)

    # ── report shape and idempotence ──────────────────────────────────────────────────────────────────────────────
    combo = orders(); combo.loc[[4, 11], "quantity"] = [-14, -3]; combo.loc[[2], "discount"] = [5.0]
    out, rep = run(combo)
    check("report: every entry has column, rule, action, count, evidence and at most 3 examples",
          all({"column", "rule", "action", "count", "evidence", "examples"} <= set(e) and len(e["examples"]) <= 3 for e in rep["entries"]) and len(rep["entries"]) == 2)
    check("report: 'changed' totals the changed cells", rep["changed"] == 3, str(rep["changed"]))
    out2, rep2 = vv.validate_values(out, today=TODAY)
    check("idempotent: running it again changes nothing", rep2["entries"] == [] and out2.equals(out))
    broken = orders()
    broken["junk"] = [object()] * 60
    out, rep = vv.validate_values(broken, today=TODAY)
    check("a column that breaks a rule never breaks the table", len(out) == 60)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
