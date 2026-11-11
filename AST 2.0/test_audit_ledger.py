"""audit_ledger: every change is listed with its row, and an unlisted change is caught. No model, no network."""
import numpy as np
import pandas as pd
import audit_ledger as al
import cleaning
from fill_missing import fill_missing, fill_derived
from value_validators import validate_values

FAILS = []
def check(label, cond, detail=""):
    print(("[OK] " if cond else "[FAIL] ") + label + ("" if cond else f"  {detail}"))
    if not cond: FAILS.append(label)

rng = np.random.default_rng(5)
n = 120
raw = pd.DataFrame({
    "Order ID": [f"ORD-{i:04d}" for i in range(n)],
    "Customer": [f"C{i % 20:02d}" for i in range(n)],
    "Country": ["USA", "Canada", "Germany", "USA"] * 30,
    "Qty": rng.integers(1, 10, n),
    "Price": np.round(rng.uniform(5, 50, n), 2),
})
raw["Amount"] = (raw["Qty"] * raw["Price"]).round(2)
raw["Discount Pct"] = rng.choice([0, 5, 10, 20], n).astype(float)
raw["Ordered"] = [f"2025-03-{1 + i % 27:02d}" for i in range(n)]
raw["Shipped"] = [f"2025-03-{2 + i % 27:02d}" if i % 27 < 26 else f"2025-03-{1 + i % 27:02d}" for i in range(n)]
raw = raw.astype(object)
raw.loc[[3, 9], "Country"] = ["  USA ", "GERMANY"]                         # whitespace / case variants
raw.loc[[5], "Amount"] = -999                                              # sentinel
raw.loc[[11, 12], "Discount Pct"] = [0.05, 0.1]                            # fractions among whole percents
raw.loc[[20], "Qty"] = -4                                                  # impossible negative
raw = pd.concat([raw, raw.iloc[[30]]], ignore_index=True)                  # an exact copy of row index 30 (file row 32)
raw.loc[len(raw) - 1, "Order ID"] = "ORD-0030"

clean, log = cleaning.rule_based_clean(raw)
check("the cleaner records where each row came from", len(clean.attrs.get("source_rows", [])) == len(clean))

L = al.Ledger.for_clean(raw, clean)
L.start(clean)
f, fr = fill_missing(clean)
L.advance("Fill from evidence", f, rules={e["column"]: "Filled from other rows" for e in fr["filled"]})
v, vr = validate_values(f)
L.advance("Value repairs", v, rules={e["column"]: e["rule"] for e in vr["entries"]})
d, de = fill_derived(v)
L.advance("Exact arithmetic", d, default_rule="Computed from other columns")
S = L.summary(d)
page = L.page(limit=500)["entries"]
def find(**kw):
    return [e for e in page if all(e[k] == v for k, v in kw.items())]

check("the duplicate row is listed, at its file row, as a copy of the first",
      any(e["action"] == "removed" and e["row"] == n + 2 and "row 32" in e["after"].lower() for e in page), str([e for e in page if e["action"] == "removed"]))
check("whitespace/case standardisation lists the exact rows", {e["row"] for e in page if e["column"] == "country" and e["rule"].startswith(("Spacing", "Spelling"))} >= {5, 11})
sent = find(column="amount", row=7)
check("the -999 placeholder shows its row, before and after", len(sent) >= 1 and sent[0]["before"] in ("-999", "-999.0") and sent[0]["action"] in ("blanked", "filled"), str(sent))
check("the discount fractions are listed with before and after",
      {(e["row"], e["before"], e["after"]) for e in page if e["column"] == "discount_pct" and e["rule"] == "percent_scale"} == {(13, "0.05", "5.0"), (14, "0.1", "10.0")},
      str([e for e in page if e["column"] == "discount_pct"]))
check("the impossible negative quantity is listed", any(e["column"] == "qty" and e["row"] == 22 and e["before"] in ("-4", "-4.0") for e in page), str([e for e in page if e["column"] == "qty"]))
check("every parsed column is one column-level entry, not one per cell", all(c["to"] in ("dates", "numbers") for c in S["conversions"]))
check("totals add up to the entries", S["totals"]["rowsRemoved"] == 1 and S["totals"]["rowsRaw"] == n + 1 and S["totals"]["rowsFinal"] == n)
check("nothing is unexplained when every stage is recorded", S["unexplained"]["checked"] and S["unexplained"]["count"] == 0, str(S["unexplained"]))
check("the key column is the order id", L.key_col == "order_id" and any(e["key"] == "ORD-0030" for e in page))

# an unlisted change is caught
sneaky = d.copy(); sneaky.loc[40, "price"] = 999.0
check("a change no stage recorded is reported as unexplained", L.reconcile(sneaky)["count"] == 1 and L.reconcile(sneaky)["examples"][0]["row"] == 42, str(L.reconcile(sneaky)))
gone = d.drop(index=[50])
check("a row that vanished without a record is reported", L.reconcile(gone)["count"] == 1)

# duplicates that stay (the cleaner keeps identical rows when nothing proves they are mistakes)
same = pd.DataFrame({"item": ["bolt", "nut", "bolt", "washer", "nut", "bolt"] * 6, "qty": [1, 2, 1, 3, 4, 1] * 6})
same = pd.concat([same, pd.DataFrame({"item": ["gear"] * 2, "qty": [9, 9]})], ignore_index=True)
cs, _ = cleaning.rule_based_clean(same)
LD = al.Ledger.for_clean(same, cs); LD.start(cs)
G = LD.summary(cs)["duplicates"]
check("identical rows that stay are grouped with their file rows", G["groups"] >= 1 and any(g["rows"][:2] == [2, 4] for g in G["sample"]), str(G)[:300])
check("each group says how many copies and shows the values", all(g["count"] >= 2 and "item" in g["values"] for g in G["sample"]))

# page filters and CSV
check("filter by action", all(e["action"] == "removed" for e in L.page(action="removed")["entries"]))
check("filter by column", all(e["column"] == "country" for e in L.page(column="country")["entries"]))
check("search by key or value", L.page(q="ORD-0030")["total"] >= 1)
csv_text = L.to_csv().splitlines()
check("the CSV has a header and one line per stored entry", csv_text[0].startswith("stage,rule,action,row") and len(csv_text) == len(L.entries) + 1)

# header promotion
hp = pd.DataFrame([["a", "b"], ["x", "1"], ["y", "2"], ["z", "3"]] * 5)
hp.columns = ["Unnamed: 0", "Unnamed: 1"]
c2, _ = cleaning.rule_based_clean(hp)
L2 = al.Ledger.for_clean(hp, c2); L2.start(c2)
check("a promoted header row does not break the audit", L2.summary(c2)["unexplained"]["count"] == 0, str(L2.summary(c2)["unexplained"]))

# a clean table has an empty trail
tidy = pd.DataFrame({"id": [f"A{i}" for i in range(30)], "v": range(30)})
ct, _ = cleaning.rule_based_clean(tidy)
L3 = al.Ledger.for_clean(tidy, ct); L3.start(ct)
S3 = L3.summary(ct)
check("a clean file has no entries and nothing unexplained", S3["entryCount"] == 0 and S3["unexplained"]["count"] == 0, str(S3["totals"]))
print("\nDONE — all checks passed." if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
raise SystemExit(1 if FAILS else 0)
