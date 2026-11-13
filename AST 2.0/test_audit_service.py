"""audit_service end to end on real file bytes: the trail, the page filters, the CSV, and the duplicate choice."""
import io
import numpy as np
import pandas as pd
import audit_ledger as al
import audit_service as svc
import cache_layer
import session_cache

FAILS = []
def check(label, cond, detail=""):
    print(("[OK] " if cond else "[FAIL] ") + label + ("" if cond else f"  {detail}"))
    if not cond: FAILS.append(label)

rng = np.random.default_rng(11)
n = 200
df = pd.DataFrame({
    "order_id": [f"ORD-{i:05d}" for i in range(n)],
    "customer_id": [f"C{i % 25:03d}" for i in range(n)],
    "country": ["USA", "Canada", "Germany", "USA", "Namibia"] * 40,
    "state": ["CA", "ON", "BY", "CA", "KH"] * 40,
    "quantity": rng.integers(1, 12, n),
    "unit_price": np.round(rng.uniform(5, 100, n), 2),
})
df["amount"] = (df.quantity * df.unit_price).round(2)
df["discount_pct"] = rng.choice([0, 5, 10, 15], n).astype(float)
df["order_date"] = [f"2025-04-{1 + i % 28:02d}" for i in range(n)]
df["ship_date"] = [f"2025-04-{2 + i % 28:02d}" if i % 28 < 27 else f"2025-04-{1 + i % 28:02d}" for i in range(n)]
dirty = df.copy().astype(object)
dirty.loc[[4, 8], "country"] = ["  usa", "CANADA"]
dirty.loc[10, "amount"] = -999
dirty.loc[[15, 16], "discount_pct"] = [0.05, 0.1]
dirty.loc[20, "state"] = ""
dirty = pd.concat([dirty, dirty.iloc[[3, 7]]], ignore_index=True)                  # two accidental copies (same order id)
buf = io.StringIO(); dirty.to_csv(buf, index=False)
content = buf.getvalue().encode()
S, U = "sess1", "http://x/orders.csv"
cache_layer.clear_all(); session_cache.clear()

L, summ = svc.build_audit(content, "orders.csv", S, U, "clean")
page = L.page(limit=500)["entries"]
check("scope clean: the two copies are listed as removed, with the row they copy",
      sorted(e["row"] for e in page if e["action"] == "removed") == [n + 2, n + 3] and all(e["rule"] in ("Repeated record (same ID) removed", "Exact duplicate removed") and "row" in e["after"] for e in page if e["action"] == "removed"), str([e for e in page if e["action"] == "removed"]))
check("scope clean: nothing unexplained", summ["unexplained"]["checked"] and summ["unexplained"]["count"] == 0, str(summ["unexplained"]))
check("scope clean: the whitespace/case rows are listed", {e["row"] for e in page if e["column"] == "country"} >= {6, 10})
check("the summary lists the rule, column and count for each kind of change", all({"rule", "column", "count", "stage", "action"} <= set(r) for r in summ["byRule"]) and sum(r["count"] for r in summ["byRule"]) == len(page))

L2, summ2 = svc.build_audit(content, "orders.csv", S, U, "clean")
check("a second call for the same file is served from the store", L2 is L)

La, sa = svc.build_audit(content, "orders.csv", S, U, "agent")
pa = La.page(limit=500)["entries"]
check("scope agent: the fill and the repairs add their own stages", set(sa["stages"]) >= {"Rule cleaning", "Fill from evidence", "Value repairs"}, str(sa["stages"]))
check("scope agent: the discount fractions are listed with their rows", {(e["row"], e["before"], e["after"]) for e in pa if e["column"] == "discount_pct" and e["stage"] == "Value repairs"} == {(17, "0.05", "5.0"), (18, "0.1", "10.0")}, str([e for e in pa if e["column"] == "discount_pct"]))
check("scope agent: the amount placeholder is blanked then computed exactly, both listed",
      any(e["column"] == "amount" and e["row"] == 12 and e["stage"] == "Rule cleaning" for e in pa) and any(e["column"] == "amount" and e["row"] == 12 and e["stage"] == "Exact arithmetic" for e in pa), str([e for e in pa if e["row"] == 12]))
check("scope agent: nothing unexplained", sa["unexplained"]["count"] == 0, str(sa["unexplained"]))
check("scope agent: the evidence for a repair is attached to its rule", any(r["evidence"] for r in sa["byRule"] if r["stage"] == "Value repairs"))

csv_text = svc.audit_csv(content, "orders.csv", S, U, "agent")
check("the CSV is the whole change list", csv_text.count("\n") == len(La.entries) + 1)

# a working table the AI / the user's choices changed
work = La  # placeholder to keep the name used below clear
final_before = svc.build_audit(content, "orders.csv", S, U, "agent")[1]["totals"]["rowsFinal"]
w = session_cache  # the audit reads the session's working table
from fill_missing import fill_derived, fill_missing
from value_validators import validate_values
from cleaning import rule_based_clean
from file_reader import read_tabular_file
raw = read_tabular_file(content, "orders.csv")
clean, _ = cache_layer.cached_rule_clean(raw, content, rule_based_clean)
d, _ = fill_derived(validate_values(fill_missing(clean)[0])[0])
changed = d.copy(); changed.loc[30, "country"] = "Atlantis"; changed["amount_is_outlier"] = False
session_cache.put(S, U, "orders.csv", content, changed)
Lw, sw = svc.build_audit(content, "orders.csv", S, U, "agent", force=True)
pw = Lw.page(limit=500)["entries"]
check("the AI step's change is listed under its own stage", any(e["stage"] == "AI fix and your choices" and e["column"] == "country" and e["after"] == "Atlantis" and e["row"] == 32 for e in pw), str([e for e in pw if e["stage"].startswith("AI")]))
check("a flag column the AI added is listed as added", "amount_is_outlier" in sw["totals"]["columnsAdded"])
check("still nothing unexplained", sw["unexplained"]["count"] == 0, str(sw["unexplained"]))

# the duplicate choice: identical rows with no id proving they are mistakes
rows = pd.DataFrame({"item": ["bolt", "nut", "washer", "gear"] * 10, "qty": [1, 2, 3, 4] * 10})
rows = pd.concat([rows, rows.iloc[:6]], ignore_index=True)
b2 = io.StringIO(); rows.to_csv(b2, index=False); c2 = b2.getvalue().encode()
S2, U2 = "sess2", "http://x/items.csv"
L3, s3 = svc.build_audit(c2, "items.csv", S2, U2, "agent")
check("kept identical rows are grouped with their file rows", s3["duplicates"]["groups"] >= 1 and s3["duplicates"]["extra"] >= 6, str(s3["duplicates"])[:300])
out = svc.remove_duplicates(c2, "items.csv", S2, U2)
check("the duplicate answer carries the table's counts before and after",
      out["before"]["rows"] - out["after"]["rows"] == out["removed"] and out["after"]["rows"] == out["rows"]
      and out["after"]["nullValues"] <= out["before"]["nullValues"], str(out))
check("removing keeps the first copy of each and reports the count", out["success"] and out["removed"] > 0 and out["rows"] == len(rows) - out["removed"], str(out))
wt = session_cache.get(S2, U2).cleaned_df
check("the working table has no identical rows left", not wt.duplicated().any())
L4, s4 = svc.build_audit(c2, "items.csv", S2, U2, "agent")
rem = [e for e in L4.page(limit=500)["entries"] if e["action"] == "removed"]
check("the removal is in the audit, as the user's choice, with its rows", len(rem) == out["removed"] and all("your choice" in e["rule"].lower() and e["after"].startswith("copy of row") and e["key"] for e in rem), str(rem[:2]))
check("... and nothing is unexplained", s4["unexplained"]["count"] == 0, str(s4["unexplained"]))
check("no identical rows are left in the summary", s4["duplicates"]["groups"] == 0, str(s4["duplicates"]))
none = svc.remove_duplicates(c2, "items.csv", S2, U2)
check("asking again removes nothing", none["removed"] == 0)

print("\nDONE — all checks passed." if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
raise SystemExit(1 if FAILS else 0)
