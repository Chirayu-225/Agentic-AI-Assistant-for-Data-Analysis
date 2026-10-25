"""
test_cleaning_realworld.py — dev-set reproductions of the cleaning gaps the
held-out eval exposed (eval_data/heldout/). Deliberately uses DIFFERENT data
from the held-out files, so these fixes are tested on the general problem,
not tuned to the held-out answers.

Run: python test_cleaning_realworld.py
"""
import sys

import numpy as np
import pandas as pd

from cleaning import rule_based_clean

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


print("PART 1 — mixed currency formats in one column")
df = pd.DataFrame({
    "order_no": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "fee": ["₹1,20,000.50", "Rs 2,500", "INR 300", "1500", "(₹2,000)", "Rs. 75.25"],
})
c, log = rule_based_clean(df)
check("money column becomes numeric", pd.api.types.is_numeric_dtype(c["fee"]), str(c["fee"].tolist()))
check("lakh separators parsed (₹1,20,000.50 → 120000.5)", c["fee"].iloc[0] == 120000.5, str(c["fee"].iloc[0]))
check("Rs / INR prefixes stripped", list(c["fee"].iloc[1:4]) == [2500, 300, 1500], str(c["fee"].tolist()))
check("accounting negative (₹2,000) → -2000", c["fee"].iloc[4] == -2000, str(c["fee"].iloc[4]))
check("sum is right", abs(c["fee"].sum() - (120000.5 + 2500 + 300 + 1500 - 2000 + 75.25)) < 1e-6)

print("\nPART 2 — total rows are excluded")
df = pd.DataFrame({"branch": ["Pune", "Nagpur", "Nashik", "Grand Total"], "sales": [10, 20, 30, 60]})
c, log = rule_based_clean(df)
check("Grand Total row removed", len(c) == 3 and c["sales"].sum() == 60, str(c.to_dict("list")))
df = pd.DataFrame({"name": ["Total Care Plan", "Basic"], "price": [10, 5]})
c, _ = rule_based_clean(df)
check("a real value that merely contains 'Total' is kept", len(c) == 2)

print("\nPART 3 — spelling variants merge, distinct values don't")
df = pd.DataFrame({
    "city": ["Pune", "pune", "PUNE ", "Mumbai", "Mumbai", "mumbai", "Pu.", "Delhi", "Delhi", "Delhi"],
    "tier": ["Gold", "gold", "Silver", "Silver", "Gold", "GOLD", "Silver", "Bronze", "Bronze", "Gold"],
    "units": range(10),
})
c, log = rule_based_clean(df)
check("case/space variants merged", set(c["city"]) == {"Pune", "Mumbai", "Delhi"}, str(sorted(set(c["city"]))))
check("dotted abbreviation with one match merged ('Pu.' → 'Pune')", (c["city"] == "Pune").sum() == 4)
check("tier variants merged", set(c["tier"]) == {"Gold", "Silver", "Bronze"}, str(set(c["tier"])))
df = pd.DataFrame({"city": ["M.", "Mumbai", "Mysuru", "Mumbai", "Mysuru", "Delhi"]})
c, _ = rule_based_clean(df)
check("ambiguous abbreviation ('M.' could be Mumbai or Mysuru) left alone", "M." in set(c["city"]))
df = pd.DataFrame({"note": [f"customer said {i} things" for i in range(40)]})
c, _ = rule_based_clean(df)
check("free text is not treated as a category", c["note"].nunique() == 40)

print("\nPART 4 — mixed date formats")
df = pd.DataFrame({"shipped": ["2025-11-02", "14 Dec 2025", "2025/12/31", "Jan 5, 2026", "2026-01-20"]})
c, log = rule_based_clean(df)
check("column becomes datetime", pd.api.types.is_datetime64_any_dtype(c["shipped"]), str(c["shipped"].tolist()))
check("every format parsed", c["shipped"].notna().all() and c["shipped"].dt.year.tolist() == [2025, 2025, 2025, 2026, 2026])
df = pd.DataFrame({"code": ["AX-12", "BX-99", "CX-07", "DX-31"]})
c, _ = rule_based_clean(df)
check("codes are not mistaken for dates", not pd.api.types.is_datetime64_any_dtype(c["code"]))

print("\nPART 5 — percentages keep their unit")
df = pd.DataFrame({"margin": ["12%", "8%", "15%", "5%"]})
c, _ = rule_based_clean(df)
check("12% → 12 (percent points, not 0.12)", c["margin"].tolist() == [12, 8, 15, 5], str(c["margin"].tolist()))

print("\nPART 6 — missing values are not invented")
df = pd.DataFrame({"days": [10.0, None, 30.0, None], "owner": ["Asha", "Ravi", None, "Asha"]})
c, log = rule_based_clean(df)
check("blank numbers stay blank (no median fill)", int(c["days"].isna().sum()) == 2, str(c["days"].tolist()))
check("blank text stays blank (no 'Unknown')", int(c["owner"].isna().sum()) == 1 and "Unknown" not in set(c["owner"].dropna()))
check("log says where the blanks are", any("missing value" in l for l in log))

print("\nPART 7 — repeated records")
df = pd.DataFrame({
    "claim_id": ["C1", "C2", "C3", "C2", "C4"],
    "dept": ["Finance", "Legal", "Ops", "LEGAL", "Ops"],
    "kind": ["Taxi", "Hotel", "Meals", None, "Taxi"],
    "amount": ["100", "250", "40", "250", "60"],
})
c, log = rule_based_clean(df)
check("same claim submitted twice (case + blank cell differ) kept once", len(c) == 4 and (c["claim_id"] == "C2").sum() == 1, str(c.to_dict("list")))
check("the more complete copy was kept", c.loc[c["claim_id"] == "C2", "kind"].iloc[0] == "Hotel")
df = pd.DataFrame({"employee_id": [7, 7, 8], "amount": [10, 20, 30]})
c, _ = rule_based_clean(df)
check("a non-unique ID (one employee, many claims) never triggers removal", len(c) == 3)
df = pd.DataFrame({"claim_id": ["C1", "C1"], "amount": ["100", "999"]})
c, _ = rule_based_clean(df)
check("same ID with conflicting values is NOT silently dropped", len(c) == 2)

print()
if FAIL:
    print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("DONE — all checks passed.")
