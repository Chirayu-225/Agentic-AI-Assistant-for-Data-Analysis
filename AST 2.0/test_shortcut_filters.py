"""
test_shortcut_filters.py — shortcuts must never answer a question that
restricts rows (found by the held-out eval: "total amount of overdue
invoices" returned the grand total), and the code-generation schema must
carry exact value spellings. Dev data only — not the held-out files.
Run: python test_shortcut_filters.py
"""
import sys

import pandas as pd

from query_engine import describe_schema, detect_shortcut

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

df = pd.DataFrame({
    "city": ["Pune", "Mumbai", "Pune", "Nagpur"],
    "stage": ["League", "Playoff", "League", "Final"],
    "pts": [3, 5, 2, 7],
    "played_on": pd.to_datetime(["2025-01-03", "2025-02-11", "2025-03-09", "2025-03-20"]),
    "player_name": ["Asha Rao", "Ravi K", "Asha Rao", "Meera S"],
})

print("PART 1 — whole-column questions still use the shortcut")
for q, expect in [("What is the total pts?", "sum()"), ("average of pts", "mean()"),
                  ("How many rows are there?", "len(df)"), ("maximum pts", "max()")]:
    code = detect_shortcut(q, df) or ""
    check(f"shortcut: {q!r}", expect in code, code)

print("\nPART 2 — anything that restricts rows goes to the full pipeline")
for q in ["What is the total pts for Mumbai?",
          "Total pts in the playoff stage",
          "What were total pts in March 2025?",
          "total pts in Q1",
          "average of pts excluding the final",
          "maximum pts since February",
          "How many rows have city Pune?",
          "total pts where pts above 2"]:
    check(f"no shortcut: {q!r}", detect_shortcut(q, df) is None, detect_shortcut(q, df) or "")

codes = pd.DataFrame({"country": ["UK", "US", "UK", "IN"], "state": ["CA", "NY", "CA", "TX"], "sales": [10, 20, 30, 40]})
for q in ["What is the total sales for UK?", "total sales in CA", "average sales for NY"]:
    check(f"2-letter code blocks the shortcut: {q!r}", detect_shortcut(q, codes) is None, detect_shortcut(q, codes) or "")
check("plain whole-column question still uses it", "sum()" in (detect_shortcut("What is the total sales?", codes) or ""))

print("\nPART 3 — schema carries exact value spellings, not personal data")
desc = describe_schema(df)
check("text values listed exactly", "'League'" in desc and "'Playoff'" in desc, desc)
check("date range listed", "2025-01-03 to 2025-03-20" in desc, desc)
check("numbers marked as numbers", "pts (number)" in desc)
check("personal column values withheld", "Asha" not in desc and "personal" in desc, desc)
named = pd.DataFrame({"name": ["Asha", "Ravi"], "product_name": ["Pen", "Ink"]})
d2 = describe_schema(named)
check("a bare 'name' column is withheld", "Asha" not in d2, d2)
check("product_name values are still listed", "'Pen'" in d2, d2)
many = pd.DataFrame({"note": [f"n{i}" for i in range(40)]})
check("high-cardinality text is summarised, not dumped", "40 distinct" in describe_schema(many))

print()
if FAIL:
    print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("DONE — all checks passed.")
