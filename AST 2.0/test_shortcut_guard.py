"""
test_shortcut_guard.py — the no-LLM shortcut layer must answer only plain
whole-column questions. Before this fix, "list the regions ranked by total
revenue" silently returned the grand total (one number) with HIGH confidence.

Run: python3 test_shortcut_guard.py
"""
import os, sys
import pandas as pd
os.environ.setdefault("AUDIT_LOG_DISABLED", "1")
from query_engine import detect_shortcut

df = pd.DataFrame({"region": ["N", "S"], "revenue": [1, 2], "units": [3, 4], "order_date": ["a", "b"]})
FAIL = []
cases = [  # (question, should a shortcut fire?)
    ("What is the total revenue?", True), ("sum of revenue", True), ("average of units", True),
    ("highest revenue", True), ("what is the lowest units", True), ("How many rows are in the dataset?", True),
    ("list the regions ranked by total revenue", False), ("which region has the highest total revenue", False),
    ("total revenue by region", False), ("sum of revenue per region", False), ("average units for each region", False),
    ("total revenue and units", False), ("top 3 regions by revenue", False), ("compare revenue across regions", False),
    ("Plot total revenue by region as a bar chart", False),
]
for q, want in cases:
    got = detect_shortcut(q, df) is not None
    ok = got == want
    print(f"[{'OK' if ok else 'FAIL'}] {q!r} -> shortcut={'yes' if got else 'no'} (want {'yes' if want else 'no'})")
    if not ok:
        FAIL.append(q)
check = detect_shortcut("What is the total revenue?", df)
if check != "result = df['revenue'].sum()":
    FAIL.append("wrong code"); print("[FAIL] wrong generated code:", check)
print("=" * 70)
if FAIL:
    print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
print("DONE — all checks passed.")
