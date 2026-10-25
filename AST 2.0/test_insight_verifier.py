"""
test_insight_verifier.py — insight_verifier.py: number matching, numbers
stored as text, and the wording checks added after the insight eval.
No LLM needed. Run: python test_insight_verifier.py
"""
import sys

import pandas as pd

from insight_verifier import verify_finding as v

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

rev = pd.Series({"South": 24300.0, "East": 19500.0, "West": 14100.0})

print("PART 1 — numbers")
check("matching figures verify", v("South leads with $24,300 in revenue", rev)["verified"])
check("an invented figure is caught", not v("South leads with $31,000", rev)["verified"])
check("a share written as a percent matches a computed fraction", v("Phones are 45% of units", 0.4512)["verified"])
check("an abbreviated figure matches ($1.2M)", v("Revenue reached $1.2M", 1204331.0)["verified"])
check("a finding with no number at all is not verified", not v("Revenue was strong", 5)["verified"])

print("\nPART 2 — numbers stored as text in table cells")
text_table = pd.DataFrame({"segment": ["Enterprise", "SMB"], "largest": ["₹95,235.87", "Rs. 9,340.85"]})
check("₹95,236 matches the text cell '₹95,235.87'", v("The largest Enterprise invoice was ₹95,236", text_table)["verified"])
check("a figure absent from the text cells is still caught", not v("The largest invoice was ₹99,000", text_table)["verified"])

print("\nPART 3 — ratio words")
months = pd.Series({"Apr": 190989.37, "Jul": 362813.15})
r = v("Monthly expenses grew from 190,989.37 in April to 362,813.15 in July, more than doubling", months)
check("'more than doubling' at 1.90x is rejected", not r["verified"], str(r))
check("'nearly doubling' style wording at 1.90x passes as 'double'",
      v("Monthly expenses grew from 190,989.37 in April to 362,813.15 in July, almost doubling", months)["verified"])
check("a true 'tripled' passes", v("Units tripled from 100 to 310", pd.Series([100, 310]))["verified"])
check("a false 'tripled' is rejected", not v("Units tripled from 100 to 250", pd.Series([100, 250]))["verified"])
check("'half' with only one figure is rejected", not v("West makes half the revenue at 14,100", rev)["verified"])

print("\nPART 4 — qualifiers one figure can't establish")
r = v("Ten stores consistently fall below the median of 6,200", pd.Series([6200.0]))
check("'consistently' is rejected", not r["verified"], str(r))
check("'every month' is rejected", not v("Sales beat 24,300 every month", rev)["verified"])
check("'majority' needs a share above 50%", not v("The majority of revenue (42%) is from South", 0.42)["verified"])
check("'majority' with 58% passes", v("The majority of revenue (58%) is from South", 0.58)["verified"])

print("\nPART 5 — currency and data artefacts")
inr = {"₹ (INR)"}
check("₹ figure passes when the data is in rupees", v("South billed ₹24,300", rev, inr)["verified"])
check("$ figure is rejected when the data is in rupees", not v("South billed $24,300", rev, inr)["verified"])
check("'Rs. 24,300' counts as rupees", v("South billed Rs. 24,300", rev, inr)["verified"])
check("any currency symbol is rejected when the data has none", not v("South billed $24,300", rev, set())["verified"])
check("no currency check when currencies is None", v("South billed $24,300", rev)["verified"])
check("a finding reporting NaN is rejected", not v("Growth was 24,300 but many entries are NaN", rev)["verified"])
check("a raw -1.000 ratio is rejected", not v("Store fell -1.000 to 24,300", rev)["verified"])

print()
if FAIL:
    print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("DONE — all checks passed.")
