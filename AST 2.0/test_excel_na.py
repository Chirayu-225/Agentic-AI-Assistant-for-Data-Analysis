"""Excel keeps a real "NA" under the safe policy (like CSV); the legacy policy is unchanged. Number blanks are flagged numeric."""
import io
import pandas as pd
import file_reader as fr
from fill_missing import fill_missing
FAILS = []
def check(label, cond, detail=""):
    print(("[OK] " if cond else "[FAIL] ") + label + ("" if cond else f"  {detail}"))
    if not cond: FAILS.append(label)

n = 40
df = pd.DataFrame({"id": range(n), "territory": ["NA" if i % 2 else "EMEA" for i in range(n)],
                   "note": [None if i % 5 == 0 else "x" for i in range(n)], "v": [float(i) for i in range(n)]})
b = io.BytesIO(); df.to_excel(b, index=False); one = b.getvalue()
leg = fr._read_tabular_file_uncached(one, "t.xlsx")
saf = fr._read_tabular_file_uncached(one, "t.xlsx", na_policy="safe")
check("legacy Excel read is unchanged (NA is missing)", (leg.territory == "NA").sum() == 0 and leg.territory.isna().sum() == 20)
check("safe Excel read keeps real NA", (saf.territory == "NA").sum() == 20 and saf.territory.isna().sum() == 0)
check("safe: genuine blanks stay missing", saf.note.isna().sum() == 8)
check("safe: numbers and dtypes unchanged", saf.v.tolist() == leg.v.tolist() and str(saf.id.dtype) == str(leg.id.dtype))

b = io.BytesIO()
with pd.ExcelWriter(b) as w:
    df.to_excel(w, sheet_name="orders", index=False)
    pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="summary", index=False)
many = fr._read_tabular_file_uncached(b.getvalue(), "t.xlsx", na_policy="safe")
check("multi-sheet workbook: the picked sheet keeps real NA", (many.territory == "NA").sum() == 20)

num = pd.DataFrame({"qty": [1.0, None, 3.0] * 10})
rep = fill_missing(num)[1]
check("a number blank with no evidence is marked numeric (not an 'Unknown' question)", rep["skipped"][0]["kind"] == "no_evidence" and rep["skipped"][0].get("numeric") is True, str(rep["skipped"]))
txt = pd.DataFrame({"c": ["a", None, "b"] * 10})
check("a text blank is not marked numeric", "numeric" not in fill_missing(txt)[1]["skipped"][0])
print("\nDONE — all checks passed." if not FAILS else f"\n{len(FAILS)} FAILED")
raise SystemExit(1 if FAILS else 0)
