"""The x1000 question: detection, the guards that keep it quiet, and the apply step."""
import numpy as np
import pandas as pd
import scale_review as sr

FAILS = []
def check(label, cond, detail=""):
    print(("[OK] " if cond else "[FAIL] ") + label + ("" if cond else f"  {detail}"))
    if not cond: FAILS.append(label)

rng = np.random.default_rng(3)
n = 400
q = rng.integers(1, 21, n)
df = pd.DataFrame({"order_id": [f"O{i:05d}" for i in range(n)], "quantity": q.copy(), "region": ["a", "b"] * (n // 2)})
bad = [5, 50, 77, 120, 300, 301]
for i in bad:
    df.loc[i, "quantity"] = int(q[i]) * 1000

s = sr.find_scale_suspects(df)
check("finds the quantity column", len(s) == 1 and s[0]["column"] == "quantity", str(s))
check("counts the planted values", s and s[0]["count"] == len(bad), str(s))
check("examples show the row as a spreadsheet does and the suggestion",
      s and s[0]["examples"][0]["row"] == bad[0] + 2 and s[0]["examples"][0]["suggested"] == int(q[bad[0]]), str(s))
check("typical value is the ordinary one", s and 5 <= s[0]["typical"] <= 15, str(s))

# quiet cases
ids = pd.DataFrame({"customer_id": list(range(1000, 1400)), "x": 1})
check("an id column is left alone", sr.find_scale_suspects(ids) == [])
prices = pd.DataFrame({"price": rng.choice([1000, 2000, 3000, 4000, 5000], 400)})
check("a column that is all thousands is left alone", sr.find_scale_suspects(prices) == [])
frac = pd.DataFrame({"amount": rng.uniform(1, 20, 400).round(2)})
frac.loc[3, "amount"] = 5000.0
check("a column with decimals is left alone", sr.find_scale_suspects(frac) == [])
small = pd.DataFrame({"quantity": [1, 2, 3, 5000]})
check("too few rows to judge", sr.find_scale_suspects(small) == [])
many = df.copy()
many.loc[:60, "quantity"] = 5000
check("many large values: it is not a typo pattern", sr.find_scale_suspects(many) == [])
huge = df.copy(); huge.loc[7, "quantity"] = 90000
check("a value too large to be a slip once divided is not flagged",
      all(e["value"] != 90000 for e in (sr.find_scale_suspects(huge) or [{"examples": []}])[0]["examples"]))

new, done = sr.apply_scale_fix(df, ["quantity", "nope"])
check("apply divides only the suspects", done == [{"column": "quantity", "count": len(bad)}], str(done))
check("applied values are the original ordinary ones", all(int(new.loc[i, "quantity"]) == int(q[i]) for i in bad))
check("other cells and the input table are untouched",
      (new.drop(index=bad)["quantity"] == df.drop(index=bad)["quantity"]).all() and df.loc[bad[0], "quantity"] >= 1000)
check("a second run finds nothing", sr.apply_scale_fix(new, ["quantity"])[1] == [])
check("dtype stays whole numbers", pd.api.types.is_integer_dtype(new["quantity"]))

# end to end: the apply step, then the audit explains each changed cell
import audit_service as svc
import scale_service
csv = df.to_csv(index=False).encode()
S, U = "s-scale-1", "local://scale-test"
a0 = svc.audit_response(csv, "orders.csv", S, U, scope="clean", limit=500)
check("before the answer, the trail shows no scale change", not any("1000" in e["rule"] for e in a0["entries"]))
out = scale_service.run_scale_fix(csv, "orders.csv", S, U, ["quantity"])
check("service reports the change", out["success"] and out["changed"] == len(bad), str(out))
a1 = svc.audit_response(csv, "orders.csv", S, U, scope="clean", limit=500)
sc = [e for e in a1["entries"] if e["rule"] == "Divided by 1000 (your choice)"]
check("every changed cell is in the trail with its row, before and after",
      len(sc) == len(bad) and {e["row"] for e in sc} == {i + 2 for i in bad}
      and all(int(e["before"]) == int(e["after"]) * 1000 for e in sc), str(sc[:2]))
check("nothing unexplained", a1["summary"]["unexplained"]["count"] == 0)
again = scale_service.run_scale_fix(csv, "orders.csv", S, U, ["quantity"])
check("answering twice changes nothing more", again["changed"] == 0, str(again))

print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nDONE - all checks passed.")
raise SystemExit(1 if FAILS else 0)
