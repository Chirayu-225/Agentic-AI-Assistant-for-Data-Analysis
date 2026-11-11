"""Fill-step safety: exact arithmetic fills, stricter inferred fills, median off by default. No model, no network."""
import random
import numpy as np
import pandas as pd
from fill_missing import fill_derived, fill_missing

FAILS = []
def check(label, cond, detail=""):
    print(("[OK] " if cond else "[FAIL] ") + label + ("" if cond else f"  {detail}"))
    if not cond: FAILS.append(label)

random.seed(1)
n = 600
q = [random.randint(1, 12) for _ in range(n)]; pr = [round(random.uniform(5, 400), 2) for _ in range(n)]
truth = pd.DataFrame({"qty": q, "price": pr, "amount": [round(a * b, 2) for a, b in zip(q, pr)]})
df = truth.copy()
df.loc[range(0, 40, 2), "amount"] = np.nan
df.loc[range(1, 41, 4), "qty"] = np.nan

out, rep = fill_missing(df)
check("median is OFF by default: numbers with no evidence stay blank", rep["total_filled"] == 0 and out["amount"].isna().sum() == 20, str(rep["filled"]))
out, rep = fill_missing(df, median=True)
check("median still available when asked", any(f["method"] == "median" for f in rep["filled"]))

out, ent = fill_derived(df)
m = df["amount"].isna()
check("amount = qty x price recovered exactly", np.allclose(out.loc[m, "amount"], truth.loc[m, "amount"], atol=0.011))
m = df["qty"].isna() & df["price"].notna() & df["amount"].notna()
check("qty = amount / price recovered exactly, stays whole", (out.loc[m, "qty"] == truth.loc[m, "qty"]).all())
check("derived entries are reported with the formula", {e["column"] for e in ent} == {"qty", "amount"} and all(e["method"] == "derived" and "price" in e["value"] for e in ent), str(ent))
check("input not modified", df["amount"].isna().sum() == 20)

# no formula -> nothing filled
rnd = pd.DataFrame({"a": np.random.RandomState(0).rand(200), "b": np.random.RandomState(1).rand(200), "c": np.random.RandomState(2).rand(200)})
rnd.loc[3, "c"] = np.nan
check("no relation between columns: nothing is filled", fill_derived(rnd)[0]["c"].isna().sum() == 1)

# a blank is not filled when a factor is blank too
both = df.copy(); both.loc[0, "qty"] = np.nan
o2, _ = fill_derived(both)
check("blank amount AND blank factor: stays blank", pd.isna(o2.loc[0, "amount"]) and pd.isna(o2.loc[0, "qty"]))

# inferred fills: 2 agreeing rows are not enough; 3 are
def groups(per):
    rows = []
    for g in range(12):
        rows += [(f"k{g}", f"v{g}")] * per
    return pd.DataFrame(rows, columns=["k", "v"])
g2 = groups(2); g2.loc[0, "v"] = None
check("group with only 1 known row left (2 per group): not filled", fill_missing(g2)[0]["v"].isna().sum() == 1)
g4 = groups(4); g4.loc[0, "v"] = None
check("group with 3 agreeing known rows: filled", fill_missing(g4)[0]["v"].isna().sum() == 0 and fill_missing(g4)[0].loc[0, "v"] == "v0")
g5 = groups(4); g5.loc[[0, 1], "v"] = None          # 2 of 4 blank = 50% of the group
check("half of the group blank: not filled", fill_missing(g5)[0]["v"].isna().sum() == 2)

# vacuous evidence: the column has ONE distinct known value
v = pd.DataFrame({"k": [f"k{i%10}" for i in range(100)], "v": ["same"] * 100}); v.loc[5, "v"] = None
o, r = fill_missing(v)
check("single distinct known value is not 'evidence'", o["v"].isna().sum() == 1 and r["total_filled"] == 0, str(r["filled"]))

# per-column Unknown only touches that column
t = pd.DataFrame({"a": ["x", None] * 30, "b": ["y", None] * 30, "c": range(60)})
o, r = fill_missing(t, placeholder_columns=["a"])
check("Unknown for one column leaves the other text column blank", o["a"].isna().sum() == 0 and o["b"].isna().sum() == 30)

# key preference: a specific repeating key beats a coarse one
rs = random.Random(3)
cust = [f"c{i}" for i in range(40)]; city = {c: f"city{i%2}" for i, c in enumerate(cust)}
zipc = {c: f"z{i}" for i, c in enumerate(cust)}
rows = []
for _ in range(400):
    c = rs.choice(cust); rows.append((c, city[c], zipc[c]))
z = pd.DataFrame(rows, columns=["customer", "city", "zip"])
blank = z.index[:30]; z.loc[blank, "zip"] = None
o, r = fill_missing(z)
ok = all(o.loc[i, "zip"] == zipc[z.loc[i, "customer"]] for i in blank if pd.notna(o.loc[i, "zip"]))
check("fills come from the specific key (customer), never the coarse one (city)", ok and o["zip"].isna().sum() < 30, str(r["filled"]))

print("\nDONE — all checks passed." if not FAILS else f"\n{len(FAILS)} FAILED")
raise SystemExit(1 if FAILS else 0)
