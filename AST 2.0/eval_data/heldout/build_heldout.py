"""
build_heldout.py — regenerates the HELD-OUT evaluation sets.

RULE: never tune prompts, shortcuts or cleaning against these files.
They exist to measure, not to develop against. Iterate on
eval_data/sample_*.csv (the dev set) and only run these to report.

How ground truth stays independent of the app: each dataset is generated
as a clean "truth" table first, every expected answer is computed from
that truth table here, and only THEN is the presentation corrupted the
way real exports are (currency symbols, spelling variants, mixed date
formats, a TOTAL row, duplicates, blank cells, a dashboard sheet). The
app only ever sees the corrupted file, so it has to recover the truth to
score. Spot-check a handful of expected values by hand (Excel/pivot) the
first time you use these — that is what makes them trustworthy.

KNOWABLE ANSWERS ONLY: any information the corruption destroys (e.g. a
category cell blanked out) is removed from the truth table BEFORE the
answers are computed, so every expected answer can be recovered from the
file the app actually sees. (An earlier version blanked categories after
computing answers, which made some questions unanswerable by anyone.)

FRESH SETS: once a held-out set has been run and looked at, it is no
longer unseen. Generate a new one with a new seed before quoting a number:
    python build_heldout.py --seed 20261101
The seed used is written to seed.txt next to the files.

Run: python build_heldout.py [--seed N]   (deterministic for a given seed)
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEED = 20261025
RNG = np.random.default_rng(DEFAULT_SEED)  # replaced in main() from --seed
FIELDS = ["id", "question", "expected", "answer_type", "tolerance", "category", "manual_seconds"]


def _write_questions(name: str, rows: list[dict]) -> None:
    with open(os.path.join(HERE, f"{name}_questions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            w.writerow({"id": f"{name[:3].upper()}{i:02d}", "tolerance": "", "manual_seconds": "", **r})


def _num(x: float, nd: int = 2) -> str:
    return f"{round(float(x), nd)}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. invoices_messy.csv — AR ledger export
# ─────────────────────────────────────────────────────────────────────────────
def build_invoices() -> None:
    n = 400
    regions = np.array(["North", "South", "East", "West"])
    segments = np.array(["Enterprise", "Mid-Market", "SMB"])
    t = pd.DataFrame({
        "invoice_id": np.arange(50001, 50001 + n),
        "invoice_date": pd.to_datetime("2026-01-01") + pd.to_timedelta(RNG.integers(0, 181, n), unit="D"),
        "region": RNG.choice(regions, n, p=[0.3, 0.35, 0.2, 0.15]),
        "segment": RNG.choice(segments, n, p=[0.2, 0.35, 0.45]),
        "status": RNG.choice(["Paid", "Overdue", "Pending"], n, p=[0.7, 0.18, 0.12]),
    })
    base = {"Enterprise": 90000, "Mid-Market": 35000, "SMB": 9000}
    t["amount"] = [round(RNG.gamma(4, base[s] / 4), 2) for s in t["segment"]]
    t["discount_pct"] = RNG.choice([0, 5, 10, 15], n, p=[0.5, 0.25, 0.15, 0.1])
    days = RNG.integers(5, 90, n).astype(float)
    days[t["status"] != "Paid"] = np.nan
    t["days_to_pay"] = days

    # ── ground truth from the clean table ──
    q = []
    by_region = t.groupby("region")["amount"].sum().sort_values(ascending=False)
    by_seg_avg = t.groupby("segment")["amount"].mean()
    month = t["invoice_date"].dt.to_period("M").astype(str)
    by_month = t.groupby(month)["amount"].sum()
    q += [
        dict(question="How many invoices are there?", expected=n, answer_type="number", tolerance=0, category="count"),
        dict(question="What is the total invoiced amount?", expected=_num(t.amount.sum()), answer_type="number", tolerance=0.001, category="aggregate"),
        dict(question="What is the average invoice amount?", expected=_num(t.amount.mean()), answer_type="number", tolerance=0.001, category="aggregate"),
        dict(question="Which region has the highest total invoiced amount?", expected=by_region.index[0], answer_type="text", category="groupby"),
        dict(question="List regions ranked by total invoiced amount, highest first", expected="|".join(by_region.index), answer_type="contains", category="ranking"),
        dict(question="How many distinct regions are there?", expected=4, answer_type="number", tolerance=0, category="count"),
        dict(question="How many invoices are overdue?", expected=int((t.status == "Overdue").sum()), answer_type="number", tolerance=0, category="filter"),
        dict(question="What is the total amount of overdue invoices?", expected=_num(t.loc[t.status == "Overdue", "amount"].sum()), answer_type="number", tolerance=0.001, category="filter"),
        dict(question="Which customer segment has the highest average invoice amount?", expected=by_seg_avg.idxmax(), answer_type="text", category="groupby"),
        dict(question="What percentage of total invoiced amount comes from the Enterprise segment?",
             expected=_num(100 * t.loc[t.segment == "Enterprise", "amount"].sum() / t.amount.sum(), 1), answer_type="number", tolerance=0.02, category="share"),
        dict(question="What is the median days to pay for paid invoices?", expected=_num(t.days_to_pay.median(), 1), answer_type="number", tolerance=0.01, category="aggregate"),
        dict(question="How many invoices have no days_to_pay value?", expected=int(t.days_to_pay.isna().sum()), answer_type="number", tolerance=0, category="missing"),
        dict(question="Which month had the highest total invoiced amount? Answer as YYYY-MM.", expected=by_month.idxmax(), answer_type="text", category="time"),
        dict(question="What was the total invoiced amount in March 2026?", expected=_num(by_month.get("2026-03", 0)), answer_type="number", tolerance=0.001, category="time"),
        dict(question="How many invoices are larger than 50,000?", expected=int((t.amount > 50000).sum()), answer_type="number", tolerance=0, category="filter"),
        dict(question="What is the average discount percentage?", expected=_num(t.discount_pct.mean()), answer_type="number", tolerance=0.01, category="aggregate"),
        dict(question="How many pending invoices are in the South region?", expected=int(((t.status == "Pending") & (t.region == "South")).sum()), answer_type="number", tolerance=0, category="filter"),
        dict(question="Show all overdue invoices in the North region", expected=int(((t.status == "Overdue") & (t.region == "North")).sum()), answer_type="rows", category="filter"),
        dict(question="Plot total invoiced amount by month as a line chart", expected="", answer_type="chart", category="chart"),
    ]
    _write_questions("invoices", q)

    # ── corrupt the presentation ──
    m = t.copy().astype(object)
    variants = {"North": ["North", "north", "NORTH ", "N."], "South": ["South", "south", " South"],
                "East": ["East", "east", "EAST"], "West": ["West", "west ", "W."]}
    m["region"] = [RNG.choice(variants[r], p=None) if RNG.random() < 0.3 else r for r in t.region]
    m["segment"] = [{"SMB": RNG.choice(["SMB", "smb", "S.M.B."]), "Mid-Market": RNG.choice(["Mid-Market", "Mid Market", "mid-market"])}.get(s, s)
                    if RNG.random() < 0.3 else s for s in t.segment]
    fmts = ["%Y-%m-%d", "%d %b %Y", "%Y/%m/%d"]
    m["invoice_date"] = [d.strftime(fmts[RNG.integers(0, 3)]) for d in t.invoice_date]
    m["amount"] = [RNG.choice([f"₹{a:,.2f}", f"{a:.2f}", f"Rs. {a:,.2f}"]) for a in t.amount]
    m["discount_pct"] = [f"{d}%" for d in t.discount_pct]
    m["days_to_pay"] = ["" if pd.isna(d) else int(d) for d in t.days_to_pay]
    total = {c: "" for c in m.columns}
    total.update({"invoice_id": "TOTAL", "amount": f"₹{t.amount.sum():,.2f}"})
    m = pd.concat([m, pd.DataFrame([total])], ignore_index=True)
    m.to_csv(os.path.join(HERE, "invoices_messy.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. store_units_wide.xlsx — wide monthly KPI workbook with a dashboard tab
# ─────────────────────────────────────────────────────────────────────────────
def build_store_units() -> None:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cities = ["Bengaluru", "Mumbai", "Delhi", "Pune", "Chennai", "Hyderabad"]
    n = 36
    t = pd.DataFrame({"store_id": [f"ST{100 + i}" for i in range(n)],
                      "city": RNG.choice(cities, n, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1])})
    season = np.array([0.8, 0.85, 1.0, 0.95, 0.9, 0.9, 1.0, 1.05, 1.0, 1.25, 1.4, 1.3])
    scale = RNG.gamma(5, 120, n)
    for j, mo in enumerate(months):
        units = np.round(scale * season[j] * RNG.uniform(0.8, 1.2, n)).astype(int)
        units[RNG.random(n) < 0.03] = 0  # a few closed-store months
        t[mo] = units
    t["annual"] = t[months].sum(axis=1)

    q = [
        dict(question="How many stores are there?", expected=n, answer_type="number", tolerance=0, category="count"),
        dict(question="What were total units sold in March across all stores?", expected=int(t.Mar.sum()), answer_type="number", tolerance=0, category="aggregate"),
        dict(question="Which store sold the most units over the whole year?", expected=t.loc[t.annual.idxmax(), "store_id"], answer_type="text", category="ranking"),
        dict(question="Which city has the most stores?", expected=t.city.value_counts().idxmax(), answer_type="text", category="groupby"),
        dict(question="Which month had the highest total units sold across all stores?", expected=t[months].sum().idxmax(), answer_type="text", category="time"),
        dict(question="What is the average monthly units sold per store, across all stores and months?", expected=_num(t[months].values.mean()), answer_type="number", tolerance=0.01, category="aggregate"),
        dict(question="How many store-months had zero units sold?", expected=int((t[months] == 0).sum().sum()), answer_type="number", tolerance=0, category="filter"),
        dict(question="What were total annual units for stores in Bengaluru?", expected=int(t.loc[t.city == "Bengaluru", "annual"].sum()), answer_type="number", tolerance=0, category="filter"),
        dict(question="By what percentage did total units in December exceed total units in January?",
             expected=_num(100 * (t.Dec.sum() - t.Jan.sum()) / t.Jan.sum(), 1), answer_type="number", tolerance=0.02, category="share"),
        dict(question="How many stores sold more than 10,000 units in the year?", expected=int((t.annual > 10000).sum()), answer_type="number", tolerance=0, category="filter"),
        dict(question="List the top 3 cities by total annual units", expected="|".join(t.groupby("city").annual.sum().sort_values(ascending=False).index[:3]), answer_type="contains", category="ranking"),
        dict(question="What share of the year's units were sold in Q4 (Oct-Dec)? Give a percentage.",
             expected=_num(100 * t[["Oct", "Nov", "Dec"]].values.sum() / t[months].values.sum(), 1), answer_type="number", tolerance=0.02, category="share"),
        dict(question="Show the stores in Pune", expected=int((t.city == "Pune").sum()), answer_type="rows", category="filter"),
        dict(question="Plot total units by month as a bar chart", expected="", answer_type="chart", category="chart"),
    ]
    _write_questions("store_units", q)

    dash = pd.DataFrame({"KPI": ["Total units", "Stores", "Best month"],
                         "Value": [int(t.annual.sum()), n, t[months].sum().idxmax()]})
    with pd.ExcelWriter(os.path.join(HERE, "store_units_wide.xlsx")) as xw:
        dash.to_excel(xw, sheet_name="Dashboard", index=False)
        t.drop(columns=["annual"]).to_excel(xw, sheet_name="Store Data", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. expenses_dirty.csv — expense claims with duplicates and refunds
# ─────────────────────────────────────────────────────────────────────────────
def build_expenses() -> None:
    n = 300
    t = pd.DataFrame({
        "expense_id": [f"EX-{i:05d}" for i in range(1, n + 1)],
        "employee_id": RNG.integers(100200, 100260, n),
        "department": RNG.choice(["Sales", "Engineering", "Finance", "Operations"], n, p=[0.35, 0.3, 0.15, 0.2]),
        "category": RNG.choice(["Travel", "Meals", "Software", "Office"], n, p=[0.35, 0.3, 0.2, 0.15]),
    })
    amt = np.round(RNG.gamma(2.5, 1800, n), 2)
    refund = RNG.random(n) < 0.06
    amt[refund] = -np.round(RNG.uniform(200, 3000, refund.sum()), 2)
    t["amount_inr"] = amt
    t["submitted"] = (pd.to_datetime("2026-04-01") + pd.to_timedelta(RNG.integers(0, 120, n), unit="D")).strftime("%Y-%m-%d")
    # Blank some categories IN THE TRUTH TABLE, before answers are computed:
    # nobody can know what a blanked category was, so the expected answers
    # must not depend on it (category questions simply exclude those rows).
    t.loc[RNG.random(n) < 0.04, "category"] = np.nan

    spend = t[t.amount_inr > 0]
    q = [
        dict(question="How many unique expense claims are there?", expected=n, answer_type="number", tolerance=0, category="count"),
        dict(question="What is the net total of all expenses including refunds?", expected=_num(t.amount_inr.sum()), answer_type="number", tolerance=0.001, category="aggregate"),
        dict(question="How many refunds (negative amounts) are there?", expected=int(refund.sum()), answer_type="number", tolerance=0, category="filter"),
        dict(question="Which department has the highest net expense total?", expected=t.groupby("department").amount_inr.sum().idxmax(), answer_type="text", category="groupby"),
        dict(question="What is the average expense amount, excluding refunds?", expected=_num(spend.amount_inr.mean()), answer_type="number", tolerance=0.001, category="aggregate"),
        dict(question="Which expense category has the largest net total?", expected=t.groupby("category").amount_inr.sum().idxmax(), answer_type="text", category="groupby"),
        dict(question="How many distinct employees submitted expenses?", expected=int(t.employee_id.nunique()), answer_type="number", tolerance=0, category="count"),
        dict(question="What is the largest single expense?", expected=_num(t.amount_inr.max()), answer_type="number", tolerance=0.0001, category="aggregate"),
        dict(question="What is the net Travel spend in the Sales department?", expected=_num(t[(t.department == "Sales") & (t.category == "Travel")].amount_inr.sum()), answer_type="number", tolerance=0.001, category="filter"),
        dict(question="How many expenses were submitted in May 2026?", expected=int(t.submitted.str.startswith("2026-05").sum()), answer_type="number", tolerance=0, category="time"),
        dict(question="What percentage of net spend is Software?", expected=_num(100 * t[t.category == "Software"].amount_inr.sum() / t.amount_inr.sum(), 1), answer_type="number", tolerance=0.02, category="share"),
        dict(question="List departments ranked by net expense total, highest first", expected="|".join(t.groupby("department").amount_inr.sum().sort_values(ascending=False).index), answer_type="contains", category="ranking"),
        dict(question="Show all refunds", expected=int(refund.sum()), answer_type="rows", category="filter"),
        dict(question="Plot net expense total by department as a bar chart", expected="", answer_type="chart", category="chart"),
    ]
    _write_questions("expenses", q)

    m = t.copy()
    dup_idx = RNG.choice(n, 12, replace=False)
    m = pd.concat([m, m.iloc[dup_idx]], ignore_index=True)  # exact duplicate submissions
    m = m.sample(frac=1, random_state=7).reset_index(drop=True)
    m["department"] = [d.upper() if RNG.random() < 0.15 else d for d in m.department]
    m["amount_inr"] = [f"({abs(a):,.2f})" if a < 0 and RNG.random() < 0.5 else f"{a:,.2f}" for a in m.amount_inr]
    m["category"] = m["category"].fillna("")
    m.to_csv(os.path.join(HERE, "expenses_dirty.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the held-out evaluation sets.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    RNG = np.random.default_rng(args.seed)
    build_invoices()
    build_store_units()
    build_expenses()
    with open(os.path.join(HERE, "seed.txt"), "w", encoding="utf-8") as f:
        f.write(f"{args.seed}\n")
    print(f"held-out sets written to {HERE} (seed {args.seed})")
