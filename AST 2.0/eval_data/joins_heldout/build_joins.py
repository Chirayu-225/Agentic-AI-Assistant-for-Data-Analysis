"""
build_joins.py — held-out evaluation set for the Join Datasets (multi-table
SQL) feature. Generates four related tables and computes every expected
answer independently with pandas, so the app's AI-written SQL is checked
against ground truth it never saw.

The questions deliberately include the classic join traps:
  * FAN-OUT: totals of an order-level column (shipping_fee) after joining
    to order lines — a naive JOIN counts each fee once per line.
  * ORPHANS: some order lines reference products missing from the catalogue
    (an INNER JOIN silently drops them) and some customers never ordered.
  * ANTI-JOINS: "never ordered" questions need LEFT JOIN ... IS NULL.

RULE: never tune prompts against these files. Regenerate with a new seed
before quoting a number:  python build_joins.py --seed N

Run: python build_joins.py [--seed N]
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["id", "question", "expected", "answer_type", "tolerance", "trap"]


def build(seed: int) -> None:
    rng = np.random.default_rng(seed)

    n_cust = 60
    customers = pd.DataFrame({
        "customer_id": [f"CU{i:03d}" for i in range(1, n_cust + 1)],
        "city": rng.choice(["Pune", "Chennai", "Jaipur", "Kochi"], n_cust),
        "segment": rng.choice(["Retail", "Corporate", "Wholesale"], n_cust, p=[0.5, 0.3, 0.2]),
    })

    n_prod = 25
    products = pd.DataFrame({
        "product_id": [f"PR{i:03d}" for i in range(1, n_prod + 1)],
        "category": rng.choice(["Furniture", "Stationery", "Electronics"], n_prod),
        "list_price": rng.choice([120, 250, 480, 900, 1500, 3200], n_prod),
    })

    # ~10% of customers never order
    buyers = customers.customer_id.sample(frac=0.9, random_state=seed).tolist()
    n_ord = 220
    orders = pd.DataFrame({
        "order_id": [f"OR{i:04d}" for i in range(1, n_ord + 1)],
        "customer_id": rng.choice(buyers, n_ord),
        "order_date": (pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 120, n_ord), unit="D")).strftime("%Y-%m-%d"),
        "status": rng.choice(["Completed", "Cancelled", "Returned"], n_ord, p=[0.8, 0.12, 0.08]),
        "shipping_fee": rng.choice([0, 40, 60, 99], n_ord),
    })

    rows = []
    sold_products = products.product_id.sample(frac=0.85, random_state=seed + 1).tolist()  # some never ordered
    ghost_products = ["PR901", "PR902"]                                                     # not in catalogue
    for oid in orders.order_id:
        for _ in range(int(rng.integers(1, 6))):
            pid = rng.choice(ghost_products) if rng.random() < 0.03 else rng.choice(sold_products)
            rows.append({"order_id": oid, "product_id": pid, "quantity": int(rng.integers(1, 5))})
    items = pd.DataFrame(rows)
    items.insert(0, "line_id", [f"LN{i:05d}" for i in range(1, len(items) + 1)])
    price = dict(zip(products.product_id, products.list_price))
    items["unit_price"] = [price.get(p, 500) for p in items.product_id]  # ghost lines still have a price

    # ── ground truth (pandas, independent of the app) ──
    li = items.assign(revenue=items.quantity * items.unit_price)
    o_li = li.merge(orders, on="order_id")
    completed = o_li[o_li.status == "Completed"]
    full = o_li.merge(customers, on="customer_id").merge(products, on="product_id", how="left")
    q = []

    def add(question, expected, kind="number", tol=0.001, trap=""):
        q.append({"question": question, "expected": expected, "answer_type": kind, "tolerance": tol, "trap": trap})

    add("What is the total revenue (quantity x unit_price) from Completed orders?",
        round(float(completed.revenue.sum()), 2))
    cat_rev = full[full.status == "Completed"].groupby("category").revenue.sum()
    add("Which product category brings in the most revenue from Completed orders?",
        cat_rev.idxmax(), "text", "")
    add("How many distinct Retail customers placed at least one order?",
        int(orders.merge(customers, on="customer_id").query("segment == 'Retail'").customer_id.nunique()), tol=0)
    elec_orders = full[full.category == "Electronics"].order_id.unique()
    add("What is the total shipping fee across orders that contain at least one Electronics item?",
        round(float(orders[orders.order_id.isin(elec_orders)].shipping_fee.sum()), 2), trap="fan-out")
    add("What is the total shipping fee across all Completed orders?",
        round(float(orders[orders.status == "Completed"].shipping_fee.sum()), 2), trap="fan-out if joined to lines")
    per_order_qty = completed.groupby("order_id").quantity.sum()
    add("On average, how many units (sum of quantity) are in a Completed order?",
        round(float(per_order_qty.mean()), 2), tol=0.01)
    add("How many customers have never placed an order?",
        int((~customers.customer_id.isin(orders.customer_id)).sum()), tol=0, trap="anti-join")
    add("How many products in the catalogue have never been ordered?",
        int((~products.product_id.isin(items.product_id)).sum()), tol=0, trap="anti-join")
    seg_rev = completed.merge(customers, on="customer_id").groupby("segment").revenue.sum()
    add("Which customer segment generated the most revenue from Completed orders?", seg_rev.idxmax(), "text", "")
    add("What is the total quantity of Furniture items sold in Completed orders?",
        int(full[(full.status == "Completed") & (full.category == "Furniture")].quantity.sum()), tol=0)
    mar_corp = completed.merge(customers, on="customer_id")
    mar_corp = mar_corp[(mar_corp.segment == "Corporate") & mar_corp.order_date.str.startswith("2026-03")]
    add("What was the revenue from Corporate customers' Completed orders placed in March 2026?",
        round(float(mar_corp.revenue.sum()), 2))
    lines_per_order = items.groupby("order_id").size()
    add("How many orders have more than 3 order lines?", int((lines_per_order > 3).sum()), tol=0)
    add("How many order lines reference a product that is not in the products table?",
        int((~items.product_id.isin(products.product_id)).sum()), tol=0, trap="orphans")
    grand = completed.revenue.sum() + orders[orders.status == "Completed"].shipping_fee.sum()
    add("What is the total of revenue plus shipping fees across all Completed orders?",
        round(float(grand), 2), trap="fan-out")

    customers.to_csv(os.path.join(HERE, "customers.csv"), index=False)
    products.to_csv(os.path.join(HERE, "products.csv"), index=False)
    orders.to_csv(os.path.join(HERE, "orders.csv"), index=False)
    items.to_csv(os.path.join(HERE, "order_lines.csv"), index=False)
    with open(os.path.join(HERE, "join_questions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(q, start=1):
            w.writerow({"id": f"JN{i:02d}", **row})
    with open(os.path.join(HERE, "seed.txt"), "w", encoding="utf-8") as f:
        f.write(f"{seed}\n")
    print(f"join held-out set written to {HERE} (seed {seed}, {len(q)} questions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261102)
    build(ap.parse_args().seed)
