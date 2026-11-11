import sys; sys.path.insert(0, ".")
import file_reader, cache_layer
from cleaning import rule_based_clean
from fill_missing import fill_missing, fill_derived
from value_validators import validate_values
data = open("_to_delete/messy_orders_demo.csv","rb").read()
raw = file_reader.read_tabular_file(data, "messy_orders_demo.csv")
df, _ = rule_based_clean(raw)[:2] if isinstance(rule_based_clean(raw), tuple) else (rule_based_clean(raw), None)
df, fr = fill_missing(df)
df, vr = validate_values(df)
df, de = fill_derived(df)
print("filled:", [(f["column"], f["count"], f["method"]) for f in fr["filled"]], [(d["column"], d["count"], d["value"]) for d in de])
print("validators:", [(e.get("column"), e.get("rule"), e.get("count")) for e in vr["entries"]])
print("rows", len(df), "nulls", df.isna().sum()[lambda x: x>0].to_dict())
print("total_amount", round(float(df.amount.sum()),2), "avg_qty", round(float(df.quantity.mean()),3), "max_qty", df.quantity.max(), "avg_disc", round(float(df.discount_pct.mean()),3))
print("territory NA", int((df.territory=="NA").sum()), "usa", int((df.country=="USA").sum()), "namibia", int((df.country=="Namibia").sum()), "customers", df.customer_id.nunique())
print("ship<order", int((df.ship_date<df.order_date).sum()), "future", int((df.order_date>"2026-11-11").sum()))
