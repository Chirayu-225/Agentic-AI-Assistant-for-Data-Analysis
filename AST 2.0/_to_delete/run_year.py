import sys, time, resource
sys.path.insert(0, ".")
import pandas as pd
import file_reader, cache_layer
from cleaning import rule_based_clean
from fill_missing import fill_missing, fill_derived
from value_validators import validate_values
name = sys.argv[1]
path = sys.argv[2]
data = open(path, "rb").read()
t = time.time()
raw = file_reader.read_tabular_file(data, name)
t_read = time.time() - t
df, log = cache_layer.cached_rule_clean(raw, data, rule_based_clean)
t_clean = time.time() - t - t_read
before = df.copy()
df, fr = fill_missing(df)
df, vr = validate_values(df)
df, de = fill_derived(df)
tt = time.time() - t
dup = [l for l in log if "exact copies" in l]
print(f"{name}: read {raw.shape} notes={raw.attrs.get('reader_notes')} | cleaned {before.shape} | read {t_read:.1f}s clean {t_clean:.1f}s total {tt:.1f}s | peakMB {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}")
print("   log:", [l[:90] for l in log if not l.startswith("✓ Normalised")])
print("   fill:", [(f["column"], f["count"], f["method"]) for f in fr["filled"]], "derived:", [(d["column"], d["count"]) for d in de])
print("   validators:", [(e["column"], e["rule"], e["count"]) for e in vr["entries"]], "dropped", vr["rowsDropped"], "final", df.shape)
print("   request_date nulls:", int(df.request_date.isna().sum()), "| id dtypes:", {c: str(df[c].dtype) for c in ("order_id","customer_id","item_id")}, "| price:", str(df.price_exact.dtype), "zero:", int((df.price_exact==0).sum()))
for c in ("item_description","project_description","machine_description","prodmodel_description"):
    a, b = raw[c.upper()].astype(str), (df[c].astype(str) if c in df else None)
    same = (a.reset_index(drop=True) == b.reset_index(drop=True)).all() if b is not None and len(a)==len(b) else "rows differ"
    print(f"   {c} unchanged:", same)
