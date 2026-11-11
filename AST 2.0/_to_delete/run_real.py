import sys; sys.path.insert(0, ".")
import file_reader, cache_layer
from cleaning import rule_based_clean
from fill_missing import fill_missing, fill_derived
from value_validators import validate_values
data = open("_to_delete/orders2026_sample.csv","rb").read()
raw = file_reader.read_tabular_file(data, "orders2026_sample.csv")
print("read:", raw.shape, "| notes:", raw.attrs.get("reader_notes"))
df, log = cache_layer.cached_rule_clean(raw, data, rule_based_clean)
print("cleaned:", df.shape)
for l in log: print("  ", l[:230])
df, fr = fill_missing(df); print("fill:", [(f["column"], f["count"], f["method"]) for f in fr["filled"]], "| questions:", [(s["column"], s["count"]) for s in fr["skipped"] if s["kind"]=="no_evidence"])
df, vr = validate_values(df); print("validators:", [(e["column"], e["rule"], e["count"]) for e in vr["entries"]], "dropped", vr["rowsDropped"])
df, de = fill_derived(df); print("derived:", [(d["column"], d["count"], d["value"]) for d in de])
print("dtypes:", df.dtypes.astype(str).to_dict())
print("REQUEST_DATE range:", df.REQUEST_DATE.min(), df.REQUEST_DATE.max(), "nulls:", int(df.REQUEST_DATE.isna().sum()))
print("LOCATION:", df.LOCATION.value_counts().to_dict())
