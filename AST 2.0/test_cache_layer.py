"""
test_cache_layer.py — the bounded TTL/LRU caches and what hangs off them:
parse caching in file_reader, rule-clean caching, deterministic-result
caching. Includes a real timing measurement on a real Excel file.

Run: python3 test_cache_layer.py
"""

import io
import os
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("AUDIT_LOG_DISABLED", "1")
os.environ.pop("CACHE_ENABLED", None)

import cache_layer
import file_reader
from cache_layer import BoundedCache

FAIL = []


def check(label, condition, detail=""):
    print(f"[{'OK' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


print("=" * 70)
print("PART 1 — BoundedCache: LRU by bytes, TTL, oversize, stats")
print("=" * 70)
now = [0.0]
c = BoundedCache("t", max_bytes=100, ttl_seconds=60, clock=lambda: now[0])
c.put("a", "A", 40); c.put("b", "B", 40)
check("two 40-byte entries fit in 100 bytes", c.get("a") == "A" and c.get("b") == "B")
c.get("a")                      # a becomes most recently used
c.put("c", "C", 40)             # needs room: b is least recently used
check("LRU evicts the least recently used entry", c.get("b") is None and c.get("a") == "A" and c.get("c") == "C")
check("eviction is counted", c.stats()["evictions"] == 1)
check("an entry bigger than half the budget is refused", c.put("big", "X", 60) is False and c.get("big") is None)
check("refusals are counted", c.stats()["tooLargeToCache"] == 1)
check("total bytes never exceed the budget", c.stats()["bytes"] <= 100, str(c.stats()))
now[0] += 61
check("entries expire after the TTL", c.get("a") is None and c.get("c") is None)
check("hit rate is reported", c.stats()["hitRate"] is not None)
c.put("x", 1, 10); c.put("x", 2, 10)
check("re-putting a key replaces it without double-counting bytes", c.get("x") == 2 and c.stats()["bytes"] == 10)

print("\n" + "=" * 70)
print("PART 2 — cached_parse / read_tabular_file")
print("=" * 70)
cache_layer.clear_all()
csv_bytes = b"region,sales\nNorth,10\nSouth,20\nEast,30\n"

calls = {"n": 0}
real_uncached = file_reader._read_tabular_file_uncached
def counting(file_bytes, filename):
    calls["n"] += 1
    return real_uncached(file_bytes, filename)
file_reader._read_tabular_file_uncached = counting

df1 = file_reader.read_tabular_file(csv_bytes, "a.csv")
df2 = file_reader.read_tabular_file(csv_bytes, "a.csv")
check("same bytes parsed only once", calls["n"] == 1, f"parse calls={calls['n']}")
check("both calls return equal data", df1.equals(df2))

df1.loc[0, "sales"] = 9999
df1["extra"] = 1
df3 = file_reader.read_tabular_file(csv_bytes, "a.csv")
check("mutating a returned frame never corrupts the cache", int(df3.loc[0, "sales"]) == 10 and "extra" not in df3.columns)
check("each hit returns an independent object", df2 is not df3)

file_reader.read_tabular_file(csv_bytes + b"West,40\n", "a.csv")
check("different bytes are parsed again", calls["n"] == 2)

try:
    file_reader.read_tabular_file(b"x", "notes.pdf")
    check("unsupported type raises", False)
except ValueError:
    check("unsupported type still raises ValueError", True)
before = calls["n"]
try:
    file_reader.read_tabular_file(b"x", "notes.pdf")
except ValueError:
    pass
check("failures are never cached (parsed again, raised again)", calls["n"] == before + 1)

os.environ["CACHE_ENABLED"] = "0"
n0 = calls["n"]
file_reader.read_tabular_file(csv_bytes, "a.csv"); file_reader.read_tabular_file(csv_bytes, "a.csv")
check("CACHE_ENABLED=0 bypasses the cache entirely", calls["n"] == n0 + 2)
del os.environ["CACHE_ENABLED"]
file_reader._read_tabular_file_uncached = real_uncached

print("\n" + "=" * 70)
print("PART 3 — cached_rule_clean")
print("=" * 70)
cache_layer.clear_all()
from cleaning import rule_based_clean
n = {"c": 0}
def counting_clean(df):
    n["c"] += 1
    return rule_based_clean(df)
raw = pd.DataFrame({"price": ["$10", "$20", "$30"], "name": ["a", "b", "c"]})
d1, log1 = cache_layer.cached_rule_clean(raw, b"bytes-1", counting_clean)
d2, log2 = cache_layer.cached_rule_clean(raw, b"bytes-1", counting_clean)
check("cleaning runs once for the same file", n["c"] == 1)
check("cleaned data and log are identical on a hit", d1.equals(d2) and log1 == log2 and len(log1) > 0)
d2["price"] = -1; log2.append("tampered")
d3, log3 = cache_layer.cached_rule_clean(raw, b"bytes-1", counting_clean)
check("mutating a cleaned frame / log does not corrupt the cache",
      (d3["price"] != -1).all() and "tampered" not in log3)
cache_layer.cached_rule_clean(raw, b"bytes-2", counting_clean)
check("a different file is cleaned separately", n["c"] == 2)

print("\n" + "=" * 70)
print("PART 4 — cached_result (deterministic endpoints only)")
print("=" * 70)
cache_layer.clear_all()
k = {"n": 0}
def compute_ok():
    k["n"] += 1
    return {"success": True, "rows": [1, 2, 3], "filename": "a.csv"}
r1 = cache_layer.cached_result("profile", b"file", compute_ok, extra="a.csv")
r1["rows"].append("tampered")
r2 = cache_layer.cached_result("profile", b"file", compute_ok, extra="a.csv")
check("computed once, served from cache second time", k["n"] == 1)
check("callers get a deep copy (mutation can't leak)", r2["rows"] == [1, 2, 3])
cache_layer.cached_result("profile", b"file", compute_ok, extra="renamed.csv")
check("the echoed filename is part of the key", k["n"] == 2)
cache_layer.cached_result("clean", b"file", compute_ok, extra="a.csv")
check("endpoint name is part of the key", k["n"] == 3)

fails = {"n": 0}
def compute_fail():
    fails["n"] += 1
    return {"success": False, "error": "boom"}
cache_layer.cached_result("profile", b"bad", compute_fail); cache_layer.cached_result("profile", b"bad", compute_fail)
check("error responses are never cached", fails["n"] == 2)

print("\n" + "=" * 70)
print("PART 5 — real measurement: a 15,000-row, 8-column Excel workbook")
print("=" * 70)
cache_layer.clear_all()
rng = np.random.default_rng(0)
big = pd.DataFrame({f"col{i}": rng.integers(0, 1000, 15000) for i in range(6)})
big["region"] = rng.choice(["North", "South", "East", "West"], 15000)
big["note"] = [f"customer {i}" for i in range(15000)]
buf = io.BytesIO()
big.to_excel(buf, index=False)
xlsx = buf.getvalue()
print(f"    workbook size: {len(xlsx) / 1024:.0f} KB")
t0 = time.perf_counter(); first = file_reader.read_tabular_file(xlsx, "big.xlsx"); t_first = time.perf_counter() - t0
t0 = time.perf_counter(); second = file_reader.read_tabular_file(xlsx, "big.xlsx"); t_second = time.perf_counter() - t0
print(f"    first read (parse): {t_first * 1000:8.1f} ms")
print(f"    second read (cache): {t_second * 1000:7.1f} ms   ({t_first / max(t_second, 1e-9):.0f}x faster)")
check("the cached read returns the same data", first.equals(second))
check("the cached read is at least 5x faster than parsing", t_second * 5 < t_first,
      f"parse={t_first:.3f}s cached={t_second:.3f}s")
st = cache_layer.stats()["frame"]
check("stats show the hit", st["hits"] >= 1 and st["entries"] >= 1, str(st))

print("\n" + "=" * 70)
if FAIL:
    print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("DONE — all checks passed.")
