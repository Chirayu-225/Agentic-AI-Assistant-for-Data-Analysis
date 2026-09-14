"""
Quick functional proof for sandbox_proxy.py.

Confirms:
  1. The three concrete bypasses found in the code review are now blocked.
  2. Ordinary analysis code (the kind query_engine.py's prompt asks the
     LLM to generate) still works exactly as before.

Run: python3 test_sandbox_proxy.py
"""

import pandas as pd
from sandbox import safe_exec
from sandbox_proxy import build_safe_namespace, unwrap_result

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Dave"],
    "sales": [1200, 2500, 500, 1800],
    "category": ["Electronics", "Clothing", "Electronics", "Clothing"],
})

print("=" * 70)
print("PART 1 — previously-working exploits, should now be BLOCKED")
print("=" * 70)

exploits = [
    ("pd.read_pickle arbitrary deserialization",
     "result = pd.read_pickle('/tmp/anything.pkl')"),
    ("pd.read_csv arbitrary file read",
     "result = pd.read_csv('/etc/passwd')"),
    ("plt.savefig arbitrary file write",
     "plt.figure()\nplt.plot([1,2,3])\nplt.savefig('/tmp/pwned.png')\nresult = 1"),
]

for label, code in exploits:
    ns = build_safe_namespace(df.copy())
    success, error = safe_exec(code, ns)
    status = "BLOCKED (correct)" if not success else "!!! NOT BLOCKED !!!"
    print(f"\n[{status}] {label}")
    print(f"    code: {code.splitlines()[-1]!r}")
    print(f"    error: {error}")

print("\n" + "=" * 70)
print("PART 2 — legitimate analysis code, should still WORK")
print("=" * 70)

legit = [
    ("groupby aggregation",
     "result = df.groupby('category')['sales'].mean()"),
    ("sort + head",
     "result = df.sort_values('sales', ascending=False).head(2)"),
    ("describe",
     "result = df.describe()"),
    ("fillna (used by the cleaning pass)",
     "cleaned_df = df.fillna(0)"),
    ("matplotlib chart without save (result = plt.gcf() pattern)",
     "plt.figure()\nplt.bar(df['name'], df['sales'])\nresult = plt.gcf()"),
]

for label, code in legit:
    ns = build_safe_namespace(df.copy())
    success, error = safe_exec(code, ns)
    status = "OK" if success else "!!! BROKEN !!!"
    print(f"\n[{status}] {label}")
    if success:
        key = "cleaned_df" if "cleaned_df" in ns else "result"
        value = unwrap_result(ns[key])
        print(f"    -> {type(value).__name__}: {str(value)[:80]}")
    else:
        print(f"    error: {error}")

print("\n" + "=" * 70)
print("Done.")
