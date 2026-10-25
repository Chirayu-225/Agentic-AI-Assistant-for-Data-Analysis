"""
test_sandbox_helpers.py — the read-only helpers added to the sandbox
allowlist after the held-out eval (np.issubdtype / np.number, pd.api.types
predicates, series.name, np.triu), run through the REAL sandbox, plus checks
that the widening didn't open anything else.

Run: python test_sandbox_helpers.py
"""
import sys

import pandas as pd

from sandbox_limits import run_sandboxed

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)

DF = pd.DataFrame({"region": ["N", "S", "N", "E"], "units": [3, 5, 2, 7], "price": [1.5, 2.0, 1.0, 3.0]})

def run(code):
    return run_sandboxed(code, DF, include_charts=False, timeout_seconds=20)

ALLOWED = {
    "np.issubdtype with np.number":
        ("result = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]", ["units", "price"]),
    "pd.api.types.is_numeric_dtype":
        ("result = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]", ["units", "price"]),
    "series.name after groupby":
        ("s = df.groupby('region')['units'].sum()\nresult = s.name", "units"),
    "np.triu to de-duplicate a correlation matrix":
        ("m = df[['units', 'price']].corr()\nresult = float(np.triu(m.to_numpy(), k=1).max())", None),
    "names visible inside a lambda (scoping regression)":
        ("f = lambda c: df[c].sum()\nresult = f('units')", 17),
    "names visible inside a nested function (scoping regression)":
        ("def total(col):\n    return float(df[col].sum())\nresult = total('price')", 7.5),
    "astype with a numpy type":
        ("result = df['units'].astype(np.float64).sum()", 17.0),
}
for label, (code, expected) in ALLOWED.items():
    r = run(code)
    ok = r.success and (expected is None or r.value == expected)
    check(f"allowed: {label}", ok, f"success={r.success} value={r.value!r} error={r.error}")

BLOCKED = {
    "pd.api.extensions (not on the nested allowlist)": "result = pd.api.extensions",
    "pd.io": "result = pd.io",
    "np.lib": "result = np.lib",
    "dunder on an exposed numpy type": "result = np.number.__subclasses__()",
    "dunder on the pd.api proxy": "result = pd.api.types.__dict__",
    "a non-listed pd.api.types function": "result = pd.api.types.pandas_dtype('int64')",
    "file write via a frame method": "df.to_csv('x.csv')\nresult = 1",
    "open() still absent after the scoping change": "result = open('x.txt', 'w')",
    "__import__ still absent": "result = __builtins__['__import__']('os')",
}
for label, code in BLOCKED.items():
    r = run(code)
    check(f"still blocked: {label}", not r.success, f"value={r.value!r}")

print()
if FAIL:
    print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("DONE — all checks passed.")
