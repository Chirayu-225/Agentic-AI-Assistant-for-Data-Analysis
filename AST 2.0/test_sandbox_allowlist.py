"""
test_sandbox_allowlist.py — the broadened allowlist: common pandas/numpy/plotly
idioms must work, and the new chainable objects (resample/ewm) must NOT expose
file I/O or the raw object.   Run: python3 test_sandbox_allowlist.py
"""
import os, sys
import numpy as np
import pandas as pd

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(label)

def main():
    from sandbox_limits import run_sandboxed
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "region": ["N", "S", "E", "W"] * 3,
        "units": range(1, 13),
        "price": np.linspace(10, 50, 12),
    })

    print("PART 1 — idioms that must work")
    idioms = {
        "groupby.size":      "result = df.groupby('region').size().reset_index(name='n')",
        "resample":          "result = df.set_index('date')['units'].resample('QS').sum().reset_index()",
        "ewm":               "result = df['units'].ewm(span=3).mean()",
        "to_period":         "result = df['date'].dt.to_period('Q').astype(str).value_counts()",
        "cumcount":          "result = df.groupby('region').cumcount()",
        "crosstab":          "result = pd.crosstab(df['region'], df['units'] > 5)",
        "cut":               "result = pd.cut(df['price'], bins=3).astype(str).value_counts()",
        "np.cumsum":         "result = np.cumsum(df['units'])",
        "np.histogram":      "result = np.histogram(df['price'], bins=4)[0]",
        "arith method":      "result = df['units'].add(1).mul(2)",
        "unstack":           "result = df.groupby(['region', df['date'].dt.quarter])['units'].sum().unstack()",
        "px.treemap":        "result = px.treemap(df, path=['region'], values='units')",
        "go.Heatmap":        "result = go.Figure(go.Heatmap(z=[[1,2],[3,4]]))",
        "px.violin":         "result = px.violin(df, x='region', y='price')",
    }
    for name, code in idioms.items():
        r = run_sandboxed(code, df, timeout_seconds=30)
        check(f"idiom: {name}", r.success, str(r.error))

    print("\nPART 2 — escapes that must stay blocked")
    attacks = {
        "resample().to_csv":    "df.set_index('date').resample('QS').sum().to_csv('x.csv')",
        "resample _obj":        "r = df.set_index('date')['units'].resample('QS')\nresult = r._obj",
        "ewm _obj":             "result = df['units'].ewm(span=3)._obj",
        "groupby.to_pickle":    "df.groupby('region').sum().to_pickle('x.pkl')",
        "iterrows raw rows":    "for i, row in df.iterrows():\n    row.to_csv('x.csv')",
        "itertuples":           "result = list(df.itertuples())",
        "items":                "result = list(df.items())",
        "pd.read_csv":          "result = pd.read_csv('x.csv')",
        "np.save":              "np.save('x.npy', df['units'].values)",
        "np.load":              "result = np.load('x.npy')",
        "df.style":             "result = df.style",
        "df.query":             "result = df.query('units > 3')",
        "df.eval":              "result = df.eval('units + 1')",
        "pd.eval":              "result = pd.eval('1+1')",
        "go.Figure().write_html": "go.Figure().write_html('x.html')",
        "px savefig-like":      "px.bar(df, x='region', y='units').write_image('x.png')",
    }
    for name, code in attacks.items():
        r = run_sandboxed(code, df, timeout_seconds=30)
        wrote = any(os.path.exists(f) for f in ("x.csv", "x.pkl", "x.npy", "x.html", "x.png"))
        check(f"blocked: {name}", (not r.success) and not wrote, f"success={r.success} wrote={wrote}")
        for f in ("x.csv", "x.pkl", "x.npy", "x.html", "x.png"):
            if os.path.exists(f): os.remove(f)

    print()
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
    print("DONE — all checks passed.")

if __name__ == "__main__":
    main()
