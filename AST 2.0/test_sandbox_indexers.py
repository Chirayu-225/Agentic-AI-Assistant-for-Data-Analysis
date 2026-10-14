"""
test_sandbox_indexers.py — regression tests for two findings that came out of
running the evaluation harness against real model output:

  1. Pure pandas idioms the model uses constantly (idxmax, loc, iloc, isin...)
     were blocked, so correct questions failed with a red error.
  2. SECURITY: `df._obj` handed generated code the raw, unproxied DataFrame,
     so `df._obj.to_csv("x.csv")` wrote a file from inside the sandbox.

Every escape attempt below is run through the REAL sandbox (subprocess).

Run: python3 test_sandbox_indexers.py
"""
import os, sys, tempfile
import pandas as pd

os.environ.setdefault("AUDIT_LOG_DISABLED", "1")
FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def main():
    from sandbox_limits import run_sandboxed
    df = pd.DataFrame({"region": ["North", "South", "South", "East"], "rev": [10, 20, 30, 5],
                       "units": [1, 2, 3, 4]})

    def run(code):
        return run_sandboxed(code, df, timeout_seconds=30)

    print("=" * 70); print("PART 1 — legitimate idioms now work"); print("=" * 70)
    r = run("result = df.groupby('region')['rev'].sum().idxmax()")
    check("groupby().sum().idxmax()", r.success and r.value == "South", str(r.error))
    r = run("result = df.groupby('region')['rev'].sum().idxmin()")
    check("idxmin()", r.success and r.value == "East", str(r.error))
    r = run("result = df.loc[df['rev'] > 15, 'region']")
    check("df.loc[mask, 'col']", r.success and list(r.value) == ["South", "South"], str(r.error))
    r = run("result = df.iloc[0]['rev']")
    check("df.iloc[0]['rev']", r.success and int(r.value) == 10, str(r.error))
    r = run("result = df.iloc[1:3]")
    check("df.iloc[slice]", r.success and len(r.value) == 2, str(r.error))
    r = run("result = df.at[1, 'rev']")
    check("df.at scalar", r.success and int(r.value) == 20, str(r.error))
    r = run("result = df[df['region'].isin(['South', 'East'])]")
    check("isin() filter", r.success and len(r.value) == 3, str(r.error))
    r = run("result = df[df['rev'].between(10, 20)]")
    check("between() filter", r.success and len(r.value) == 2, str(r.error))
    r = run("df.loc[df['rev'] < 10, 'rev'] = 0\nresult = df['rev'].sum()")
    check("assignment through .loc works on the sandbox's own copy", r.success and int(r.value) == 60, str(r.error))
    check("...and never touches the caller's DataFrame", int(df["rev"].sum()) == 65)
    r = run("result = df['rev'].shift(1).fillna(0).sum()")
    check("shift()", r.success, str(r.error))

    print("\n" + "=" * 70); print("PART 2 — escape attempts are blocked"); print("=" * 70)
    tmp = tempfile.mkdtemp(prefix="escape_")
    target = os.path.join(tmp, "stolen.csv").replace("\\", "/")
    attempts = {
        "df._obj.to_csv (the original escape)": f"raw = df._obj\nraw.to_csv('{target}')\nresult = 1",
        "df._obj returned directly": "result = df._obj",
        "pd._mod raw module": "m = pd._mod\nresult = m.read_csv('/etc/hostname')",
        "np._mod raw module": "result = np._mod",
        "df.loc.obj raw frame": "result = df.loc.obj",
        "df.loc._idx raw indexer": "result = df.loc._idx",
        "df.iloc.obj raw frame": f"df.iloc.obj.to_csv('{target}')\nresult = 1",
        "callable key hands lambda the raw frame": f"result = df.loc[lambda d: d.to_csv('{target}') or d['rev'] > 0]",
        "groupby proxy private slot": "result = df.groupby('region')._obj",
    }
    for label, code in attempts.items():
        r = run(code)
        check(f"blocked: {label}", not r.success, "succeeded: value=" + repr(r.value))
    check("no file was written by any attempt", os.listdir(tmp) == [], str(os.listdir(tmp)))
    r = run("result = df.loc.__class__")
    check("dunder access still blocked", not r.success)
    r = run("result = df.to_csv('/tmp/x.csv')")
    check("direct to_csv still blocked", not r.success)

    print("\n" + "=" * 70); print("PART 2b — operators (comparisons, arithmetic, boolean masks)"); print("=" * 70)
    ops = {
        "df[df['rev'] > 15] (filter)": ("result = len(df[df['rev'] > 15])", 2),
        "df[df['region'] == 'South'] (== is not silently False)": ("result = len(df[df['region'] == 'South'])", 2),
        "df[df['region'] != 'South']": ("result = len(df[df['region'] != 'South'])", 2),
        "(a) & (b) mask": ("result = len(df[(df['rev'] > 5) & (df['units'] < 4)])", 3),
        "(a) | (b) mask": ("result = len(df[(df['rev'] > 25) | (df['units'] == 1)])", 2),
        "~mask": ("result = len(df[~df['region'].isin(['South'])])", 2),
        "arithmetic between columns": ("result = (df['rev'] + df['units']).sum()", 65 + 10),
        "scalar arithmetic and reflected op": ("result = (2 * df['rev'] - 1).sum()", 2 * 65 - 4),
        "division": ("result = (df['rev'] / df['units']).round(1).iloc[0]", 10.0),
        "power and unary minus": ("result = (-(df['units'] ** 2)).sum()", -(1 + 4 + 9 + 16)),
        "new column from arithmetic": ("df['tot'] = df['rev'] * df['units']\nresult = df['tot'].sum()", 10 + 40 + 90 + 20),
        "filter then groupby then idxmax": ("result = df[df['rev'] > 5].groupby('region')['rev'].sum().idxmax()", "South"),
    }
    for label, (code, want) in ops.items():
        r = run(code)
        got = r.value
        try:
            got = got.item() if hasattr(got, "item") else got
        except Exception:
            pass
        check(label, r.success and got == want, f"{r.error!r} got={got!r} want={want!r}")
    r = run("result = df[df['rev'] > 15]")
    check("a filtered frame comes back as a normal DataFrame (proxy not leaked)",
          r.success and isinstance(r.value, pd.DataFrame) and len(r.value) == 2, repr(r.error))

    print("\n" + "=" * 70); print("PART 2c — lambda escape (raw pandas objects handed to user lambdas)"); print("=" * 70)
    tmp2 = tempfile.mkdtemp(prefix="lambda_escape_")
    tgt = os.path.join(tmp2, "x").replace("\\", "/")
    lam = {
        "apply: col.to_csv": f"result = df.apply(lambda col: col.to_csv('{tgt}.csv'))",
        "apply axis=1: r.to_pickle": f"result = df.apply(lambda r: r.to_pickle('{tgt}.pkl'), axis=1)",
        "agg lambda: s.to_csv": f"result = df.groupby('region')['rev'].agg(lambda s: s.to_csv('{tgt}.csv'))",
        "agg dict of lambdas": f"result = df.agg({{'rev': lambda s: s.to_csv('{tgt}.csv')}})",
        "sort key lambda": f"result = df.sort_values('rev', key=lambda s: s.to_csv('{tgt}.csv'))",
        "transform lambda": f"result = df.groupby('region')['rev'].transform(lambda s: s.to_json('{tgt}.json'))",
    }
    for label, code in lam.items():
        r = run(code)
        check(f"lambda escape blocked: {label}", not r.success, "succeeded: " + repr(r.value))
    check("no file was written by any lambda attempt", os.listdir(tmp2) == [], str(os.listdir(tmp2)))
    for label, code, want in (
        ("apply with a lambda", "result = df['rev'].apply(lambda v: v * 2).sum()", 130),
        ("agg with a lambda", "result = df.groupby('region')['rev'].agg(lambda s: s.max() - s.min()).max()", 10),
        ("row-wise apply uses the row", "result = df.apply(lambda r: r['rev'] * r['units'], axis=1).sum()", 10 + 40 + 90 + 20),
        ("agg dict with a lambda", "result = df.agg({'rev': lambda s: s.sum()})['rev']", 65),
        ("sort_values(key=lambda)", "result = df.sort_values('rev', key=lambda s: -s)['region'].iloc[0]", "South"),
        ("map with a lambda", "result = df['region'].map(lambda v: v.upper()).iloc[0]", "NORTH"),
        ("transform with a lambda", "result = df.groupby('region')['rev'].transform(lambda s: s - s.mean()).abs().sum()", 10.0),
        ("passing a numpy function", "result = df['rev'].apply(np.sqrt).round(2).iloc[0]", 3.16),
    ):
        r = run(code)
        got = r.value.item() if hasattr(r.value, "item") else r.value
        check(f"legit lambda still works: {label}", r.success and got == want, f"{r.error!r} got={got!r}")

    print("\n" + "=" * 70); print("PART 3 — AST guard unit checks"); print("=" * 70)
    from sandbox import is_code_safe
    for code, expect_safe in (("x = df._obj", False), ("df.to_csv('a')", False), ("df.groupby('a').agg(lambda s: s.to_pickle('x'))", False), ("x = df.groupby('a')._mod", False),
                              ("x = df.groupby('a').sum()", True), ("x = df.loc[0]", True),
                              ("x = _private_name_is_a_Name_not_an_attribute", True)):
        ok, why = is_code_safe(code)
        check(f"is_code_safe({code!r}) -> {expect_safe}", ok == expect_safe, why)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE — {len(FAIL)} FAILURE(S): {FAIL}"); sys.exit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
