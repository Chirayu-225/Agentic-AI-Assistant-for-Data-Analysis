"""
eval.py — Evaluation & benchmarking harness for Analyst Assistant.

Tests deterministic components only (no LLM calls needed).

Usage:
    python eval.py                          # all suites, built-in data
    python eval.py --csv path/to/data.csv  # use your own CSV
    python eval.py --suite shortcuts        # one suite only
    python eval.py --verbose               # show generated code
"""

import sys
import argparse
import pandas as pd
import numpy as np
from io import StringIO

sys.path.insert(0, ".")

from cleaning import rule_based_clean
from sandbox  import is_code_safe, safe_exec
from query_engine import detect_shortcut


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV = """name,sales,category,country
Alice,1200,Electronics,USA
Bob,2500,Clothing,UK
Charlie,500,Electronics,USA
Dave,1800,Clothing,Germany
Eve,1500,Electronics,USA
Frank,1500,Electronics,USA
"""


class R:
    def __init__(self, name, passed, detail=""):
        self.name   = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        icon = "✅" if self.passed else "❌"
        s = f"{icon}  {self.name}"
        if self.detail:
            s += f"\n     → {self.detail}"
        return s


# ─────────────────────────────────────────────────────────────────────────────
# SHORTCUT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_shortcut_tests(df, verbose=False):
    results = []
    cases = [
        ("how many rows",               int,          lambda r: r == len(df)),
        ("how many columns",            int,          lambda r: r == len(df.columns)),
        ("show columns",                pd.DataFrame, lambda r: "column" in r.columns),
        ("what columns are in dataset", pd.DataFrame, lambda r: "column" in r.columns),
        ("describe",                    pd.DataFrame, lambda r: len(r) > 0),
        ("summary stats",               pd.DataFrame, lambda r: len(r) > 0),
        ("show first 5",                pd.DataFrame, lambda r: len(r) <= 5),
        ("head",                        pd.DataFrame, lambda r: len(r) <= 10),
        ("show last 5",                 pd.DataFrame, lambda r: len(r) <= 5),
        ("duplicate rows",  (int, np.int64, np.int32), lambda r: r >= 0),
        ("shape of dataset",            pd.DataFrame, lambda r: "rows" in r.columns),
        ("data types",                  pd.DataFrame, lambda r: "dtype" in r.columns),
        ("null values",                 pd.DataFrame, lambda r: "null_count" in r.columns),
        ("missing values",              pd.DataFrame, lambda r: "null_count" in r.columns),
        ("correlation",                 pd.DataFrame, lambda r: len(r) > 0),
        ("sort by sales",               pd.DataFrame, lambda r: len(r) > 0),
    ]

    for query, exp_type, check_fn in cases:
        shortcut = detect_shortcut(query, df)
        if shortcut is None:
            results.append(R(f"Shortcut: '{query}'", False,
                             "detect_shortcut returned None"))
            continue

        local_vars = {"df": df, "pd": pd, "np": np}
        try:
            exec(shortcut, {"__builtins__": __builtins__}, local_vars)
            success, error = True, ""
        except Exception as e:
            success, error = False, str(e)

        if not success:
            results.append(R(f"Shortcut: '{query}'", False, f"exec failed: {error}"))
            continue

        result = local_vars.get("result")
        if result is None:
            results.append(R(f"Shortcut: '{query}'", False, "result is None"))
            continue

        type_ok  = isinstance(result, exp_type) if not isinstance(exp_type, tuple) \
                   else isinstance(result, exp_type)
        value_ok = check_fn(result)
        passed   = type_ok and value_ok
        detail   = shortcut if verbose else ("" if passed else
                   f"type={type(result).__name__}, value={str(result)[:80]}")
        results.append(R(f"Shortcut: '{query}'", passed, detail))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_cleaning_tests(verbose=False):
    results = []

    # Currency
    df = pd.DataFrame({"price": ["$1200", "$2500", "$500", "$3000", "$4000"],
                       "name":  ["A", "B", "C", "D", "E"]})
    c, log = rule_based_clean(df)
    passed = pd.api.types.is_numeric_dtype(c["price"])
    results.append(R("Cleaning: currency stripping", passed,
                     f"dtype={c['price'].dtype}, values={c['price'].tolist()[:3]}"))

    # Percentages
    df = pd.DataFrame({"rate": ["85%", "90%", "75%", "88%", "92%"]})
    c, _ = rule_based_clean(df)
    passed = pd.api.types.is_numeric_dtype(c["rate"]) and c["rate"].max() <= 1.0
    results.append(R("Cleaning: percentage conversion (÷100)", passed,
                     f"dtype={c['rate'].dtype}, max={c['rate'].max()}"))

    # Sentinel nulls — should be detected AND filled (zero nulls remain)
    df = pd.DataFrame({"score": ["85", "N/A", "none", "90", "--", "88", "na", "null"]})
    c, log = rule_based_clean(df)
    null_count = c["score"].isnull().sum()
    was_detected = any("sentinel" in l.lower() for l in log)
    was_filled   = any("filled" in l.lower() for l in log)
    passed = null_count == 0 and was_detected and was_filled
    results.append(R("Cleaning: sentinel nulls detected and filled", passed,
                     f"nulls={null_count} (expected 0), detected={was_detected}, filled={was_filled}"))

    # Duplicate removal
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
    c, _ = rule_based_clean(df)
    results.append(R("Cleaning: duplicate row removal", len(c) == 3,
                     f"rows={len(c)} (expected 3)"))

    # Header promotion
    df = pd.DataFrame({0: ["product", "Widget A", "Widget B"],
                       1: ["revenue",  "5000",     "7500"]})
    c, _ = rule_based_clean(df)
    passed = "product" in c.columns and "revenue" in c.columns
    results.append(R("Cleaning: header promotion", passed,
                     f"columns={c.columns.tolist()}"))

    # Column name normalisation
    df = pd.DataFrame({"Sales Amount ": [1, 2], " Product Name": ["A", "B"]})
    c, _ = rule_based_clean(df)
    passed = "sales_amount" in c.columns and "product_name" in c.columns
    results.append(R("Cleaning: column normalisation", passed,
                     f"columns={c.columns.tolist()}"))

    # Thousands separators
    df = pd.DataFrame({"rev": ["1,000", "2,500", "10,000", "5,000", "8,000"]})
    c, _ = rule_based_clean(df)
    passed = pd.api.types.is_numeric_dtype(c["rev"])
    results.append(R("Cleaning: thousands separator removal", passed,
                     f"dtype={c['rev'].dtype}, values={c['rev'].tolist()}"))

    # Null filling — numeric gets median, categorical gets 'Unknown', zero nulls remain
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "name": ["a", None, "c"]})
    c, log = rule_based_clean(df)
    null_count = c["name"].isnull().sum()
    filled_correctly = (null_count == 0) and (c["name"].iloc[1] == "Unknown")
    results.append(R("Cleaning: nulls filled (not left dangling)", filled_correctly,
                     f"nulls={null_count} (expected 0), filled_value='{c['name'].iloc[1]}'"))

    # All-numeric nulls get filled with median, not dropped or left as NaN
    df = pd.DataFrame({"price": [100.0, None, 300.0, None, 500.0]})
    c, log = rule_based_clean(df)
    null_count = c["price"].isnull().sum()
    expected_median = 300.0  # median of [100, 300, 500]
    passed = null_count == 0 and abs(c["price"].iloc[1] - expected_median) < 0.01
    results.append(R("Cleaning: numeric nulls filled with median", passed,
                     f"nulls={null_count}, filled_value={c['price'].iloc[1]} (expected {expected_median})"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_security_tests(verbose=False):
    results = []

    blocked = [
        ("import os; os.remove('file')",          "import"),
        ("import sys; sys.exit()",                 "import"),
        ("from os import path",                    "import"),
        ("__import__('os').remove('file')",        "dunder"),
        ("eval('1+1')",                            "eval call"),
        ("exec('x=1')",                            "exec call"),
        ("open('file.txt')",                       "open call"),
        ("getattr(pd, 'read_csv')('/etc/passwd')", "getattr"),
        ("while True: pass",                       "infinite loop"),
        ("result = __builtins__",                  "dunder"),
        ("subprocess.run(['ls'])",                 "subprocess"),
    ]

    for code, label in blocked:
        safe, reason = is_code_safe(code)
        passed = not safe
        results.append(R(
            f"Security: {label} blocked",
            passed,
            f"blocked='{reason}'" if passed else f"⚠ NOT BLOCKED: {code[:60]}"
        ))

    safe_code = [
        "result = df.head(5)",
        "result = df['sales'].sum()",
        "result = df.groupby('category')['sales'].mean()",
        "result = df.describe()",
        "result = df.isnull().sum()",
    ]
    for code in safe_code:
        safe, reason = is_code_safe(code)
        results.append(R(
            f"Security: safe code passes — {code[:45]}",
            safe,
            f"wrongly blocked: {reason}" if not safe else ""
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all(csv_path=None, verbose=False, suite="all"):
    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"Loaded: {csv_path} ({df.shape[0]}×{df.shape[1]})")
    else:
        df = pd.read_csv(StringIO(SAMPLE_CSV))
        print("Using built-in sample dataset (6 rows × 4 cols)")

    all_results = []

    if suite in ("all", "shortcuts"):
        print("\n── SHORTCUT DETECTION TESTS ──────────────────────────────")
        for r in run_shortcut_tests(df, verbose):
            print(f"  {r}")
            all_results.append(r)

    if suite in ("all", "cleaning"):
        print("\n── CLEANING PIPELINE TESTS ────────────────────────────────")
        for r in run_cleaning_tests(verbose):
            print(f"  {r}")
            all_results.append(r)

    if suite in ("all", "security"):
        print("\n── SANDBOX SECURITY TESTS ─────────────────────────────────")
        for r in run_security_tests(verbose):
            print(f"  {r}")
            all_results.append(r)

    passed  = sum(1 for r in all_results if r.passed)
    total   = len(all_results)
    failures = [r for r in all_results if not r.passed]
    pct     = passed / total * 100 if total else 0

    print(f"\n{'='*58}")
    print(f"  RESULT: {passed}/{total} tests passed  ({pct:.0f}%)")
    print(f"{'='*58}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for r in failures:
            print(f"    {r}")

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyst Assistant — Eval Harness")
    parser.add_argument("--csv",     default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--suite",   default="all",
                        choices=["all", "shortcuts", "cleaning", "security"])
    args = parser.parse_args()
    success = run_all(args.csv, args.verbose, args.suite)
    sys.exit(0 if success else 1)
