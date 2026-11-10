"""
test_sandbox_np_shim.py — np.issubdtype inside the sandbox on pandas extension dtypes.

Under pandas 3 numpy's own np.issubdtype raises TypeError for every pandas extension dtype (string, Int64, Float64,
boolean, category, tz-aware datetime) with every kind. The sandbox's np proxy (and ONLY the proxy) falls back to
pandas.api.types for np.number / np.integer / np.floating / np.datetime64, and only when the first argument is a pandas
ExtensionDtype. Anything else re-raises numpy's own TypeError. Global numpy is never patched.

No LLM, no network. Run: python test_sandbox_np_shim.py
"""
import sys

import numpy as np
import pandas as pd

from sandbox_proxy import build_safe_namespace

FAIL = []
ORIGINAL_ISSUBDTYPE = np.issubdtype


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def sub(ns_np, dtype, kind):
    """-> True / False, or the exception type name if it raised."""
    try:
        return ns_np.issubdtype(dtype, kind)
    except Exception as e:
        return type(e).__name__


def main():
    df = pd.DataFrame({"x": [1]})
    npp = build_safe_namespace(df)["np"]          # the sandbox's np, as model code sees it

    text = pd.Series(["a", "b"]).dtype            # pandas 3 default str dtype: StringDtype(na_value=nan)
    check("(a) pandas str dtype, np.number -> False", sub(npp, text, np.number) is False, str(sub(npp, text, np.number)))
    check("(a) 'string' dtype, np.number -> False", sub(npp, pd.StringDtype(), np.number) is False)

    i64 = pd.Series([1, None], dtype="Int64").dtype
    check("(b) Int64: np.number True", sub(npp, i64, np.number) is True)
    check("(b) Int64: np.integer True", sub(npp, i64, np.integer) is True)
    check("(b) Int64: np.floating False", sub(npp, i64, np.floating) is False)

    f64 = pd.Series([1.0, None], dtype="Float64").dtype
    check("(c) Float64: np.number True", sub(npp, f64, np.number) is True)
    check("(c) Float64: np.floating True", sub(npp, f64, np.floating) is True)
    check("(c) Float64: np.integer False", sub(npp, f64, np.integer) is False)

    check("(d) nullable boolean, np.number -> False", sub(npp, pd.BooleanDtype(), np.number) is False)
    check("(d) plain bool, np.number -> False (numpy's own answer)", sub(npp, np.dtype(bool), np.number) is False)

    cat = pd.Series(["a", "b"], dtype="category").dtype
    check("(e) category, np.number -> False", sub(npp, cat, np.number) is False)

    # (f) True is correct because it is a datetime column; numpy's refusal is a limit of its dtype system, not an
    # answer about the data.
    tz = pd.Series(pd.to_datetime(["2024-01-01"]).tz_localize("UTC")).dtype
    check("(f) tz-aware datetime, np.datetime64 -> True", sub(npp, tz, np.datetime64) is True)

    plain = [np.dtype("int64"), np.dtype("float64"), np.dtype("O"), np.dtype(bool), np.dtype("datetime64[ns]")]
    kinds = [np.number, np.integer, np.floating, np.datetime64, np.object_, np.bool_, np.generic]
    same = all(sub(npp, d, k) == sub(np, d, k) for d in plain for k in kinds)
    check("(g) plain numpy dtypes: exactly numpy's own answer for every kind", same)

    # (h) UNMAPPED: numpy raised TypeError AND the second argument is not in the fallback table -> numpy's own error.
    try:
        np.issubdtype(text, np.object_)
        numpy_msg = None
    except TypeError as e:
        numpy_msg = str(e)
    try:
        npp.issubdtype(text, np.object_)
        got = None
    except Exception as e:
        got = e
    check("(h) unmapped second argument re-raises TypeError", isinstance(got, TypeError), repr(got))
    check("(h) ... with the message numpy produced", numpy_msg is not None and got is not None and str(got) == numpy_msg,
          f"{numpy_msg!r} vs {got!r}")
    for k in (np.str_, np.character, np.generic, np.bool_):
        check(f"(h) {k.__name__} on a string dtype re-raises", sub(npp, text, k) == "TypeError")

    # first argument is not a pandas ExtensionDtype -> numpy's TypeError, never a silent pandas answer
    check("a non-dtype first argument re-raises numpy's TypeError", sub(npp, "not-a-dtype", np.number) == "TypeError")

    # (j) the shim lives on the proxy: global numpy is untouched, before and after a sandbox run
    from sandbox import safe_exec
    ns = build_safe_namespace(pd.DataFrame({"t": ["a"], "n": [1]}))
    ok, err = safe_exec("result = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]", ns)
    check("(j) a sandbox run using the shim succeeds", ok and ns.get("result") == ["n"], f"{ok} {err} {ns.get('result')}")
    check("(j) numpy's own np.issubdtype is the same object as before", np.issubdtype is ORIGINAL_ISSUBDTYPE)
    try:
        np.issubdtype(text, np.number)
        still_raises = False
    except TypeError:
        still_raises = True
    check("(j) global numpy still raises on the string dtype (not patched)", still_raises)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
