"""
test_na_policy_golden.py — the LEGACY reading / cleaning path must never change.

For every CSV under eval_data (except eval_data/clean_bench/generated and eval_data/exp) it hashes
  * the raw frame from file_reader._read_tabular_file_uncached(bytes, name)      (legacy by default)
  * the output of cleaning.rule_based_clean(raw): the frame AND the log lines     (legacy by default)
and compares them with the constants below, which were computed BEFORE the NA-policy work started. If a hash
differs, the legacy path changed: that is a bug in the change, not a reason to update the constants.

When na_policy exists, an explicit na_policy="legacy" must give exactly the same hashes as the default.
The hashes are specific to the installed pandas (pandas 3.0.5 here); on another pandas version the test prints
[SKIPPED] for the constants and only checks that default and explicit "legacy" agree.

No LLM, no network. Run: python test_na_policy_golden.py
"""
import hashlib
import inspect
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval_data")
EXCLUDED = (os.path.join(EVAL, "clean_bench", "generated"), os.path.join(EVAL, "exp"))
GOLDEN_PANDAS = "3.0.5"

# relative path -> (raw frame hash, cleaned frame + log hash); "ERR:<type>" if reading/cleaning raised
GOLDEN = {
    "heldout/expenses_dirty.csv": ('a96519a9ee81521eade3b866', '67db8b7244f6b1725891c934'),
    "heldout/expenses_questions.csv": ('b0d65b3fab040f5d3b67b1f0', '971c6b84d4ae78834b2aa26c'),
    "heldout/invoices_messy.csv": ('b8bff5702799933912893954', 'ffa230ed5c005ecaa1977d25'),
    "heldout/invoices_questions.csv": ('75cdfa144ab15e483b6da174', '819a22ac945e0968c8c5e2f0'),
    "heldout/store_units_questions.csv": ('ea51bb45912ffeb97bd888b7', 'd061b68063a4eb21c1da37a1'),
    "joins_heldout/customers.csv": ('ed634a15eb073ae5e763ee72', 'f0fa9903d758baa480b0f792'),
    "joins_heldout/join_questions.csv": ('ce39ea3a887882425bd872c8', 'd7bb054e8174f186b4246213'),
    "joins_heldout/order_lines.csv": ('99d1a5b76a4e03a4d3d8ad4b', 'd31fef2499bd0b3b06c76ed2'),
    "joins_heldout/orders.csv": ('939611b28eab80500c4db2ba', '0762f664ffd7f0e4233cf628'),
    "joins_heldout/products.csv": ('26b22969a3c9d149d6a227e5', '755dba7db4a5486dfaa50fd1'),
    "sample_questions.csv": ('fb03788cbbeedf9f6ced36d7', '29e11ee23da541c519017a39'),
    "sample_sales.csv": ('ca66e9c62f4158b2a2596bba', '25460b58b157baf1492ad522'),
}

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def csv_files():
    out = []
    for root, _dirs, files in os.walk(EVAL):
        if any(os.path.abspath(root).startswith(os.path.abspath(x)) for x in EXCLUDED):
            continue
        out += [os.path.join(root, f) for f in files if f.lower().endswith(".csv")]
    return sorted(out)


def frame_hash(df) -> str:
    h = hashlib.sha256()
    h.update(repr([str(c) for c in df.columns]).encode())
    h.update(repr([str(t) for t in df.dtypes]).encode())
    h.update(repr(df.shape).encode())
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        h.update(pd.util.hash_pandas_object(col, index=True).to_numpy().tobytes())
        h.update(repr(col.isna().sum()).encode())
    return h.hexdigest()[:24]


def compute(path, **kwargs):
    """-> (raw hash, cleaned hash) with the legacy reader and cleaner (kwargs: e.g. na_policy='legacy')."""
    import cleaning
    import file_reader
    with open(path, "rb") as f:
        data = f.read()
    name = os.path.basename(path)
    try:
        raw = file_reader._read_tabular_file_uncached(data, name, **kwargs)
        raw_h = frame_hash(raw)
    except Exception as e:
        return "ERR:" + type(e).__name__, "ERR:" + type(e).__name__
    try:
        cleaned, log = cleaning.rule_based_clean(raw, **kwargs)
        h = hashlib.sha256()
        h.update(frame_hash(cleaned).encode())
        h.update("\n".join(log).encode("utf-8"))
        return raw_h, h.hexdigest()[:24]
    except Exception as e:
        return raw_h, "ERR:" + type(e).__name__


def main():
    import cleaning
    import file_reader

    files = csv_files()
    check("found the eval_data CSV files (12)", len(files) == 12, str(len(files)))
    have_policy = ("na_policy" in inspect.signature(file_reader._read_tabular_file_uncached).parameters
                   and "na_policy" in inspect.signature(cleaning.rule_based_clean).parameters)
    same_pandas = pd.__version__ == GOLDEN_PANDAS
    if not same_pandas:
        print(f"[SKIPPED] golden constants are for pandas {GOLDEN_PANDAS}; this is {pd.__version__}")

    for path in files:
        rel = os.path.relpath(path, EVAL).replace(os.sep, "/")
        got = compute(path)
        if same_pandas:
            check(f"legacy raw frame unchanged: {rel}", GOLDEN.get(rel, (None, None))[0] == got[0], f"{got[0]}")
            check(f"legacy cleaned frame + log unchanged: {rel}", GOLDEN.get(rel, (None, None))[1] == got[1], f"{got[1]}")
        if have_policy:
            explicit = compute(path, na_policy="legacy")
            check(f"explicit na_policy='legacy' == default: {rel}", explicit == got, f"{explicit} vs {got}")

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
