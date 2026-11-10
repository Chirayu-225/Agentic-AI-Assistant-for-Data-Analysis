"""
test_na_policy_app.py — the "safe" NA policy through the app path: file_reader, cleaning, cache_layer, the report.

The literal text "NA" / "None" is a real value in a column such as TERRITORY (NA = North America) and a missing-value
marker in a numeric or date column. Library code defaults to the legacy behaviour; the app asks for "safe" at ONE choke
point (file_reader.read_tabular_file, and cache_layer.cached_rule_clean for the cleaner).

Works on pandas 2.x and 3.x. No LLM, no network. Run: python test_na_policy_app.py
"""
import ast
import contextlib
import csv
import io
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def csv_bytes(columns: dict, encoding="utf-8") -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(columns))
    for row in zip(*columns.values()):
        w.writerow(row)
    return buf.getvalue().encode(encoding)


def values(df, col):
    return [None if pd.isna(v) else v for v in df[col].tolist()]


def sales_file():
    rng = random.Random(0)
    terr = ["EMEA"] * 1407 + ["NA"] * 1074 + ["APAC"] * 221 + ["Japan"] * 121
    rng.shuffle(terr)
    return csv_bytes({"ORDERNUMBER": [str(10000 + i) for i in range(2823)], "SALES": [f"{100 + i % 50}.5" for i in range(2823)],
                      "TERRITORY": terr}), terr


def main():
    import cache_layer
    import cleaning
    import file_reader
    import missing_tokens as mt

    os.environ.pop("CACHE_ENABLED", None)
    cache_layer.clear_all()
    data, terr = sales_file()
    read_legacy = lambda b, n="t.csv": file_reader._read_tabular_file_uncached(b, n)
    read_safe = lambda b, n="t.csv": file_reader._read_tabular_file_uncached(b, n, na_policy="safe")

    # ---- the reader -------------------------------------------------------------------------------------------------
    legacy = read_legacy(data)
    safe = read_safe(data)
    check("legacy (the default) still loses the 1,074 NA", int(legacy["TERRITORY"].isna().sum()) == 1074)
    check("safe keeps all 1,074 NA as values; nothing missing", values(safe, "TERRITORY") == terr)
    check("safe leaves the numeric columns exactly as legacy",
          safe["SALES"].equals(legacy["SALES"]) and safe["ORDERNUMBER"].equals(legacy["ORDERNUMBER"])
          and safe["SALES"].dtype == legacy["SALES"].dtype)
    check("safe == legacy when the file has no NA tokens at all",
          read_safe(csv_bytes({"a": ["x", "y"], "b": ["1", "2"]})).equals(read_legacy(csv_bytes({"a": ["x", "y"], "b": ["1", "2"]}))))
    try:
        read_safe_bad = file_reader._read_tabular_file_uncached(data, "t.csv", na_policy="bogus")
        check("an unknown policy is refused", False)
    except ValueError:
        check("an unknown policy is refused", True)

    # a numeric column with NA tokens: dtype and missing values identical to legacy
    b = csv_bytes({"qty": [str(i) for i in range(50)] + ["NA"] * 5, "k": ["a"] * 55})
    check("numeric column with NA tokens: identical to legacy (dtype, values, missing)", read_safe(b).equals(read_legacy(b)))

    # leading-zero ids: whatever legacy does, safe does the same
    b = csv_bytes({"code": ["007", "012", "NA", "045"] * 5, "k": ["a"] * 20})
    check("leading-zero id column: identical to legacy", read_safe(b).equals(read_legacy(b)))

    # cp850 file: not valid UTF-8, so the reader falls back to latin1; the second read must use the SAME encoding
    names = ["Müller", "Ösgood", "Zoë", "Ünal"] * 10
    b = csv_bytes({"name": names, "region": ["NA", "EMEA"] * 20}, encoding="cp850")
    try:
        b.decode("utf-8")
        utf8_ok = True
    except UnicodeDecodeError:
        utf8_ok = False
    got = read_safe(b)
    legacy_got = read_legacy(b)
    check("cp850 file (setup: not valid UTF-8)", not utf8_ok)
    check("cp850 file: NA kept, accented text identical to the legacy read",
          values(got, "region") == ["NA", "EMEA"] * 20 and got["name"].tolist() == legacy_got["name"].tolist())

    # bytes straight from an upload buffer (a BytesIO / memoryview / bytearray source)
    buffer = io.BytesIO(data)
    from_buffer = read_safe(buffer.getvalue())
    check("bytes taken from an upload buffer behave the same", from_buffer.equals(safe))

    # a quoted multi-line field
    b = csv_bytes({"note": ["first line\nsecond line", "NA", 'quote " inside', "plain"] * 5, "n": [str(i) for i in range(20)]})
    got = read_safe(b)
    check("quoted multi-line field: intact, NA kept", values(got, "note") == ["first line\nsecond line", "NA", 'quote " inside', "plain"] * 5)

    # fail closed at the file level: stub the SECOND read (the one with usecols) to misalign
    real_read_csv = pd.read_csv

    def run_with_bad_reread(transform):
        def fake(*a, **k):
            out = real_read_csv(*a, **k)
            return transform(out) if "usecols" in k else out
        pd.read_csv = fake
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                frame = read_safe(data)
        finally:
            pd.read_csv = real_read_csv
        return frame, buf.getvalue()

    frame, out = run_with_bad_reread(lambda d: d.iloc[:-1])
    check("misaligned second read (fewer rows): the legacy frame comes back", frame.equals(legacy))
    check("... and the reason is logged", "[file_reader]" in out and "legacy" in out, out[:200])
    frame, out = run_with_bad_reread(lambda d: pd.concat([d.iloc[1:], d.iloc[:1]], ignore_index=True))
    check("shifted second read (same rows): the legacy frame comes back", frame.equals(legacy))

    # Excel is untouched by the policy
    try:
        xbuf = io.BytesIO()
        pd.DataFrame({"t": ["NA", "EMEA", "NA", "APAC"], "n": [1, 2, 3, 4]}).to_excel(xbuf, index=False)
        xb = xbuf.getvalue()
        check("Excel: safe == legacy", file_reader._read_tabular_file_uncached(xb, "t.xlsx", na_policy="safe")
              .equals(file_reader._read_tabular_file_uncached(xb, "t.xlsx")))
    except ImportError:
        print("[SKIPPED] no Excel writer installed")

    # ---- the cleaner -----------------------------------------------------------------------------------------------
    raw = pd.DataFrame({
        "colour": ["red", "na", "blue", "green", "NA", "red", "blue", "None"] * 5,
        "qty": ["1", "2", "na", "4", "5", "NA", "7", "8"] * 5,
        "k": [str(i) for i in range(40)],
    })
    legacy_out, legacy_log = cleaning.rule_based_clean(raw)
    explicit_out, explicit_log = cleaning.rule_based_clean(raw, na_policy="legacy")
    check("cleaner: explicit legacy == default (frame and log)", legacy_out.equals(explicit_out) and legacy_log == explicit_log)
    check("cleaner (legacy): blanks na / NA / None in the category column", int(legacy_out["colour"].isna().sum()) == 15)
    safe_out, safe_log = cleaning.rule_based_clean(raw, na_policy="safe")
    check("cleaner (safe): lowercase 'na', 'NA' and 'None' kept as values in a category column",
          int(safe_out["colour"].isna().sum()) == 0 and "na" in safe_out["colour"].tolist() and "None" in safe_out["colour"].tolist())
    check("cleaner (safe): the numeric-like column still blanks them", int(safe_out["qty"].isna().sum()) == 10
          and pd.api.types.is_numeric_dtype(safe_out["qty"]), str(safe_out["qty"].tolist()[:8]))
    check("cleaner (safe): the log says what was kept",
          any("'colour': kept" in line and "values as real values" in line for line in safe_log), str(safe_log))
    check("cleaner: an unknown policy is refused", _raises(lambda: cleaning.rule_based_clean(raw, na_policy="x")))
    unambiguous = pd.DataFrame({"s": ["a", "N/A", "null", "--", "b", "-", "NULL", "nan"] * 4, "k": list(range(32))})
    check("cleaner (safe): the unambiguous sentinels are blanked exactly as before",
          cleaning.rule_based_clean(unambiguous, na_policy="safe")[0].equals(cleaning.rule_based_clean(unambiguous)[0]))

    # reader notes reach the cleaning log (the marker case: "'author': 3 'NA' treated as missing.")
    b = csv_bytes({"pages": [str(i) for i in range(60)] + ["NA"] * 3, "author": ["Ann", "Bo", "Cy"] * 21})
    frame = file_reader.read_tabular_file(b, "books.csv")        # the choke point: safe
    check("the reader (choke point) is safe", values(frame, "pages") == values(read_legacy(b), "pages"))
    # (a text column of 95%+ numeric strings with a few codes and 3 NA: marker context. A purely numeric column is
    # handled by pandas itself and is never touched, so it carries no note.)
    b = csv_bytes({"author_no": [str(i) for i in range(97)] + ["K1A", "K2B", "K3C"] + ["NA"] * 3, "title": ["x"] * 103})
    frame = file_reader.read_tabular_file(b, "books.csv")
    check("a marker-context token is blanked and the note rides along on the frame",
          int(frame["author_no"].isna().sum()) == 3 and any("treated as missing" in n for n in frame.attrs.get("na_notes", [])),
          str(frame.attrs))
    _, log = cleaning.rule_based_clean(frame, na_policy="safe")
    check("... and the cleaning log shows it", any("'author_no': 3 'NA' treated as missing." in line for line in log), str(log))
    _, log_legacy = cleaning.rule_based_clean(frame)
    check("... but the legacy cleaner log never does", not any("treated as missing" in line for line in log_legacy))

    # ---- the choke point and the cache ---------------------------------------------------------------------------
    cache_layer.clear_all()
    via_app = file_reader.read_tabular_file(data, "t.csv")
    check("read_tabular_file (the app) keeps the NA", values(via_app, "TERRITORY") == terr)
    legacy_cached = cache_layer.cached_parse(data, "t.csv", file_reader._read_tabular_file_uncached)
    check("the parse cache returns DIFFERENT entries for the same bytes under the two policies",
          int(legacy_cached["TERRITORY"].isna().sum()) == 1074 and values(via_app, "TERRITORY") == terr)
    again = file_reader.read_tabular_file(data, "t.csv")
    check("... and the safe entry is a stable cache hit that cannot be mutated through the caller",
          values(again, "TERRITORY") == terr and again is not via_app)

    cache_layer.clear_all()
    seen = []

    def fake_clean(df, na_policy=None):
        seen.append(na_policy)
        return df.copy(), ["log"]

    def fake_clean_legacy_only(df):
        seen.append("legacy-only")
        return df.copy(), ["log"]

    df0 = pd.DataFrame({"a": [1]})
    cache_layer.cached_rule_clean(df0, b"same-bytes", fake_clean, na_policy="safe")
    cache_layer.cached_rule_clean(df0, b"same-bytes", fake_clean, na_policy="legacy")
    check("the cleaned cache has separate entries per policy (the cleaner ran twice)", len(seen) == 2, str(seen))
    check("the policy is passed to the cleaner only when it is not legacy", seen == ["safe", None], str(seen))
    cache_layer.cached_rule_clean(df0, b"same-bytes", fake_clean, na_policy="safe")
    check("a repeat of the same policy is a cache hit", len(seen) == 2)
    cache_layer.clear_all()
    seen.clear()
    cache_layer.cached_rule_clean(df0, b"other-bytes", fake_clean_legacy_only)
    check("a cleaner that does not know about policies (old callers, tests) still works", seen == ["legacy-only"], str(seen))

    # ---- the report ---------------------------------------------------------------------------------------------
    cache_layer.clear_all()
    import profile_service
    report = profile_service.get_cleaning_report(data, "sales.csv")
    log = report.get("log", [])
    check("the cleaning report says territory kept 1,074 'NA' values", any("'territory': kept 1,074 'NA' values as real values." in line for line in log), str(log))
    check("the report no longer counts them as missing", report["before"]["nullValues"] == 0 and report["after"]["nullValues"] == 0, str(report["before"]) + str(report["after"]))
    check("the report has no 'territory ... missing value' warning", not any("'territory' has" in line for line in log))

    # ---- who may call the legacy reader ------------------------------------------------------------------------
    allowed = {"file_reader.py", "clean_adapter_legacy.py", "exp_run.py"}
    offenders = []
    skip_dirs = {"astfs", "node_modules", "__pycache__", "legacy_streamlit", "eval_data", "chroma_data", "logs", ".git"}
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            if not name.endswith(".py") or name.startswith("test_") or name in allowed:
                continue
            try:
                tree = ast.parse(open(os.path.join(root, name), encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                ident = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
                if ident == "_read_tabular_file_uncached":
                    offenders.append(os.path.relpath(os.path.join(root, name), HERE))
                    break
    check("only file_reader.py, clean_adapter_legacy.py and exp_run.py use _read_tabular_file_uncached",
          not offenders, str(offenders))

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
