"""
test_error_scrub.py — error_scrub.scrub_error_text / scrub_result: no traceback, file path or file name reaches the
user, while useful model-code feedback ("KeyError: 'sales'") stays.

Pure, stdlib only, no network. Run: python test_error_scrub.py
"""
import copy
import sys

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


WIN = r"C:\Users\x\secret.py"
WIN_SPACES = r"C:\Users\Test\Downloads\AST-work\AST 2.0\sandbox_limits.py"
POSIX = "/home/u/app/x.py"

TRACEBACK = (
    "Traceback (most recent call last):\n"
    f'  File "{WIN}", line 12, in run\n'
    "    result = df['sales'].sum()\n"
    "             ^^^^^^^^^^^^^^^^^\n"
    f'  File "{POSIX}", line 40, in helper\n'
    "    return self._data[key]\n"
    "KeyError: 'sales'"
)


def leaks(text):
    return [bad for bad in ("Traceback", ".py", "C:\\", "/home/", "File \"", "line 12") if bad in text]


def main():
    from error_scrub import scrub_error_text, scrub_result

    # ---- scrub_error_text -------------------------------------------------------------------------------------
    out = scrub_error_text(TRACEBACK)
    check("traceback block dropped: no Traceback, no .py, no C:\\, no /home/, no frame lines", not leaks(out), repr(out))
    check("traceback block: the final exception line is kept", out.strip() == "KeyError: 'sales'", repr(out))
    check("traceback block: source lines are gone", "df['sales'].sum()" not in out and "self._data" not in out, repr(out))

    out = scrub_error_text("Analysis failed: " + TRACEBACK)
    check("a header in the middle of a line: the words before it stay",
          out.startswith("Analysis failed:") and out.rstrip().endswith("KeyError: 'sales'") and not leaks(out), repr(out))

    chained = (TRACEBACK + "\n\nDuring handling of the above exception, another exception occurred:\n\n"
               + TRACEBACK.replace("KeyError: 'sales'", "RuntimeError: worker failed"))
    out = scrub_error_text(chained)
    check("chained tracebacks: both exception lines kept, no frames, no chain phrase",
          "KeyError: 'sales'" in out and "RuntimeError: worker failed" in out and not leaks(out)
          and "During handling" not in out, repr(out))

    check("KeyError line unchanged", scrub_error_text("KeyError: 'sales'") == "KeyError: 'sales'")
    check("ValueError line unchanged", scrub_error_text("ValueError: Column not found") == "ValueError: Column not found")
    check("a long clean sentence is unchanged", scrub_error_text("Could not parse the date column 'when' (3 values).")
          == "Could not parse the date column 'when' (3 values).")

    out = scrub_error_text(f"Cannot open {WIN} for reading")
    check("Windows path removed, rest of the sentence readable",
          out.startswith("Cannot open ") and out.endswith(" for reading") and "C:" not in out and "secret" not in out, repr(out))
    out = scrub_error_text(f"failed in {WIN_SPACES} at once")
    check("Windows path with spaces in a folder name removed completely",
          "C:" not in out and "sandbox_limits" not in out and "2.0" not in out and out.endswith(" at once"), repr(out))
    out = scrub_error_text(r"copy from C:/Users/x/in/secret.py to D:\out\file.csv")
    check("forward-slash Windows path and a non-.py file path removed", "C:" not in out and "D:" not in out and "secret" not in out, repr(out))
    out = scrub_error_text(f"Cannot import {POSIX} here")
    check("POSIX .py path removed, sentence readable", out.startswith("Cannot import ") and out.endswith(" here") and "/home" not in out
          and ".py" not in out, repr(out))
    out = scrub_error_text("error in sandbox_limits.py while running")
    check("a bare file name ending in .py is removed", ".py" not in out and out.startswith("error in ") and out.endswith("while running"), repr(out))
    out = scrub_error_text('bad thing File "<string>", line 3, in <module> happened')
    check('a File "<...>" fragment is removed', 'File "' not in out and out.startswith("bad thing") and out.endswith("happened"), repr(out))
    out = scrub_error_text("duckdb: no files match /tmp/uploads/abc123.csv at all")
    check("a POSIX path under a system directory is removed even if it is not .py", "/tmp" not in out and out.endswith(" at all"), repr(out))
    check("a plain fraction or URL-like text with slashes is left alone",
          scrub_error_text("ratio 3/4 and a/b") == "ratio 3/4 and a/b")

    out = scrub_error_text(f"Analysis failed: KeyError: 'x' (first attempt: ValueError: no column 'y' in {WIN} line 7)")
    check("'(first attempt: ...)' text keeps its words but loses the path",
          "(first attempt: ValueError: no column 'y' in" in out and "KeyError: 'x'" in out and not leaks(out), repr(out))

    for v in (None, 5, 2.5, True, ["a"], {"a": 1}, b"bytes"):
        check(f"non-str {type(v).__name__} returned unchanged", scrub_error_text(v) is v)

    for sample in (TRACEBACK, chained, f"Cannot open {WIN} now", "KeyError: 'sales'", "x" * 900, "a\n" * 400):
        once = scrub_error_text(sample)
        check(f"idempotent ({sample[:18]!r}...)", scrub_error_text(once) == once)

    out = scrub_error_text("x" * 900)
    check("capped at 500 characters with an ellipsis", len(out) == 500 and out.endswith("..."), str(len(out)))
    check("exactly 500 characters is not cut", scrub_error_text("y" * 500) == "y" * 500)
    out = scrub_error_text(TRACEBACK + "\n" + "z" * 900)
    check("the cap applies after the traceback is dropped", len(out) <= 500 and out.startswith("KeyError: 'sales'"), repr(out[:60]))

    # ---- scrub_result -----------------------------------------------------------------------------------------
    nested = {
        "success": False,
        "error": TRACEBACK,
        "name": TRACEBACK,                       # not an error key: untouched
        "charts": [
            {"title": "ok chart", "charts_error": f"boom in {WIN}", "chartsError": TRACEBACK},
            {"title": "fine", "resultData": {"error_rate": f"{WIN} is data, not an error"}},
        ],
        "meta": {"lastError": POSIX, "count": 3, "ok": True, "none": None},
        "errors": [f"one {WIN}", "two"],
        "ERROR_TEXT": f"upper {WIN}",
    }
    before = copy.deepcopy(nested)
    out = scrub_result(nested)
    check("input is not mutated", nested == before)
    check("returns a NEW structure", out is not nested and out["charts"] is not nested["charts"]
          and out["charts"][0] is not nested["charts"][0])
    check("top-level 'error' scrubbed", out["error"] == "KeyError: 'sales'", repr(out["error"]))
    check("a key without 'error' is untouched", out["name"] == TRACEBACK)
    check("nested list of dicts: charts_error scrubbed", ".py" not in out["charts"][0]["charts_error"] and "C:" not in out["charts"][0]["charts_error"])
    check("nested list of dicts: chartsError (camelCase) scrubbed", out["charts"][0]["chartsError"] == "KeyError: 'sales'")
    check("case-insensitive key match (ERROR_TEXT, lastError)", "C:" not in out["ERROR_TEXT"] and ".py" not in out["meta"]["lastError"])
    check("a list of strings under an 'errors' key is scrubbed", all(".py" not in s and "C:" not in s for s in out["errors"]) and out["errors"][1] == "two")
    check("non-string values are kept as they were", out["meta"]["count"] == 3 and out["meta"]["ok"] is True and out["meta"]["none"] is None)
    check("table data under resultData is never rewritten, even under an 'error_rate' column",
          out["charts"][1]["resultData"] == nested["charts"][1]["resultData"])
    check("scrub_result is idempotent", scrub_result(out) == out)

    clean = {"success": True, "resultType": "scalar", "resultData": 42, "error": None, "code": "result = 42", "rows": [1, 2]}
    check("a success result is returned unchanged (equal, new container)", scrub_result(clean) == clean and scrub_result(clean) is not clean)
    check("scrub_result of a non-container is returned as it is", scrub_result("text") == "text" and scrub_result(7) == 7)

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
