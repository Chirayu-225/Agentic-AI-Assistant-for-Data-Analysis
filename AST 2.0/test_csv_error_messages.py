"""
test_csv_error_messages.py — a file that cannot be read must give the user ONE fixed, safe message, never the
exception text (which can carry a traceback or a server path). The reader is stubbed to raise; no files, no network,
no LLM calls.

Run: python test_csv_error_messages.py
"""
import contextlib
import io

import auto_analyze_service
import profile_service

SAFE = "This file could not be read as a table. Check that it is a valid CSV or Excel file."
LEAK = ('Traceback (most recent call last):\n  File "C:\\Users\\x\\secret.py", line 3, in read\n'
        "ValueError: boom at C:\\Users\\x\\secret.py")

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def boom(*args, **kwargs):
    raise RuntimeError(LEAK)


def run(label, fn, log_prefix):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    err = out.get("error", "")
    check(f"{label}: success is False", out.get("success") is False, str(out))
    check(f"{label}: error is exactly the safe message", err == SAFE, repr(err))
    check(f"{label}: no Traceback / .py / C:\\ in the response",
          "Traceback" not in err and ".py" not in err and "C:\\" not in err, repr(err))
    logged = buf.getvalue()
    check(f"{label}: detail logged server-side with prefix {log_prefix}",
          log_prefix in logged and "secret.py" in logged, logged[:200])


if __name__ == "__main__":
    profile_service._read_csv = boom
    auto_analyze_service._read_csv = boom

    run("get_profile", lambda: profile_service.get_profile(b"x", "t.csv"), "[profile-service]")
    run("get_cleaning_report", lambda: profile_service.get_cleaning_report(b"x", "t.csv"), "[profile-service]")
    run("run_auto_analyze", lambda: auto_analyze_service.run_auto_analyze(b"x", "t.csv", "sid"), "[auto-analyze]")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        raise SystemExit(1)
    print("all passed")
