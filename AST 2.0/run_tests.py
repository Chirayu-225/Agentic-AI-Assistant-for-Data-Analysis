"""Run the test scripts under tests/ as modules.   python run_tests.py [name-fragment ...]"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def discover(fragments):
    found = []
    for d, _, files in os.walk(os.path.join(ROOT, "tests")):
        for f in sorted(files):
            if f.startswith("test_") and f.endswith(".py"):
                rel = os.path.relpath(os.path.join(d, f), ROOT)[:-3].replace(os.sep, ".")
                if not fragments or any(x in rel for x in fragments):
                    found.append(rel)
    return sorted(found)


def run(module):
    t0 = time.time()
    try:
        # Piped output on Windows defaults to cp1252, which cannot print the rupee sign or check
        # marks the tests print. Force UTF-8 so a console encoding is never a test failure.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        r = subprocess.run([sys.executable, "-m", module], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env, timeout=300)
        ok, tail = r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:]
    except subprocess.TimeoutExpired:
        ok, tail = False, ["timed out after 300s"]
    return module, ok, time.time() - t0, tail


if __name__ == "__main__":
    mods = discover(sys.argv[1:])
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, mods))
    for m, ok, secs, tail in results:
        print(("PASS " if ok else "FAIL ") + f"{m} ({secs:.1f}s)" + ("" if ok else "  " + (tail[0][:100] if tail else "")))
    failed = [m for m, ok, *_ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)
