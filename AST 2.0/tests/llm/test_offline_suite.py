"""
test_offline_suite.py - the test suite never talks to a real model provider.

Every model call in the tests is stubbed. If a real GEMINI_API_KEY is set in the
shell, the fallback becomes live, and a stubbed primary failure would quietly turn
into a request to Google: the result then depends on whose machine runs the tests,
and test prompts leave the machine. Run: python -m tests.llm.test_offline_suite

1. run_tests.py removes the fallback keys from every test's environment.
2. Run directly (not through run_tests.py) with a Gemini key set, the rate-limit test
   still passes, because it clears the key itself. The key points at a closed local
   port, so if the test did leak into the fallback it would fail at once, with no
   network traffic either way.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" - {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def main():
    import run_tests
    leaky = {"GEMINI_API_KEY": "real-looking-key", "FALLBACK_LLM_API_KEY": "k",
             "FALLBACK_LLM_BASE_URL": "https://example.test/v1", "PATH": os.environ.get("PATH", "")}
    env = run_tests.child_env(leaky)
    check("run_tests.py strips the fallback keys from each test's environment",
          not any(k in env for k in run_tests.OFFLINE_STRIP), str(sorted(env)))
    check("... and keeps everything else", env.get("PATH") == leaky["PATH"])

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               GEMINI_API_KEY="offline-guard-key", GEMINI_BASE_URL="http://127.0.0.1:9/v1")
    r = subprocess.run([sys.executable, "-m", "tests.llm.test_llm_ratelimit"], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    out = (r.stdout + r.stderr).strip().splitlines()
    check("the rate-limit test stays offline even with a Gemini key in the shell",
          r.returncode == 0, " | ".join(l for l in out if "[FAIL]" in l)[:400] or (out[-1] if out else ""))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
