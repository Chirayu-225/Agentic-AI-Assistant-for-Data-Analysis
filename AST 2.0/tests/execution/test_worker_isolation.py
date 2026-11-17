"""
Process-isolation tests for the sandbox worker (hazards H3 and H5).

Run from the "AST 2.0" folder:
    python -m tests.execution.test_worker_isolation

What these prove, each as a boundary that holds EVEN IF the code-guard layers
are wrong:
  1. A sandbox worker starts with NO secrets in its environment (API keys,
     tokens, passwords, database URLs) - only an allowlist of harmless
     variables the interpreter needs.
  2. Every worker entry point goes through the same hardening function, so a
     new worker cannot be added that forgets it.
  3. The memory cap is enforced on every platform (including Windows, where
     the old setrlimit-based cap silently did nothing), and breaching it fails
     fast instead of exhausting the host.

Not covered here (needs a container): the worker can still read files on disk,
including .env. That gap is documented in the assurance case.
"""
from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import time

import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


SECRETS = {
    "GROQ_API_KEY": "gsk_test_secret_value",
    "OPENAI_API_KEY": "sk-test-secret-value",
    "DATABASE_URL": "postgres://user:pw@host/db",
    "AWS_SECRET_ACCESS_KEY": "aws-test-secret",
    "MY_SERVICE_TOKEN": "tok_test",
    "SOME_PASSWORD": "pw_test",
    "UNRECOGNISED_VARIABLE": "anything not on the allowlist must go too",
}


def _probe_env(result_queue) -> None:
    """Runs inside a real worker process (importable, so 'spawn' works)."""
    from execution import sandbox_limits as sl
    sl._harden_worker_process(2560)
    result_queue.put(sorted(os.environ.keys()))


def main():
    from execution import sandbox_limits as sl

    print("=" * 70)
    print("PART 1 - worker environment holds no secrets")
    print("=" * 70)
    os.environ.update(SECRETS)
    ctx = mp.get_context(sl._START_METHOD)
    q = ctx.Queue()
    p = ctx.Process(target=_probe_env, args=(q,), daemon=True)
    p.start()
    try:
        keys = q.get(timeout=60)
    except Exception:
        keys = None
    p.join(timeout=10)
    if p.is_alive():
        p.kill()
    check("worker reported its environment (hardening function exists and runs)",
          keys is not None, "worker produced nothing - _harden_worker_process missing or crashed")
    if keys is not None:
        for name in SECRETS:
            check(f"secret not visible in worker: {name}", name not in keys)
        check("interpreter-critical variables survive (PATH)", any(k.upper() == "PATH" for k in keys))

    print("\n" + "=" * 70)
    print("PART 2 - every worker entry point is hardened")
    print("=" * 70)
    for name in ("_worker", "_sql_worker", "_large_csv_worker", "_multi_table_mixed_worker"):
        src = inspect.getsource(getattr(sl, name))
        check(f"{name} calls _harden_worker_process", "_harden_worker_process(" in src)

    print("\n" + "=" * 70)
    print("PART 3 - memory cap is enforced on this platform (" + os.name + ")")
    print("=" * 70)
    df = pd.DataFrame({"a": [1, 2, 3]})
    # ~1.6 GB string against a 1000 MB cap. Must fail fast and say why.
    code = "x = 'a' * 1_600_000_000\nresult = len(x)"
    t0 = time.time()
    r = sl.run_sandboxed(code, df, timeout_seconds=30, memory_mb=1000)
    elapsed = time.time() - t0
    check("allocation past the cap is refused", not r.success,
          f"it SUCCEEDED - the cap did nothing (result={r.value!r})")
    check("refusal is fast (cap, not the timeout, stopped it)", elapsed < 20, f"took {elapsed:.1f}s")

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
