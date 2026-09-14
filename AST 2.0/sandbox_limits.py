"""
sandbox_limits.py — Wall-clock timeout and (where available) memory cap
around a single sandboxed execution.

WHY THIS EXISTS
    safe_exec() (sandbox.py) and build_safe_namespace() (sandbox_proxy.py)
    stop code from doing disallowed *things* (imports, file I/O, dangerous
    calls). Neither stops code from doing an allowed thing for too long or
    with too much memory — a `df.groupby(...).apply(slow_func)` on a huge
    frame, or an accidental O(n^2) blow-up, isn't unsafe by pattern, it's
    just expensive. Left unbounded, one such query hangs the whole process
    (Streamlit/Flask is typically single-process, multi-session) — every
    other user is blocked until it finishes or the server restarts.

    This module runs the exec in a SEPARATE PROCESS with:
      - a hard wall-clock timeout (cross-platform)
      - a hard memory cap (Linux/macOS only — see PLATFORM NOTE)
    and kills the worker if either limit is exceeded, returning a normal
    (success=False, error=...) result — the same failure shape the app's
    existing retry logic already knows how to handle.

    A useful side effect of process isolation: matplotlib's global
    "current figure" state no longer leaks between queries or between
    users, since each query gets a brand-new process. The old
    `plt.close("all")` call site in main.py is no longer load-bearing,
    though it's harmless to leave in place.

PLATFORM NOTE
    Memory limiting uses the `resource` module, which only exists on Unix
    (Linux/macOS) — not Windows. On Windows (your dev machine) this module
    still enforces the timeout, but NOT the memory cap. Your deployment
    target (Render, Linux) gets both. This is a real, known gap during
    local development, not a bug — Windows has no equivalent OS-level
    per-process memory limit primitive without third-party tooling.

KNOWN LIMITATION
    Return values cross a process boundary via pickling. pandas
    DataFrames/Series, scalars, and plotly Figures pickle reliably.
    matplotlib Figures are known to be fragile to pickle in some
    environments (backend/canvas state). If a result can't be pickled
    back, the query fails cleanly with a descriptive error instead of
    hanging or crashing — verify this specifically for matplotlib charts
    in your environment (see the test script).
PLATFORM NOTE (part 2)
    This module uses "fork" as the multiprocessing start method when the
    OS supports it (Linux/macOS — i.e. your Render deployment target),
    and falls back to "spawn" only where fork isn't available (Windows).
    This matters beyond performance: "spawn" re-imports whatever Python
    considers the running script's __main__ module to bootstrap the
    child process. Under plain `python script.py` that's your script,
    guarded by `if __name__ == "__main__":`. Under `streamlit run main.py`,
    Streamlit's own launcher — not main.py — is what Python sees as
    __main__, and re-importing that inside a child process is exactly
    the kind of fragile interaction that broke the very first version of
    this module's test script (see the RuntimeError about "bootstrapping
    phase" if you ever see it return). Using "fork" on Linux/macOS
    sidesteps this entirely, since fork clones the current process
    instead of re-importing anything. On Windows, spawn is the only
    option Python provides — if you hit that same RuntimeError running
    the real app locally on Windows, it's this exact issue, and the
    reliable places to verify this layer are (a) the standalone test
    script, run directly with `python test_sandbox_limits.py`, and
    (b) your actual Linux deployment target, where fork applies.

WHY WINDOWS LOCAL DEV CAN HIT THE TIMEOUT ON ORDINARY QUERIES
    "spawn" doesn't just re-import __main__ — it re-imports EVERY module
    this file depends on, fresh, in the new process: pandas, numpy,
    matplotlib, plotly. That cold-import cost alone can eat several
    seconds before the actual analysis code runs at all, on top of
    Windows process-creation overhead (frequently compounded by
    antivirus scanning new python.exe processes). Render (Linux, fork)
    doesn't pay any of this — fork clones already-imported memory
    instantly. Two things below help without weakening production:

    1. Every run now prints timing to stdout (visible in your uvicorn
       terminal) — bootstrap time vs. actual exec time, so a timeout is
       diagnosable instead of a mystery.
    2. SANDBOX_LOCAL_FAST_MODE=1 (env var) skips the subprocess entirely
       for local dev — runs safe_exec() directly in-process. You keep
       every content-based protection (AST/pattern guard, the allowlist
       proxy, SAFE_BUILTINS) — you only lose the OS-level timeout/memory
       kill-switch and the process-boundary isolation itself. That's a
       real, deliberate trade-off: acceptable for a solo dev iterating
       locally against their own data, NOT acceptable for anything
       public-facing. This must never be set in your Render/production
       environment — there's a loud startup print if it's ever enabled,
       specifically so it's hard to leave on by accident.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from sandbox import safe_exec
from sandbox_proxy import build_safe_namespace, unwrap_result

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend — required inside a worker process
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import resource  # Unix only
    _HAS_RESOURCE = True
except ImportError:
    resource = None
    _HAS_RESOURCE = False

# See PLATFORM NOTE (part 2) above for why fork is preferred over spawn.
_START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"

_FAST_MODE = os.environ.get("SANDBOX_LOCAL_FAST_MODE", "").lower() in ("1", "true", "yes")
if _FAST_MODE:
    print(
        "\n"
        "*** WARNING: SANDBOX_LOCAL_FAST_MODE is enabled ***\n"
        "Sandboxed code now runs WITHOUT process isolation, timeout, or\n"
        "memory limits — only the AST/pattern guard and allowlist proxy\n"
        "still apply. This is a local-dev-only convenience. If you see\n"
        "this in a deployed/production log, turn it off immediately.\n",
        flush=True,
    )


DEFAULT_TIMEOUT_SECONDS = 10
# RLIMIT_AS caps the WHOLE process's virtual address space, not just new
# allocations from this exec. On "fork" (this module's default on
# Linux/macOS), the child inherits every page the parent already had
# mapped — pandas/numpy/plotly/matplotlib alone measure ~390MB of virtual
# size before a single sandboxed line runs, and that baseline only grows
# once chromadb/sentence-transformers get lazily imported by a RAG
# request earlier in the server's life. A 512MB cap left almost no
# headroom for that baseline, so ordinary exec (including just pickling
# a Plotly figure to send back through the result queue) could trip the
# limit and die with a bare MemoryError — which surfaces as a silent
# feeder-thread crash and looks exactly like a hung/timed-out worker,
# not a memory error. 1536MB clears realistic baseline + headroom for
# working data while still catching genuine runaway allocations (an
# accidental O(n^2) blow-up is typically many GB, not a few hundred MB).
DEFAULT_MEMORY_MB = 1536


@dataclass
class SandboxResult:
    success: bool
    value: Any = None          # the resolved result / cleaned_df, already unwrapped
    produced: bool = False     # True if a usable value was found at all
    auto_captured: bool = False  # True if value came from an implicit plt.gcf(), not an explicit `result =`
    error: str = ""


def _is_plotly_fig(obj) -> bool:
    t = str(type(obj))
    return "plotly" in t and "Figure" in t


def _apply_memory_limit(memory_mb: int) -> None:
    """Best-effort hard memory cap for the current process. No-op on Windows."""
    if not _HAS_RESOURCE:
        return
    limit_bytes = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError):
        # Some platforms restrict or ignore RLIMIT_AS — fail open rather
        # than crash the worker during setup.
        pass


def _run_core(code: str, df: pd.DataFrame, include_charts: bool) -> dict:
    """
    The actual exec logic, shared by both the subprocess worker (normal
    mode) and the in-process fast path (SANDBOX_LOCAL_FAST_MODE). Returns
    a plain dict (not a SandboxResult) since this also has to survive
    being pickled across the process boundary in normal mode.
    """
    ns = build_safe_namespace(df, include_charts=include_charts)
    success, error = safe_exec(code, ns)

    if not success:
        # safe_exec() catches all exceptions internally and returns
        # str(e) — for a bare MemoryError, str(e) is "". Without this,
        # a memory-limit failure would silently report no error at all.
        if not error:
            error = (
                "Execution failed with no error detail — this usually "
                "means the memory limit was hit during allocation."
            )
        return {"success": False, "error": error}

    value = None
    produced = False
    auto_captured = False

    if "cleaned_df" in ns:
        value = unwrap_result(ns["cleaned_df"])
        produced = True
    elif "result" in ns:
        value = unwrap_result(ns["result"])
        produced = True
        if (
            plt is not None
            and not hasattr(value, "patch")
            and not _is_plotly_fig(value)
            and plt.get_fignums()
        ):
            value = plt.gcf()
            auto_captured = True
    elif plt is not None and plt.get_fignums():
        value = plt.gcf()
        produced = True
        auto_captured = True

    return {
        "success": True, "error": "",
        "produced": produced, "auto_captured": auto_captured,
        "value": value,
    }


def _worker(code: str, df: pd.DataFrame, memory_mb: int, include_charts: bool,
            result_queue: "mp.Queue") -> None:
    """
    Runs entirely inside the child process.
    Only picklable plain values cross the process boundary going in
    (code string, DataFrame, ints/bools) — no proxy objects are ever
    pickled, sidestepping any cross-process pickling issues with them.
    """
    t_worker_start = time.time()
    print(f"[sandbox worker pid={os.getpid()}] started (module imports already loaded by now)", flush=True)

    _apply_memory_limit(memory_mb)
    try:
        payload = _run_core(code, df, include_charts)
        exec_seconds = time.time() - t_worker_start
        print(f"[sandbox worker pid={os.getpid()}] exec finished in {exec_seconds:.2f}s "
              f"(success={payload['success']})", flush=True)

        try:
            result_queue.put(payload)
        except Exception:
            # Value couldn't be pickled back across the process boundary
            # (seen with some matplotlib backend states). Report it as a
            # normal execution failure — the app's existing retry/error
            # UI already handles this shape.
            result_queue.put({
                "success": False,
                "error": "Result could not be returned from the sandbox "
                         "(unsupported object for this chart/result type).",
            })
    except MemoryError:
        result_queue.put({"success": False, "error": "Memory limit exceeded."})
    except Exception:
        result_queue.put({"success": False, "error": traceback.format_exc(limit=1)})


def run_sandboxed(
    code: str,
    df: pd.DataFrame,
    include_charts: bool = True,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SandboxResult:
    """
    Drop-in replacement for the app's previous pattern of:
        local_vars = build_safe_namespace(df)
        success, error = safe_exec(code, local_vars)
        result = unwrap_result(local_vars["result"])

    Runs the exec in a separate process with a wall-clock timeout and
    (on Linux/macOS) a memory cap, and returns a SandboxResult instead of
    a raw (success, error) tuple + a namespace dict to pick apart.

    See SANDBOX_LOCAL_FAST_MODE in this module's docstring for the
    local-dev opt-out of process isolation entirely.
    """
    if _FAST_MODE:
        t0 = time.time()
        payload = _run_core(code, df, include_charts)
        print(f"[sandbox fast-mode] exec finished in {time.time() - t0:.2f}s "
              f"(success={payload['success']}) — no timeout/memory enforcement applied", flush=True)
        if not payload["success"]:
            return SandboxResult(success=False, error=payload["error"])
        return SandboxResult(
            success=True,
            value=payload.get("value"),
            produced=payload.get("produced", False),
            auto_captured=payload.get("auto_captured", False),
        )

    t0 = time.time()
    ctx = mp.get_context(_START_METHOD)
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(code, df, memory_mb, include_charts, result_queue),
        daemon=True,
    )
    print(f"[sandbox] spawning worker (start_method={_START_METHOD}, timeout={timeout_seconds}s)...", flush=True)
    proc.start()

    # Read from the queue BEFORE joining the process — not after. A
    # multiprocessing.Queue's feeder thread writes the pickled payload
    # through an OS pipe with a small buffer (~64KB on Linux); once a
    # result exceeds that (routine for a chart's Plotly JSON or any
    # sizeable table), the child blocks mid-write once the pipe fills.
    # join()-before-get() then deadlocks: the parent waits for the child
    # to exit, the child waits for the parent to drain the pipe. This
    # looks exactly like a timeout — the worker's own "exec finished"
    # print shows it succeeded well under the limit — but is this queue
    # anti-pattern (see the multiprocessing docs' warning about joining
    # before a queue is drained). get(timeout=...) unblocks the moment
    # data arrives, draining the pipe so the child can finish writing
    # and exit normally.
    try:
        payload = result_queue.get(timeout=timeout_seconds)
    except _queue.Empty:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
        print(f"[sandbox] TIMED OUT after {time.time() - t0:.2f}s wall time "
              f"(limit was {timeout_seconds}s) — see worker prints above, if any, "
              f"for how far it got before being killed", flush=True)
        return SandboxResult(
            success=False,
            error=(
                f"Query exceeded the {timeout_seconds}s execution limit and "
                f"was stopped. Try a simpler query or a smaller dataset."
            ),
        )

    print(f"[sandbox] total wall time (spawn + exec + return): {time.time() - t0:.2f}s", flush=True)
    proc.join(2)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if not payload.get("success"):
        return SandboxResult(success=False, error=payload.get("error", "Unknown sandbox error."))

    return SandboxResult(
        success=True,
        value=payload.get("value"),
        produced=payload.get("produced", False),
        auto_captured=payload.get("auto_captured", False),
    )


# ─────────────────────────────────────────────────────────────────────────
# MULTI-TABLE SQL SUPPORT — a parallel worker/entry-point, not a
# generalisation of the single-`df` one above.
#
# Deliberately a near-duplicate of _worker/run_sandboxed's subprocess
# lifecycle (spawn, timeout, queue-drain-before-join, memory cap) rather
# than a shared abstraction the two funnel through. That subprocess
# lifecycle is exactly the code that had two real, non-obvious bugs
# found and fixed in it this session (the join-before-get queue
# deadlock, the RLIMIT_AS-too-tight-for-baseline-memory issue) — under
# time pressure, duplicating a proven, battle-tested pattern is safer
# than risking a shared abstraction subtly breaking both call sites at
# once. What's genuinely different here (DuckDB connection setup/
# teardown, no single `df`) lives in its own worker; what's proven and
# unrelated to that difference (the process lifecycle) is copied
# verbatim, not re-derived.
# ─────────────────────────────────────────────────────────────────────────

def _run_sql_core(code: str, named_dfs: dict) -> dict:
    """Same shape as _run_core, but builds a DuckDB connection with all
    named tables registered instead of a single `df`. See
    sandbox_proxy.SafeDuckDBConnection's docstring for why
    enable_external_access=False is the real safety boundary here, not
    a SQL-keyword denylist."""
    import duckdb
    from sandbox_proxy import build_safe_namespace_multi

    con = duckdb.connect(":memory:", config={"enable_external_access": False})
    for alias, df in named_dfs.items():
        con.register(alias, df)

    ns = build_safe_namespace_multi(con)
    success, error = safe_exec(code, ns)

    if not success:
        if not error:
            error = (
                "Execution failed with no error detail — this usually "
                "means the memory limit was hit during allocation."
            )
        return {"success": False, "error": error}

    value = None
    produced = False
    if "result" in ns:
        value = unwrap_result(ns["result"])
        produced = True

    return {"success": True, "error": "", "produced": produced, "value": value}


def _sql_worker(code: str, named_dfs: dict, memory_mb: int, result_queue: "mp.Queue") -> None:
    t_worker_start = time.time()
    print(f"[sql-sandbox worker pid={os.getpid()}] started with {len(named_dfs)} table(s): "
          f"{list(named_dfs.keys())}", flush=True)

    _apply_memory_limit(memory_mb)
    try:
        payload = _run_sql_core(code, named_dfs)
        exec_seconds = time.time() - t_worker_start
        print(f"[sql-sandbox worker pid={os.getpid()}] exec finished in {exec_seconds:.2f}s "
              f"(success={payload['success']})", flush=True)
        try:
            result_queue.put(payload)
        except Exception:
            result_queue.put({
                "success": False,
                "error": "Result could not be returned from the sandbox "
                         "(unsupported object for this result type).",
            })
    except MemoryError:
        result_queue.put({"success": False, "error": "Memory limit exceeded."})
    except Exception:
        result_queue.put({"success": False, "error": traceback.format_exc(limit=1)})


def run_sql_sandboxed(
    code: str,
    named_dfs: dict,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SandboxResult:
    """Multi-table equivalent of run_sandboxed() — see that function's
    docstring for the general contract. `named_dfs` maps each dataset's
    alias (e.g. "orders", "customers") to its DataFrame; generated code
    accesses them only via `con.sql("... FROM orders JOIN customers ...")`,
    never as direct Python variables (see build_safe_namespace_multi)."""
    if _FAST_MODE:
        t0 = time.time()
        payload = _run_sql_core(code, named_dfs)
        print(f"[sql-sandbox fast-mode] exec finished in {time.time() - t0:.2f}s "
              f"(success={payload['success']}) — no timeout/memory enforcement applied", flush=True)
        if not payload["success"]:
            return SandboxResult(success=False, error=payload["error"])
        return SandboxResult(success=True, value=payload.get("value"), produced=payload.get("produced", False))

    t0 = time.time()
    ctx = mp.get_context(_START_METHOD)
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_sql_worker,
        args=(code, named_dfs, memory_mb, result_queue),
        daemon=True,
    )
    print(f"[sql-sandbox] spawning worker (start_method={_START_METHOD}, timeout={timeout_seconds}s)...", flush=True)
    proc.start()

    # Same queue-drain-before-join ordering as run_sandboxed, and the
    # same reason — see that function's inline comment for the full
    # explanation of the deadlock this avoids.
    try:
        payload = result_queue.get(timeout=timeout_seconds)
    except _queue.Empty:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
        print(f"[sql-sandbox] TIMED OUT after {time.time() - t0:.2f}s wall time "
              f"(limit was {timeout_seconds}s)", flush=True)
        return SandboxResult(
            success=False,
            error=f"Query exceeded the {timeout_seconds}s execution limit and was stopped. Try a simpler query.",
        )

    print(f"[sql-sandbox] total wall time (spawn + exec + return): {time.time() - t0:.2f}s", flush=True)
    proc.join(2)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if not payload.get("success"):
        return SandboxResult(success=False, error=payload.get("error", "Unknown sandbox error."))

    return SandboxResult(success=True, value=payload.get("value"), produced=payload.get("produced", False))
