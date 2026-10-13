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

import functools
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
# mapped — pandas/numpy/plotly/matplotlib import alone measures ~390MB of
# virtual size before a single sandboxed line runs, and that baseline
# only grows once chromadb/sentence-transformers get lazily imported by
# a RAG request earlier in the server's life. A 512MB cap left almost no
# headroom for that baseline, so ordinary exec (including just pickling
# a Plotly figure to send back through the result queue) could trip the
# limit and die with a bare MemoryError — which surfaces as a silent
# feeder-thread crash and looks exactly like a hung/timed-out worker,
# not a memory error.
#
# REVISED — was 1536MB, now 2560MB (see task: "revisit RLIMIT_AS sizing
# for the large-CSV path"). 1536MB was sized against import cost alone
# (the paragraph above), which is the wrong baseline to size against.
# The actual, measured baseline is import cost PLUS the first real
# array/DataFrame operation: on this project's numpy build (a standard
# pip wheel linked against OpenBLAS with DYNAMIC_ARCH, MAX_THREADS=64 —
# check with `python -c "import numpy; numpy.show_config()"`, since this
# is a property of the numpy WHEEL, not this code), the first operation
# that actually touches BLAS/array internals reserves its thread-local
# buffers scaled to that compiled MAX_THREADS constant, not to the
# host's real core count — measured on this project's own dev container
# as a jump from ~370MB to ~1.4GB of virtual size from creating a single
# 2-row pandas DataFrame, BEFORE anything sandboxed even runs. That
# reservation is what actually made a trivial large-CSV query (a 2-row
# file, not a large one) intermittently fail with
# `duckdb.OutOfMemoryException: Failed to allocate block of ~32MB` under
# the old 1536MB cap — not dataset size, and not something specific to
# the large-CSV path either (run_sandboxed/run_sql_sandboxed above are
# exposed to the exact same OpenBLAS reservation, since they all fork
# from the same already-numpy-loaded parent; this was a latent risk in
# the already-shipped paths too, just never hit here because their
# workloads happened to stay under the remaining headroom).
#
# 2560MB was chosen after measuring the large-CSV path pass reliably at
# 2048MB across 5 repeated trials in this exact container; 2560MB adds a
# real margin on top of that rather than shipping the bare minimum that
# happened to pass. This number is still an estimate tied to THIS numpy
# build and THIS container — verify it directly against your actual
# Render deployment (same technique: watch for the OutOfMemoryException
# on a trivial query) rather than trusting it blind, since a different
# numpy wheel or a from-source BLAS build could shift the real baseline
# either direction.
DEFAULT_MEMORY_MB = 2560


@dataclass
class SandboxResult:
    success: bool
    value: Any = None          # the resolved result / cleaned_df, already unwrapped
    produced: bool = False     # True if a usable value was found at all
    auto_captured: bool = False  # True if value came from an implicit plt.gcf(), not an explicit `result =`
    error: str = ""


def _audited(kind: str):
    """Writes one audit record per sandbox run (audit_log.record_sandbox):
    what kind of run, whether it succeeded, whether it was a SECURITY BLOCK
    vs an ordinary error vs a timeout, how long it took, and a hash of the
    code. Runs in the PARENT process — the worker subprocess never touches
    the audit log, so the tamper-evident chain stays single-writer.
    Auditing can never change a result or raise: failures here are swallowed."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(code, *args, **kwargs):
            t0 = time.time()
            try:
                result = fn(code, *args, **kwargs)
            except Exception as e:
                _safe_audit(kind, code, False, f"{type(e).__name__}: {e}", t0)
                raise
            _safe_audit(kind, code, result.success, result.error, t0)
            return result
        return wrapper
    return decorator


def _safe_audit(kind: str, code: str, success: bool, error: str, t0: float) -> None:
    try:
        import audit_log
        audit_log.record_sandbox(kind, code, success, error, (time.time() - t0) * 1000)
    except Exception:
        pass


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


@_audited("pandas")
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

    # threads=1 — see LARGE-CSV MEMORY SIZING further down this file for
    # why this isn't a performance default left unset: DuckDB reserves
    # per-thread virtual memory up front, scaled to the HOST's CPU count,
    # not to the data — same fix applied here for the same reason, even
    # though this path was never the one that surfaced the bug.
    con = duckdb.connect(":memory:", config={"enable_external_access": False, "threads": 1})
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


@_audited("sql")
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


# ─────────────────────────────────────────────────────────────────────────
# LARGE-CSV SUPPORT — routes a big single file through DuckDB's own CSV
# engine instead of pandas, so the file is never fully materialized as a
# pandas DataFrame anywhere, in either the parent process or the worker.
#
# WHY THIS IS THE ACTUAL FIX FOR "CAN'T INTAKE BIGGER CSVs"
#     Every other path in this app (run_sandboxed, and run_sql_sandboxed
#     above via named_dfs) requires a pandas DataFrame to already exist
#     BEFORE the sandboxed call even starts — file_reader.py's
#     pd.read_csv() has already paid the full memory cost, and then
#     run_sandboxed/run_sql_sandboxed pickle that whole DataFrame across
#     the process boundary on top of that. Two full in-memory copies,
#     minimum, before a single query byte executes.
#
#     run_large_csv_sql_sandboxed() instead takes a FILE PATH, not a
#     DataFrame — nothing is parsed until it reaches the worker
#     subprocess, and even there, DuckDB's own CSV reader (multi-
#     threaded, C++, columnar) does the parsing — not pandas. The result
#     is loaded as an Arrow table, which is far more memory-compact than
#     pandas' per-cell Python object overhead, especially for string-
#     heavy data.
#
# WHY NOT JUST QUERY THE FILE DIRECTLY WITH enable_external_access=True
#     That would let generated SQL text itself call read_csv/httpfs/
#     ATTACH on arbitrary paths or URLs — exactly the hole
#     enable_external_access=False exists to close (see
#     multi_table_service.py's module docstring). Instead, a SEPARATE,
#     short-lived "loader" connection (created and torn down by this
#     trusted module's own code, never touched by generated SQL) reads
#     the file and hands off an in-memory Arrow table; the connection
#     generated code actually runs against is created fresh, with
#     enable_external_access=False, and only ever sees that pre-loaded
#     table by name. Generated code retains exactly the same inability
#     to read arbitrary files it already had — this only changes how
#     the ONE file the user actually uploaded gets loaded in the first
#     place.
#
# LARGE-CSV MEMORY SIZING — two separate findings, not one
#     Building the test coverage for this path (test_large_csv_service.py)
#     surfaced a real OOM: a query against a 2-ROW csv — not a large one —
#     reliably hit `duckdb.OutOfMemoryException: Failed to allocate block
#     of ~32MB` under the OLD DEFAULT_MEMORY_MB (1536MB). Two distinct
#     things were found chasing it down; only one of them was the actual
#     cause. Both are kept because both are real, separate hardening —
#     see DEFAULT_MEMORY_MB's own comment above for the actual root
#     cause and the corrected constant.
#
#     1. threads=1, pinned on both connections below (the loader and the
#        query connection). This does NOT fix the OOM above — measured
#        directly: raising DEFAULT_MEMORY_MB alone, with threads left
#        unset, also passes reliably. It's kept anyway because it's a
#        separate, legitimate concern: DuckDB's default thread count
#        scales with the HOST's CPU count (`nproc`), so its own resource
#        footprint differs between a dev machine, CI, and whatever tier
#        this ends up deployed on — a config that happens to work locally
#        isn't guaranteed to work in production, and vice versa. Pinning
#        threads=1 makes that piece of the footprint deterministic and
#        host-independent, at the cost of single-threaded CSV parsing —
#        measured as fine: a real 273MB, 4.5M-row CSV (load + Arrow
#        handoff + a GROUP BY query) still ran in ~6s single-threaded,
#        well inside this path's 60s timeout.
#
#     2. DEFAULT_MEMORY_MB was actually undersized — see that constant's
#        comment above for the real cause (an OpenBLAS thread-buffer
#        reservation triggered by the first pandas/numpy operation in the
#        process, unrelated to DuckDB or dataset size) and the corrected
#        value. This was the actual fix for the OOM above.
# ─────────────────────────────────────────────────────────────────────────

LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50MB — see file_reader.should_use_large_csv_path

# A HIGHER ceiling than DEFAULT_MEMORY_MB, used only by run_large_csv_sql_sandboxed
# and run_multi_table_mixed_sql_sandboxed below — deliberately not the same
# constant. DEFAULT_MEMORY_MB is sized to still catch a genuinely
# runaway allocation on what's supposed to be ordinary, small-file work;
# these two functions exist specifically to handle large, real files, so
# holding them to that same tight ceiling defeats their own purpose.
#
# 4608MB is empirically grounded, not a round-number guess: a real-world
# test case that matters more than the earlier numeric-only one (a
# 144MB, 4.5M-row CSV with a padded TEXT column, joined against a small
# lookup table) reliably OOM'd at 2560MB — string-heavy data costs
# noticeably more per row than numeric columns through DuckDB/Arrow, so
# a memory ceiling validated only against numeric data understates what
# real spreadsheets (names, addresses, free-text fields) actually need.
# 3584MB passed; 4608MB passed with more margin — chosen over 3584MB
# specifically so there's real headroom left for files larger or more
# string-heavy than this one test case, not just the exact case tested.
# Like DEFAULT_MEMORY_MB, this number is tied to this project's numpy/
# OpenBLAS build (see that constant's comment) plus the specific test
# file's shape — verify it against your own real large files, not just
# this one synthetic case, before trusting it in production.
LARGE_FILE_MEMORY_MB = 4608


def _load_csv_as_arrow_table(file_path: str, alias: str):
    """Trusted, non-generated code only. Uses a throwaway, permissive
    DuckDB connection purely to let DuckDB's own CSV engine parse the
    file — this connection is created, used, and discarded entirely
    before any generated code exists, and is never exposed to it.

    threads=1 is deliberate, not a performance default left unset — see
    the LARGE-CSV MEMORY SIZING note below this function for the OOM
    this fixes and the measurements behind it."""
    import duckdb

    loader_con = duckdb.connect(":memory:", config={"threads": 1})
    try:
        loader_con.execute(
            f"CREATE TABLE {alias} AS SELECT * FROM read_csv_auto(?, sample_size=-1)",
            [file_path],
        )
        arrow_table = loader_con.table(alias).fetch_arrow_table()
    finally:
        loader_con.close()
    return arrow_table


def _run_large_csv_sql_core(code: str, file_path: str, alias: str) -> dict:
    import duckdb
    from sandbox_proxy import build_safe_namespace_multi

    arrow_table = _load_csv_as_arrow_table(file_path, alias)

    # threads=1 here too — see LARGE-CSV MEMORY SIZING below.
    con = duckdb.connect(":memory:", config={"enable_external_access": False, "threads": 1})
    con.register(alias, arrow_table)

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


def _large_csv_worker(code: str, file_path: str, alias: str, memory_mb: int, result_queue: "mp.Queue") -> None:
    t_worker_start = time.time()
    print(f"[large-csv-sandbox worker pid={os.getpid()}] loading '{file_path}' as '{alias}' "
          f"via DuckDB's CSV engine (not pandas)...", flush=True)

    _apply_memory_limit(memory_mb)
    try:
        payload = _run_large_csv_sql_core(code, file_path, alias)
        exec_seconds = time.time() - t_worker_start
        print(f"[large-csv-sandbox worker pid={os.getpid()}] finished in {exec_seconds:.2f}s "
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
        result_queue.put({"success": False, "error": "Memory limit exceeded loading or querying this file."})
    except Exception:
        result_queue.put({"success": False, "error": traceback.format_exc(limit=1)})


@_audited("large_csv")
def run_large_csv_sql_sandboxed(
    code: str,
    file_path: str,
    alias: str = "data",
    timeout_seconds: int = 60,
    memory_mb: int = LARGE_FILE_MEMORY_MB,
) -> SandboxResult:
    """Large-file counterpart to run_sql_sandboxed(). Takes a file PATH
    (not a pre-parsed DataFrame) — parsing happens inside the worker,
    via DuckDB, not pandas. Longer default timeout than the pandas path
    (60s vs 10s) since loading a genuinely large file takes real time
    even with an efficient engine, and that cost is paid inside this
    call rather than before it. `alias` is the table name generated SQL
    should reference (e.g. `SELECT ... FROM data`)."""
    if _FAST_MODE:
        t0 = time.time()
        payload = _run_large_csv_sql_core(code, file_path, alias)
        print(f"[large-csv-sandbox fast-mode] finished in {time.time() - t0:.2f}s "
              f"(success={payload['success']}) — no timeout/memory enforcement applied", flush=True)
        if not payload["success"]:
            return SandboxResult(success=False, error=payload["error"])
        return SandboxResult(success=True, value=payload.get("value"), produced=payload.get("produced", False))

    t0 = time.time()
    ctx = mp.get_context(_START_METHOD)
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_large_csv_worker,
        args=(code, file_path, alias, memory_mb, result_queue),
        daemon=True,
    )
    print(f"[large-csv-sandbox] spawning worker (start_method={_START_METHOD}, timeout={timeout_seconds}s)...", flush=True)
    proc.start()

    try:
        payload = result_queue.get(timeout=timeout_seconds)
    except _queue.Empty:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
        print(f"[large-csv-sandbox] TIMED OUT after {time.time() - t0:.2f}s wall time "
              f"(limit was {timeout_seconds}s)", flush=True)
        return SandboxResult(
            success=False,
            error=(
                f"Loading or querying this file exceeded the {timeout_seconds}s "
                f"limit and was stopped. Try a smaller file or a simpler query."
            ),
        )

    print(f"[large-csv-sandbox] total wall time (spawn + exec + return): {time.time() - t0:.2f}s", flush=True)
    proc.join(2)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if not payload.get("success"):
        return SandboxResult(success=False, error=payload.get("error", "Unknown sandbox error."))

    return SandboxResult(success=True, value=payload.get("value"), produced=payload.get("produced", False))


# ─────────────────────────────────────────────────────────────────────────
# MIXED MULTI-TABLE SQL SUPPORT — a join where SOME tables are already
# pandas DataFrames (small files, already loaded by multi_table_service.py
# the normal way) and OTHERS are file paths (large CSV/TXT files that
# should never be pandas-loaded at all).
#
# WHY THIS EXISTS SEPARATELY FROM run_sql_sandboxed / run_large_csv_sql_sandboxed
#     Before this, multi_table_service.py called read_tabular_file() —
#     full pd.read_csv()/read_excel() — for EVERY dataset in a join,
#     regardless of size, then handed the whole dict to run_sql_sandboxed.
#     That meant a join was only ever as scalable as its SMALLEST-capacity
#     table: one large file in a join hit the exact same ceiling
#     run_large_csv_sql_sandboxed exists to remove from the single-dataset
#     path. A join is also the one place BOTH situations can be true at
#     once — a small lookup table alongside one genuinely large fact
#     table — so neither existing function's dict/file-path signature
#     fits: this one takes both, explicitly separated by caller, and
#     loads only the file-backed ones through DuckDB's CSV engine inside
#     the worker, exactly like run_large_csv_sql_sandboxed does for a
#     single file.
#
# WHY TWO SEPARATE DICTS, NOT ONE DICT OF MIXED VALUE TYPES
#     A single `Union[pd.DataFrame, str]`-valued dict would let a caller
#     accidentally pass a DataFrame where a path was meant (or vice
#     versa) and only fail at runtime, deep inside the worker. Two
#     explicitly-typed dicts make the caller's intent for each table
#     unambiguous at the call site, and match this module's existing
#     preference (see run_sql_sandboxed's docstring) for a duplicated,
#     explicit pattern over a clever shared one.
# ─────────────────────────────────────────────────────────────────────────

def _run_multi_table_mixed_core(code: str, dataframe_tables: dict, file_tables: dict) -> dict:
    """Same shape as _run_sql_core, but registers BOTH kinds of table on
    one connection before generated code ever runs: DataFrames directly
    (con.register on the actual data, already in memory), and file-backed
    tables via the same trusted loader used by run_large_csv_sql_sandboxed
    (_load_csv_as_arrow_table) — never through pandas."""
    import duckdb
    from sandbox_proxy import build_safe_namespace_multi

    # threads=1 — see LARGE-CSV MEMORY SIZING above run_large_csv_sql_sandboxed.
    con = duckdb.connect(":memory:", config={"enable_external_access": False, "threads": 1})
    for alias, df in dataframe_tables.items():
        con.register(alias, df)
    for alias, file_path in file_tables.items():
        arrow_table = _load_csv_as_arrow_table(file_path, alias)
        con.register(alias, arrow_table)

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


def _multi_table_mixed_worker(code: str, dataframe_tables: dict, file_tables: dict,
                               memory_mb: int, result_queue: "mp.Queue") -> None:
    t_worker_start = time.time()
    print(f"[multi-table-mixed-sandbox worker pid={os.getpid()}] started with "
          f"{len(dataframe_tables)} in-memory table(s) {list(dataframe_tables.keys())} and "
          f"{len(file_tables)} file-backed table(s) {list(file_tables.keys())}", flush=True)

    _apply_memory_limit(memory_mb)
    try:
        payload = _run_multi_table_mixed_core(code, dataframe_tables, file_tables)
        exec_seconds = time.time() - t_worker_start
        print(f"[multi-table-mixed-sandbox worker pid={os.getpid()}] finished in {exec_seconds:.2f}s "
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
        result_queue.put({"success": False, "error": "Memory limit exceeded loading or querying these tables."})
    except Exception:
        result_queue.put({"success": False, "error": traceback.format_exc(limit=1)})


@_audited("multi_table")
def run_multi_table_mixed_sql_sandboxed(
    code: str,
    dataframe_tables: dict,
    file_tables: dict,
    timeout_seconds: int = 60,
    memory_mb: int = LARGE_FILE_MEMORY_MB,
) -> SandboxResult:
    """Join-time equivalent of run_large_csv_sql_sandboxed(): registers
    `dataframe_tables` (alias -> already-loaded pandas DataFrame, for
    small files) and `file_tables` (alias -> file path, for large CSV/TXT
    files — loaded via DuckDB's own CSV engine inside the worker, never
    through pandas) on the SAME locked-down connection, so a single query
    can join across both. `timeout_seconds` defaults to 60, not this
    module's usual 10s, for the same reason as run_large_csv_sql_sandboxed:
    real file I/O can happen inside this call whenever file_tables is
    non-empty."""
    if _FAST_MODE:
        t0 = time.time()
        payload = _run_multi_table_mixed_core(code, dataframe_tables, file_tables)
        print(f"[multi-table-mixed-sandbox fast-mode] finished in {time.time() - t0:.2f}s "
              f"(success={payload['success']}) — no timeout/memory enforcement applied", flush=True)
        if not payload["success"]:
            return SandboxResult(success=False, error=payload["error"])
        return SandboxResult(success=True, value=payload.get("value"), produced=payload.get("produced", False))

    t0 = time.time()
    ctx = mp.get_context(_START_METHOD)
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_multi_table_mixed_worker,
        args=(code, dataframe_tables, file_tables, memory_mb, result_queue),
        daemon=True,
    )
    print(f"[multi-table-mixed-sandbox] spawning worker (start_method={_START_METHOD}, "
          f"timeout={timeout_seconds}s)...", flush=True)
    proc.start()

    # Same queue-drain-before-join ordering as run_sandboxed/run_sql_sandboxed
    # — see run_sandboxed's inline comment for the deadlock this avoids.
    try:
        payload = result_queue.get(timeout=timeout_seconds)
    except _queue.Empty:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
        print(f"[multi-table-mixed-sandbox] TIMED OUT after {time.time() - t0:.2f}s wall time "
              f"(limit was {timeout_seconds}s)", flush=True)
        return SandboxResult(
            success=False,
            error=(
                f"Loading or querying these tables exceeded the {timeout_seconds}s "
                f"limit and was stopped. Try a smaller file or a simpler query."
            ),
        )

    print(f"[multi-table-mixed-sandbox] total wall time (spawn + exec + return): {time.time() - t0:.2f}s", flush=True)
    proc.join(2)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if not payload.get("success"):
        return SandboxResult(success=False, error=payload.get("error", "Unknown sandbox error."))

    return SandboxResult(success=True, value=payload.get("value"), produced=payload.get("produced", False))
