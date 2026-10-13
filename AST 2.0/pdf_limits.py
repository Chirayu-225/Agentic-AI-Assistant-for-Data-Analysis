"""
pdf_limits.py — wall-clock timeout wrapper for PDF parsing/table
extraction.

WHY THIS EXISTS
    sandbox_limits.py time-boxes LLM-generated code because that code is
    untrusted — it needs an AST guard, an attribute-allowlist proxy, the
    works. parse_pdf() (rag_engine.py) and extract_tables_from_pdf()
    (table_extraction.py) run OUR OWN trusted code, not generated code,
    so none of that security machinery applies here. But they had a
    different, simpler gap: no timeout at all. Both iterate every page
    of a PDF synchronously, inside the request/response cycle, with
    nothing stopping a 500-page document from just running until the
    server (or the platform's serverless execution limit) kills it —
    an ungraceful failure with no useful error message, indistinguishable
    from a hang.

    run_with_timeout() below is the minimal version of what
    sandbox_limits.py does for exec: spawn a subprocess, wait up to a
    limit, kill and return a clean error if it's exceeded. No AST guard,
    no proxy, no memory cap — just the timeout, which is the actual gap
    for trusted-but-potentially-slow code.

WHY A SEPARATE SUBPROCESS, NOT threading/signal.alarm
    signal.alarm only works on the main thread and isn't available on
    Windows at all (this app's dev target). A subprocess is the one
    mechanism that (a) works cross-platform and (b) can actually be
    killed outright if pdfplumber/pypdf gets stuck deep in a C
    extension where a Python-level timeout wouldn't interrupt it.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import traceback
from dataclasses import dataclass
from typing import Any, Callable

_START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"

DEFAULT_PDF_TIMEOUT_SECONDS = 60


@dataclass
class TimedResult:
    success: bool
    value: Any = None
    error: str = ""
    timed_out: bool = False


def _worker(func: Callable, args: tuple, kwargs: dict, result_queue: "mp.Queue") -> None:
    try:
        value = func(*args, **kwargs)
        result_queue.put({"success": True, "value": value})
    except Exception:
        result_queue.put({"success": False, "error": traceback.format_exc(limit=1)})


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: int = DEFAULT_PDF_TIMEOUT_SECONDS,
) -> TimedResult:
    """Runs func(*args, **kwargs) in a subprocess, killing it and
    returning a clean timed_out result if it exceeds timeout_seconds.
    func's return value must be picklable (str, list[dict], etc. — both
    call sites here return exactly that)."""
    kwargs = kwargs or {}
    ctx = mp.get_context(_START_METHOD)
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(func, args, kwargs, result_queue), daemon=True)
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
        return TimedResult(
            success=False,
            timed_out=True,
            error=(
                f"This document took longer than {timeout_seconds}s to process "
                f"and was stopped — it's likely too large or complex for this "
                f"pipeline. Try a smaller file or a shorter page range."
            ),
        )

    proc.join(2)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if not payload.get("success"):
        return TimedResult(success=False, error=payload.get("error", "Unknown error."))

    return TimedResult(success=True, value=payload.get("value"))
