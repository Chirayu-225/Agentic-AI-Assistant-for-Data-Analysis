"""
The worker -> parent result channel (hazard H3: a compromised worker must not
be able to run code in the API process).

Run from the "AST 2.0" folder:
    python -m tests.execution.test_result_channel

The old channel was a multiprocessing.Queue, which PICKLES. The parent
unpickled whatever the worker sent, and unpickling can execute code. The new
channel carries only bytes in a strict format (JSON + Parquet) and the parent
never unpickles anything.
"""
from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import tempfile
import time

import numpy as np
import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def _run_worker(sl, target, *extra, timeout=30):
    ctx = mp.get_context(sl._START_METHOD)
    chan = sl._make_channel(ctx)
    p = ctx.Process(target=target, args=(chan, *extra), daemon=True)
    p.start()
    chan.close_child_end()
    t0 = time.time()
    try:
        payload = chan.get(timeout=timeout)
    except Exception as e:  # queue.Empty on timeout
        payload = {"_exception": type(e).__name__}
    elapsed = time.time() - t0
    p.join(5)
    if p.is_alive():
        p.kill()
    return payload, elapsed


def main():
    from execution import sandbox_limits as sl
    from tests.execution import _channel_helpers as h

    print("=" * 70)
    print("PART 1 - no pickling channel remains")
    print("=" * 70)
    src = inspect.getsource(sl)
    check("sandbox_limits creates no multiprocessing.Queue (pickle channel)",
          "ctx.Queue(" not in src and "mp.Queue()" not in src,
          f"{src.count('ctx.Queue(')} ctx.Queue() call(s) still present")

    print("\n" + "=" * 70)
    print("PART 2 - a COMPROMISED worker cannot execute code in the parent")
    print("=" * 70)
    marker = os.path.join(tempfile.gettempdir(), "ast_pickle_marker_" + str(os.getpid()))
    if os.path.exists(marker):
        os.remove(marker)
    try:
        payload, _ = _run_worker(sl, h.evil_worker, marker)
    except AttributeError as e:
        payload = None
        check("new result channel exists (_make_channel)", False, str(e))
    if payload is not None:
        check("pickle bomb did NOT run in the parent", not os.path.exists(marker),
              "marker file exists: arbitrary code ran in the API process")
        check("parent reports a clean failure instead", payload.get("success") is False, repr(payload)[:100])
    if os.path.exists(marker):
        os.remove(marker)

    print("\n" + "=" * 70)
    print("PART 3 - a worker that dies is noticed at once, not after the timeout")
    print("=" * 70)
    try:
        payload, elapsed = _run_worker(sl, h.dying_worker, timeout=60)
        check("dead worker -> failure payload", payload.get("success") is False, repr(payload)[:100])
        check("noticed in under 15s (not the 60s timeout)", elapsed < 15, f"{elapsed:.1f}s")
        payload, _ = _run_worker(sl, h.good_worker)
        check("a normal result still round-trips", payload.get("success") is True and payload.get("value") == 42,
              repr(payload)[:100])
    except AttributeError as e:
        check("channel behaves (needs _make_channel)", False, str(e))

    print("\n" + "=" * 70)
    print("PART 4 - the codec round-trips what analysis code really returns")
    print("=" * 70)
    try:
        from execution.sandbox_codec import encode_payload, decode_payload, CodecError
    except ImportError as e:
        check("sandbox_codec module exists", False, str(e))
    else:
        def rt(v):
            out = decode_payload(encode_payload({"success": True, "value": v}))
            return out["value"]

        base = pd.DataFrame({
            "i": [1, 2, 3], "f": [1.5, np.nan, 3.0], "s": ["a", None, "c"],
            "d": pd.to_datetime(["2024-01-01", "2024-06-30", None]),
            "c": pd.Categorical(["x", "y", "x"]), "b": [True, False, True],
        })
        got = rt(base)
        check("DataFrame: values, dtypes, NaN/None/NaT preserved",
              _same_frame(base, got))

        g = pd.DataFrame({"k": ["a", "a", "b"], "j": ["x", "y", "x"], "v": [1, 2, 3]}).groupby(["k", "j"])["v"].sum()
        check("Series with MultiIndex and a name", _same_series(g, rt(g)))
        gf = g.reset_index()
        check("groupby().reset_index() frame", _same_frame(gf, rt(gf)))
        pv = pd.DataFrame({"r": ["a", "a", "b"], "c": ["x", "y", "x"], "v": [1, 2, 3]}).pivot_table(
            index="r", columns="c", values="v", aggfunc="sum")
        check("pivot_table (named column axis)", _same_frame(pv, rt(pv)))
        mi = pd.DataFrame({"r": ["a", "b"], "c": ["x", "y"], "v": [1, 2]}).pivot_table(
            index="r", columns=["c"], values=["v"], aggfunc="sum")
        check("MultiIndex columns", _same_frame(mi, rt(mi)))
        ic = pd.DataFrame({1: [1, 2], 2: [3, 4]})
        check("integer column labels", _same_frame(ic, rt(ic)))
        dup = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        check("duplicate column names", _same_frame(dup, rt(dup)))
        empty = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        check("empty frame", _same_frame(empty, rt(empty)))

        mixed = pd.DataFrame({"m": [1, "two", 3.0]})
        got = rt(mixed)
        check("mixed-type object column is kept, as text (never crashes, never unpickles)",
              list(got["m"].astype(str)) == ["1", "two", "3.0"], repr(got["m"].tolist()))
        iv = pd.DataFrame({"bin": pd.cut(pd.Series([1, 5, 9]), [0, 4, 10])})
        check("pd.cut interval column survives (as text)", len(rt(iv)) == 3)

        check("numpy int64 scalar", int(rt(np.int64(7))) == 7)
        check("float nan scalar", np.isnan(rt(float("nan"))))
        check("str / None / bool scalars", rt("x") == "x" and rt(None) is None and rt(True) is True)
        ts = pd.Timestamp("2024-03-05 10:00", tz="UTC")
        check("Timestamp scalar keeps its value", rt(ts) == ts)
        check("dict and list results", rt({"a": [1, 2.5, "x"], 3: None}) == {"a": [1, 2.5, "x"], 3: None})

        counts, edges = np.histogram([1, 2, 2, 3, 9], bins=3)
        got = rt((counts, edges))
        check("np.histogram result (tuple of ndarrays)",
              isinstance(got, tuple) and np.array_equal(got[0], counts) and np.allclose(got[1], edges)
              and got[0].dtype == counts.dtype)
        arr2 = np.arange(6, dtype="float32").reshape(2, 3)
        got = rt(arr2)
        check("2-D ndarray keeps shape and dtype", isinstance(got, np.ndarray) and got.shape == (2, 3)
              and got.dtype == arr2.dtype and np.array_equal(got, arr2))
        got = rt(np.array(["a", "bb"], dtype=object))
        check("object ndarray does not crash (comes back as a list)", list(got) == ["a", "bb"])
        got = rt(pd.Index(["x", "y"], name="k"))
        check("pd.Index keeps values and name", list(got) == ["x", "y"] and got.name == "k")
        got = rt(pd.DataFrame({"a": [1], "b": [2]}).columns)
        check("df.columns as a result", list(got) == ["a", "b"])
        check("set result", rt({1, 2, 3}) == {1, 2, 3})

        class Weird:
            pass
        try:
            encode_payload({"success": True, "value": Weird()})
            check("an unsupported object is refused, never pickled", False, "it was accepted")
        except CodecError:
            check("an unsupported object is refused, never pickled", True)
        try:
            decode_payload(b"\x80\x04\x95not-our-format")
            check("foreign bytes (a pickle) are rejected by the decoder", False, "accepted")
        except CodecError:
            check("foreign bytes (a pickle) are rejected by the decoder", True)

    print("\n" + "=" * 70)
    print("PART 5 - end to end through run_sandboxed")
    print("=" * 70)
    df = pd.DataFrame({"name": ["a", "b", "a"], "sales": [1, 2, 3]})
    r = sl.run_sandboxed("result = df.groupby('name')['sales'].sum().reset_index()", df, timeout_seconds=60)
    check("table result", r.success and isinstance(r.value, pd.DataFrame) and r.value["sales"].tolist() == [4, 2],
          repr(r.error))
    r = sl.run_sandboxed("result = int(df['sales'].sum())", df, timeout_seconds=60)
    check("scalar result", r.success and r.value == 6, repr(r.error))
    r = sl.run_sandboxed("result = df.groupby('name')['sales'].sum()", df, timeout_seconds=60)
    check("series result", r.success and isinstance(r.value, pd.Series) and r.value.to_dict() == {"a": 4, "b": 2},
          repr(r.error))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


def _same_frame(a: pd.DataFrame, b) -> bool:
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=True, check_column_type=False, check_index_type=False)
        return list(map(str, a.columns)) == list(map(str, b.columns)) and a.columns.nlevels == b.columns.nlevels
    except Exception as e:
        print("      frame mismatch:", str(e).splitlines()[0][:110])
        return False


def _same_series(a: pd.Series, b) -> bool:
    try:
        pd.testing.assert_series_equal(a, b, check_index_type=False)
        return a.name == b.name
    except Exception as e:
        print("      series mismatch:", str(e).splitlines()[0][:110])
        return False


if __name__ == "__main__":
    main()
