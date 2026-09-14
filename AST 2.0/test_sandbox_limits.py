"""
Functional proof for sandbox_limits.py.

Confirms:
  1. A slow-but-not-literally-"while True" query gets killed by the
     wall-clock timeout instead of hanging forever.
  2. A memory-hungry query gets stopped by the memory cap (Linux/macOS).
  3. Normal fast queries still work and return in well under the timeout.
  4. A matplotlib chart result survives the process boundary (the one
     documented risk in sandbox_limits.py).

Run: python3 test_sandbox_limits.py
"""

import time
import pandas as pd

from sandbox_limits import run_sandboxed

def main():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Dave"],
        "sales": [1200, 2500, 500, 1800],
        "category": ["Electronics", "Clothing", "Electronics", "Clothing"],
    })

    print("=" * 70)
    print("PART 1 — wall-clock timeout")
    print("=" * 70)

    # Deliberately not "while True" / "while 1" — a finite-but-huge loop,
    # exactly the kind of thing the string-pattern guard can't catch.
    # Avoids bare builtins (len/sum/range) since those are unavailable in
    # the sandbox by design (exec() runs with __builtins__: {}) — a
    # separate, pre-existing issue, not part of this fix.
    slow_code = (
        "i = 0\n"
        "total = 0\n"
        "while i < 3000:\n"
        "    total = total + np.sum(np.arange(1_000_000))\n"
        "    i = i + 1\n"
        "result = 1"
    )
    # unrestricted, this takes ~3.2s — set the cap below that to prove the kill

    t0 = time.time()
    r = run_sandboxed(slow_code, df.copy(), timeout_seconds=2)
    elapsed = time.time() - t0
    status = "BLOCKED (correct)" if not r.success else "!!! NOT BLOCKED !!!"
    print(f"\n[{status}] slow loop (~3.2s unrestricted), 2s timeout")
    print(f"    wall time observed: {elapsed:.1f}s (should be ~2s, not the ~3.2s the loop needs)")
    print(f"    error: {r.error}")

    print("\n" + "=" * 70)
    print("PART 2 — memory cap (Linux/macOS only)")
    print("=" * 70)

    mem_bomb_code = "result = [0] * (300_000_000)"  # ~2.4GB of Python ints/refs — should blow a 256MB cap
    r = run_sandboxed(mem_bomb_code, df.copy(), timeout_seconds=10, memory_mb=256)
    status = "BLOCKED (correct)" if not r.success else "!!! NOT BLOCKED !!!"
    print(f"\n[{status}] large list allocation, 256MB cap")
    print(f"    error: {r.error}")

    print("\n" + "=" * 70)
    print("PART 3 — normal queries still work, and quickly")
    print("=" * 70)

    fast_queries = [
        ("groupby aggregation", "result = df.groupby('category')['sales'].mean()"),
        ("sort + head", "result = df.sort_values('sales', ascending=False).head(2)"),
    ]
    for label, code in fast_queries:
        t0 = time.time()
        r = run_sandboxed(code, df.copy())
        elapsed = time.time() - t0
        status = "OK" if r.success and r.produced else "!!! BROKEN !!!"
        print(f"\n[{status}] {label}  ({elapsed:.2f}s)")
        if r.success:
            print(f"    -> {type(r.value).__name__}: {str(r.value)[:80]}")

    print("\n" + "=" * 70)
    print("PART 4 — matplotlib figure across the process boundary")
    print("=" * 70)

    chart_code = "plt.figure()\nplt.bar(df['name'], df['sales'])\nresult = plt.gcf()"
    r = run_sandboxed(chart_code, df.copy())
    status = "OK — figure pickled fine" if r.success and r.produced else "!!! FAILED — see error !!!"
    print(f"\n[{status}]")
    print(f"    success={r.success} produced={r.produced} value_type={type(r.value).__name__ if r.value is not None else None}")
    if not r.success:
        print(f"    error: {r.error}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
