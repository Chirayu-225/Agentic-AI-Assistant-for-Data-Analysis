"""
eval_insights.py — measures Auto-Analyze's verified-insight pipeline on
the held-out datasets.

Reports, per dataset and pooled:
  - verified insights returned (0-5)
  - candidates dropped as unverifiable (the hallucination rate the
    verifier is catching — if this is high, the writer prompt is drifting)
  - an INDEPENDENT re-check: each returned insight's code is re-run in
    the sandbox and its finding re-verified against the fresh result, so
    a bug in the pipeline's own bookkeeping can't make it grade itself
  - latency

Usage: python eval_insights.py   (needs GROQ_API_KEY)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
HELDOUT = os.path.join(HERE, "eval_data", "heldout")
DATASETS = ["invoices_messy.csv", "store_units_wide.xlsx", "expenses_dirty.csv"]


def recheck(insight: dict, df) -> bool:
    from insight_verifier import verify_finding
    from sandbox_limits import run_sandboxed
    sbr = run_sandboxed(insight.get("code", ""), df, include_charts=False, timeout_seconds=20)
    return bool(sbr.success and sbr.produced and verify_finding(insight["finding"], sbr.value)["verified"])


def main() -> int:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 2
    from auto_analyze_service import run_auto_analyze
    from cleaning import rule_based_clean
    from file_reader import read_tabular_file

    rows = []
    for name in DATASETS:
        with open(os.path.join(HELDOUT, name), "rb") as f:
            data = f.read()
        t0 = time.time()
        out = run_auto_analyze(data, name, session_id=f"eval-insights-{int(t0)}")
        secs = round(time.time() - t0, 1)
        insights = out.get("insights") or []
        df, _ = rule_based_clean(read_tabular_file(data, name))
        rechecked = sum(recheck(i, df) for i in insights)
        rows.append({"dataset": name, "success": out.get("success"), "verified": len(insights),
                     "dropped": out.get("droppedUnverified", 0), "recheckPassed": rechecked,
                     "seconds": secs, "error": out.get("error")})
        print(f"{name}: {len(insights)} verified, {out.get('droppedUnverified', 0)} dropped, "
              f"{rechecked}/{len(insights)} pass independent re-check, {secs}s", flush=True)
        for i in insights:
            print(f"   • {i['finding']}")

    shown = sum(r["verified"] for r in rows)
    dropped = sum(r["dropped"] for r in rows)
    summary = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "datasets": rows,
        "insightsShown": shown,
        "candidatesDropped": dropped,
        "dropRate": round(dropped / (shown + dropped), 3) if shown + dropped else None,
        "independentRecheckRate": round(sum(r["recheckPassed"] for r in rows) / shown, 3) if shown else None,
    }
    out_dir = os.path.join(HERE, "eval_results", "insights")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nShown {shown}, dropped {dropped} (drop rate {summary['dropRate']}), "
          f"independent re-check {summary['independentRecheckRate']}\n{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
