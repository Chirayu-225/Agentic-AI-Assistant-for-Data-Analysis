"""
Every answer from the analysis pipeline names the dataset version it used.

Run from the "AST 2.0" folder:
    python -m tests.storage.test_dataset_version_pipeline

Uses the SQLite backend and a stub model (no network).
"""
from __future__ import annotations

import os
import tempfile

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


CSV = b"region,amount\nnorth,10\nsouth,20\nnorth,30\n"
PLAN = '{"select":[{"agg":"sum","column":"amount","as":"total"}],"group_by":["region"]}'


def main():
    os.environ.pop("AST_QUERY_PLAN", None)
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("AST_DATASET_DB", None)
    from analysis import analysis_service as svc
    from analysis import query_engine as qe
    from storage import dataset_store as ds
    from storage import session_cache
    ds.reset_for_tests()
    orig = qe.llm_with_fallback
    qe.llm_with_fallback = lambda prompt, temperature=0.1, session_id="", **k: PLAN
    q = "Give me the sum of amount for each region, largest first"
    try:
        out = svc.run_analysis(CSV, "sales.csv", q, "sess-A")
        check("with no store configured the response has no version field (unchanged behaviour)",
              out.get("success") is True and "datasetVersion" not in out, str(sorted(out)))

        os.environ["AST_DATASET_DB"] = os.path.join(tempfile.mkdtemp(prefix="ast_dsp_"), "d.sqlite")
        ds.reset_for_tests()
        store = ds.get_store()
        out = svc.run_analysis(CSV, "sales.csv", q, "sess-A", file_url="local://" + "b" * 32)
        dv = out.get("datasetVersion") or {}
        check("the answer carries a dataset version", out.get("success") is True and dv.get("versionNo") == 2
              and dv.get("kind") == "clean" and len(dv.get("sha256", "")) == 64, str(dv))
        chain = store.lineage("sess-A", dv["versionId"])
        check("that version descends from the uploaded bytes", [c.kind for c in chain] == ["upload", "clean"])
        _, raw = store.get_version("sess-A", chain[0].version_id)
        check("the upload version holds the exact uploaded bytes", raw == CSV)
        answers = store.answers_for("sess-A", dv["versionId"])
        check("the answer is recorded against that version with its route and check status",
              len(answers) == 1 and answers[0]["method"] == "query_plan" and answers[0]["verification"] == "verified", str(answers))

        cached = session_cache.get("sess-A", "local://" + "b" * 32)
        out2 = svc.run_analysis_from_cleaned_df(cached.cleaned_df, "How many rows are there?", "sess-A",
                                                raw_bytes=cached.csv_bytes, filename=cached.filename)
        check("a follow-up question on the cached table names the SAME version",
              (out2.get("datasetVersion") or {}).get("versionId") == dv["versionId"], str(out2.get("datasetVersion")))
        check("... and is recorded as a second answer", len(store.answers_for("sess-A", dv["versionId"])) == 2)

        changed = cached.cleaned_df[cached.cleaned_df["amount"] > 10].reset_index(drop=True)
        out3 = svc.run_analysis_from_cleaned_df(changed, "How many rows are there?", "sess-A",
                                                raw_bytes=cached.csv_bytes, filename=cached.filename)
        dv3 = out3.get("datasetVersion") or {}
        check("a question on an ALTERED table names a different, new version",
              dv3.get("versionId") not in (None, dv["versionId"]) and dv3.get("versionNo") == 3, str(dv3))
        _, original = store.get_version("sess-A", dv["versionId"])
        check("the earlier version still holds the earlier table", len(original) == 3)

        outb = svc.run_analysis(CSV, "sales.csv", q, "sess-B")
        check("another session uploading the same file gets its own version",
              (outb.get("datasetVersion") or {}).get("versionId") not in (None, dv["versionId"]))

        os.environ["AST_DATASET_DB"] = os.path.join(tempfile.gettempdir(), "no_such_dir_ast_xyz", "nested", "d.sqlite")
        ds.reset_for_tests()
        out4 = svc.run_analysis(CSV, "sales.csv", q, "sess-A")
        check("if the store is configured but unreachable the answer is still returned",
              out4.get("success") is True, str(out4.get("error")))
        check("... and says plainly that no version was recorded",
              out4.get("datasetVersion") is None and "datasetVersionError" in out4, str({k: out4.get(k) for k in ("datasetVersion", "datasetVersionError")}))
    finally:
        qe.llm_with_fallback = orig
        os.environ.pop("AST_DATASET_DB", None)
        ds.reset_for_tests()

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
