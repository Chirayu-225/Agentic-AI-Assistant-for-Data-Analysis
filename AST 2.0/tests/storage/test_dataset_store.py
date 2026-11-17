"""
Versioned, immutable datasets: nothing is ever edited in place, and every
answer names the exact dataset version it was computed from. (Hazard H4:
data altered without a record. Also H2: versions belong to one session.)

Run from the "AST 2.0" folder:
    python -m tests.storage.test_dataset_store

Runs against a throwaway SQLite file always, and ALSO against PostgreSQL when
AST_TEST_DATABASE_URL is set, for example:
    set AST_TEST_DATABASE_URL=postgresql://postgres:<password>@localhost:5432/ast_test
The Postgres run creates its tables inside a fresh schema and drops it at the end.
"""
from __future__ import annotations

import os
import tempfile
import uuid

import pandas as pd

FAIL: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAIL.append(label)


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception as e:  # the wrong error is a failure, and worth seeing
        print(f"       (raised {type(e).__name__}: {str(e)[:120]})")
        return False
    return False


def exercise(store, backend: str):
    from storage import dataset_store as ds
    tag = f"[{backend}] "
    raw = b"region,amount\nnorth,10\nsouth,20\nnorth,30\n"
    df1 = pd.DataFrame({"region": ["north", "south", "north"], "amount": [10, 20, 30]})
    df2 = df1[df1["amount"] > 10].reset_index(drop=True)

    print("-" * 70 + f"\n{tag}versions and lineage")
    v1 = store.register_upload("sess-A", "sales.csv", raw)
    check(tag + "an upload becomes version 1", v1.version_no == 1 and v1.kind == "upload" and len(v1.sha256) == 64)
    again = store.register_upload("sess-A", "sales.csv", raw)
    check(tag + "the same bytes from the same session are the same version, not a copy", again.version_id == v1.version_id)
    v2 = store.add_version("sess-A", v1.version_id, df1, kind="clean", note="rule-based clean")
    check(tag + "a cleaning step is a NEW version with a parent", v2.version_no == 2 and v2.parent_version_id == v1.version_id)
    v2b = store.add_version("sess-A", v1.version_id, df1, kind="clean", note="rule-based clean")
    check(tag + "the same step on the same data is the same version", v2b.version_id == v2.version_id)
    v3 = store.add_version("sess-A", v2.version_id, df2, kind="filter", note="amount > 10")
    check(tag + "a different table is a different version", v3.version_no == 3 and v3.sha256 != v2.sha256)
    chain = store.lineage("sess-A", v3.version_id)
    check(tag + "lineage walks back to the upload", [c.version_no for c in chain] == [1, 2, 3]
          and [c.kind for c in chain] == ["upload", "clean", "filter"], str([(c.version_no, c.kind) for c in chain]))

    print("-" * 70 + f"\n{tag}old versions stay readable and unchanged")
    meta, frame = store.get_version("sess-A", v2.version_id)
    check(tag + "version 2 still returns its own table after version 3 exists", frame.equals(df1), str(frame))
    meta, frame3 = store.get_version("sess-A", v3.version_id)
    check(tag + "version 3 returns the filtered table", frame3.equals(df2))
    meta, payload = store.get_version("sess-A", v1.version_id)
    check(tag + "the upload version returns the original bytes", payload == raw)
    check(tag + "stored content matches its recorded hash", all(store.verify_integrity("sess-A", v.version_id) for v in (v1, v2, v3)))

    print("-" * 70 + f"\n{tag}immutability is enforced by the database, not by good manners")
    for label, sql in (
        ("changing a version's content", "UPDATE dataset_versions SET row_count = 999 WHERE version_id = ?"),
        ("changing a version's hash", "UPDATE dataset_versions SET content_sha256 = 'forged' WHERE version_id = ?"),
        ("deleting a version", "DELETE FROM dataset_versions WHERE version_id = ?"),
    ):
        check(tag + f"{label} is refused", raises(lambda s=sql: store._raw_execute(s, (v2.version_id,)), ds.ImmutableError))
    ans = store.record_answer("sess-A", v3.version_id, "total amount?", method="query_plan", verification="verified")
    check(tag + "rewriting a recorded answer is refused",
          raises(lambda: store._raw_execute("UPDATE answers SET verification = 'verified' WHERE answer_id = ?", (ans,)), ds.ImmutableError))
    check(tag + "deleting a recorded answer is refused",
          raises(lambda: store._raw_execute("DELETE FROM answers WHERE answer_id = ?", (ans,)), ds.ImmutableError))
    meta, frame = store.get_version("sess-A", v2.version_id)
    check(tag + "after all those attempts version 2 is intact", frame.equals(df1) and meta.row_count == 3
          and store.verify_integrity("sess-A", v2.version_id))

    print("-" * 70 + f"\n{tag}answers name their version")
    store.record_answer("sess-A", v2.version_id, "how many rows?", method="shortcut", verification="")
    got = store.answers_for("sess-A", v3.version_id)
    check(tag + "an answer can be traced to the exact version", len(got) == 1 and got[0]["method"] == "query_plan"
          and got[0]["verification"] == "verified" and got[0]["version_id"] == v3.version_id, str(got))
    check(tag + "answers on another version are not mixed in", len(store.answers_for("sess-A", v2.version_id)) == 1)
    check(tag + "an answer cannot be recorded against a version that does not exist",
          raises(lambda: store.record_answer("sess-A", "no-such-version", "q", "m", ""), ds.NotFound))

    print("-" * 70 + f"\n{tag}one session cannot see another's versions")
    for label, fn in (
        ("read a version", lambda: store.get_version("sess-B", v2.version_id)),
        ("walk its lineage", lambda: store.lineage("sess-B", v3.version_id)),
        ("add a version on top of it", lambda: store.add_version("sess-B", v2.version_id, df2, kind="filter")),
        ("record an answer against it", lambda: store.record_answer("sess-B", v2.version_id, "q", "m", "")),
        ("list its answers", lambda: store.answers_for("sess-B", v3.version_id)),
        ("check its integrity", lambda: store.verify_integrity("sess-B", v2.version_id)),
    ):
        check(tag + f"another session cannot {label}", raises(fn, ds.NotFound))
    msg_other, msg_missing = "", ""
    try:
        store.get_version("sess-B", v2.version_id)
    except ds.NotFound as e:
        msg_other = str(e)
    try:
        store.get_version("sess-B", str(uuid.uuid4()))
    except ds.NotFound as e:
        msg_missing = str(e)
    check(tag + "the refusal does not reveal whether the version exists", msg_other == msg_missing and msg_other != "")
    b1 = store.register_upload("sess-B", "sales.csv", raw)
    check(tag + "the same file uploaded by another session is a separate dataset",
          b1.version_id != v1.version_id and b1.dataset_id != v1.dataset_id and b1.sha256 == v1.sha256)
    check(tag + "an empty session id is refused", raises(lambda: store.register_upload("", "x.csv", raw), ValueError))

    print("-" * 70 + f"\n{tag}awkward inputs")
    check(tag + "SQL in a file name is stored as text",
          store.register_upload("sess-A", "x'); DROP TABLE dataset_versions; --.csv", b"a\n1\n").version_no == 1
          and store.verify_integrity("sess-A", v1.version_id))
    odd = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", None]), "txt": ["a", None], "n": [1.5, float("nan")]})
    vo = store.add_version("sess-A", v1.version_id, odd, kind="clean", note="odd types")
    _, back = store.get_version("sess-A", vo.version_id)
    check(tag + "dates and missing values survive a round trip", back.equals(odd), str(back.dtypes.to_dict()))
    big = store.__class__(store.url, max_payload_mb=0.00001) if backend == "sqlite" else None
    if big is not None:
        vb = big.add_version("sess-A", v1.version_id, pd.DataFrame({"x": range(5000)}), kind="clean")
        m, payload = big.get_version("sess-A", vb.version_id)
        check(tag + "a table over the size cap keeps its hash and metadata but not its content",
              payload is None and m.row_count == 5000 and len(m.sha256) == 64)


def main():
    from storage import dataset_store as ds

    print("=" * 70 + "\nPART 1 - SQLite\n" + "=" * 70)
    path = os.path.join(tempfile.mkdtemp(prefix="ast_ds_"), "datasets.sqlite")
    exercise(ds.DatasetStore("sqlite:///" + path), "sqlite")
    reopened = ds.DatasetStore("sqlite:///" + path)
    check("[sqlite] versions survive closing and reopening the database",
          len(reopened.lineage("sess-A", reopened.register_upload("sess-A", "sales.csv",
              b"region,amount\nnorth,10\nsouth,20\nnorth,30\n").version_id)) == 1)

    print("\n" + "=" * 70 + "\nPART 2 - PostgreSQL\n" + "=" * 70)
    url = os.environ.get("AST_TEST_DATABASE_URL", "")
    if not url:
        print("[SKIPPED] set AST_TEST_DATABASE_URL to run the same checks against PostgreSQL")
    else:
        schema = "ast_test_" + uuid.uuid4().hex[:10]
        store = ds.DatasetStore(url, schema=schema)
        try:
            exercise(store, "postgres")
        finally:
            store.drop_schema_for_tests()

    print("\n" + "=" * 70 + "\nPART 3 - configuration\n" + "=" * 70)
    old = {k: os.environ.pop(k, None) for k in ("DATABASE_URL", "AST_DATASET_DB")}
    try:
        check("with nothing configured the store is off (the app behaves as before)", ds.get_store() is None)
        os.environ["AST_DATASET_DB"] = os.path.join(tempfile.mkdtemp(prefix="ast_ds_"), "x.sqlite")
        ds.reset_for_tests()
        check("AST_DATASET_DB turns on the SQLite store", ds.get_store() is not None and ds.get_store().backend == "sqlite")
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        ds.reset_for_tests()

    print("\n" + "=" * 70)
    if FAIL:
        print(f"DONE - {len(FAIL)} FAILURE(S): {FAIL}")
        raise SystemExit(1)
    print("DONE - all checks passed.")


if __name__ == "__main__":
    main()
