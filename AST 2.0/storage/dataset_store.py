"""
dataset_store.py - versioned, immutable datasets, and which version each
answer came from.

THE HAZARD (H4)
    Data changes without a record: a cleaning step rewrites a table, a later
    answer is computed from the changed table, and nobody can say which data
    an answer was based on, or get the earlier data back.

THE INVARIANTS
    1. A stored version is never changed or deleted. Every change is a NEW
       version that points at its parent.
    2. Every recorded answer names exactly one version.
    3. A version is visible only to the session that created it. A wrong
       session and a missing version give the same "not found".

THE BOUNDARY
    Invariant 1 is enforced by the DATABASE (triggers that refuse UPDATE and
    DELETE), not by this module being careful: code elsewhere in the app, or a
    bug here, still cannot rewrite history. A database administrator can drop
    a trigger; that is outside what an application can defend.
    Content is addressed by SHA-256, so verify_integrity() can prove a stored
    version is byte-for-byte what was recorded.

BACKENDS
    PostgreSQL   DATABASE_URL=postgresql://user:password@localhost:5432/ast
                 (needs the `psycopg` package)
    SQLite       AST_DATASET_DB=<path to a file>   (no server; used by tests)
    Neither set  the store is OFF and the app behaves exactly as before.

    DATASET_STORE_MAX_MB (default 50): a version larger than this keeps its
    hash, shape and lineage but not its content.

LIMITS, STATED PLAINLY
    * Immutable means kept: there is no automatic deletion. Uploaded files
      elsewhere expire after 24 hours; versions recorded here do not.
    * "Session" is the id the frontend sends. This store checks ownership by
      that id; it does not authenticate people.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

NOT_FOUND = "dataset version not found"
_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,50}$")
_TABLES = ("datasets", "dataset_versions", "answers")


class NotFound(LookupError):
    """No such version for this session (missing and not-yours look the same)."""


class ImmutableError(RuntimeError):
    """The database refused to change or delete a recorded row."""


@dataclass(frozen=True)
class VersionRef:
    dataset_id: str
    version_id: str
    version_no: int
    parent_version_id: Optional[str]
    kind: str
    sha256: str
    row_count: Optional[int]
    col_count: Optional[int]
    note: str
    created_at: str

    def to_dict(self) -> dict:
        return {"datasetId": self.dataset_id, "versionId": self.version_id, "versionNo": self.version_no,
                "parentVersionId": self.parent_version_id, "kind": self.kind, "sha256": self.sha256,
                "rows": self.row_count, "columns": self.col_count}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def frame_sha256(df: pd.DataFrame) -> str:
    """A content hash of a table: column names, types, shape and every value."""
    h = hashlib.sha256()
    h.update(repr([str(c) for c in df.columns]).encode("utf-8"))
    h.update(repr([str(t) for t in df.dtypes]).encode("utf-8"))
    h.update(repr(df.shape).encode("utf-8"))
    try:
        for i in range(df.shape[1]):
            h.update(pd.util.hash_pandas_object(df.iloc[:, i], index=False).to_numpy().tobytes())
    except TypeError:   # unhashable cells (lists, dicts): fall back to their text form
        h.update(df.astype(str).to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


def _encode_frame(df: pd.DataFrame) -> tuple[str, bytes]:
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return "parquet", buf.getvalue()
    except Exception:   # a column Parquet cannot hold: keep a readable copy instead (types are not preserved)
        return "csv", df.to_csv(index=False).encode("utf-8")


def _decode(fmt: str, payload: Optional[bytes]) -> Any:
    if payload is None:
        return None
    if fmt == "raw":
        return payload
    if fmt == "parquet":
        return pd.read_parquet(io.BytesIO(payload))
    return pd.read_csv(io.BytesIO(payload))


class DatasetStore:
    def __init__(self, url: str, schema: Optional[str] = None, max_payload_mb: Optional[float] = None):
        self.url = url
        if url.startswith("sqlite:///"):
            self.backend, self._path = "sqlite", url[len("sqlite:///"):]
        elif url.startswith(("postgresql://", "postgres://")):
            self.backend = "postgres"
        else:
            raise ValueError("dataset store URL must start with sqlite:/// or postgresql://")
        if schema is not None and not _SCHEMA_RE.match(schema):
            raise ValueError("invalid schema name")
        self.schema = schema
        self.max_payload_bytes = int(float(
            max_payload_mb if max_payload_mb is not None else os.environ.get("DATASET_STORE_MAX_MB", "50")) * 1024 * 1024)
        self._init_lock = threading.Lock()
        self._create()

    # ── connections ────────────────────────────────────────────────────────
    def _connect(self):
        if self.backend == "sqlite":
            conn = sqlite3.connect(self._path, timeout=30)
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        import psycopg
        conn = psycopg.connect(self.url, connect_timeout=10)
        if self.schema:
            conn.execute(f'SET search_path TO "{self.schema}"')
        return conn

    def _q(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.backend == "postgres" else sql

    def _run(self, fn):
        """Run fn(cursor-like execute) in one transaction; translate trigger refusals."""
        conn = self._connect()
        try:
            # No parameters -> no placeholder parsing (DDL contains a literal % for plpgsql).
            out = fn(lambda sql, params=(): conn.execute(self._q(sql), params) if params else conn.execute(sql))
            conn.commit()
            return out
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            if "immutable" in str(e).lower():
                raise ImmutableError("recorded rows are immutable: changes and deletions are refused") from None
            raise
        finally:
            conn.close()

    def _raw_execute(self, sql: str, params: tuple = ()) -> None:
        """Run one statement as-is. Used by tests to attack the store directly."""
        self._run(lambda ex: ex(sql, params) and None)

    # ── schema ─────────────────────────────────────────────────────────────
    def _create(self) -> None:
        blob = "BYTEA" if self.backend == "postgres" else "BLOB"
        ddl = [
            """CREATE TABLE IF NOT EXISTS datasets (
                   dataset_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL)""",
            f"""CREATE TABLE IF NOT EXISTS dataset_versions (
                   version_id TEXT PRIMARY KEY,
                   dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
                   version_no INTEGER NOT NULL,
                   parent_version_id TEXT REFERENCES dataset_versions(version_id),
                   kind TEXT NOT NULL, content_sha256 TEXT NOT NULL,
                   row_count INTEGER, col_count INTEGER,
                   payload_format TEXT NOT NULL, payload {blob},
                   note TEXT NOT NULL, created_at TEXT NOT NULL,
                   UNIQUE (dataset_id, version_no))""",
            """CREATE TABLE IF NOT EXISTS answers (
                   answer_id TEXT PRIMARY KEY,
                   version_id TEXT NOT NULL REFERENCES dataset_versions(version_id),
                   session_id TEXT NOT NULL, question TEXT NOT NULL, question_sha256 TEXT NOT NULL,
                   method TEXT NOT NULL, verification TEXT NOT NULL, created_at TEXT NOT NULL)""",
            "CREATE INDEX IF NOT EXISTS idx_versions_lookup ON dataset_versions (dataset_id, kind, content_sha256)",
            "CREATE INDEX IF NOT EXISTS idx_datasets_session ON datasets (session_id)",
            "CREATE INDEX IF NOT EXISTS idx_answers_version ON answers (version_id)",
        ]
        with self._init_lock:
            if self.backend == "postgres" and self.schema:
                import psycopg
                with psycopg.connect(self.url, connect_timeout=10) as c:
                    c.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')

            def build(ex):
                for stmt in ddl:
                    ex(stmt)
                if self.backend == "sqlite":
                    for t in _TABLES:
                        for op in ("UPDATE", "DELETE"):
                            ex(f"""CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t}
                                   BEGIN SELECT RAISE(ABORT, '{t} rows are immutable'); END""")
                else:
                    ex("""CREATE OR REPLACE FUNCTION ast_refuse_change() RETURNS trigger AS $$
                          BEGIN RAISE EXCEPTION '% rows are immutable', TG_TABLE_NAME; END;
                          $$ LANGUAGE plpgsql""")
                    for t in _TABLES:
                        ex(f"""DO $$ BEGIN
                                 IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = '{t}_immutable_row'
                                                AND tgrelid = '{t}'::regclass) THEN
                                   CREATE TRIGGER {t}_immutable_row BEFORE UPDATE OR DELETE ON {t}
                                     FOR EACH ROW EXECUTE FUNCTION ast_refuse_change();
                                   CREATE TRIGGER {t}_immutable_all BEFORE TRUNCATE ON {t}
                                     FOR EACH STATEMENT EXECUTE FUNCTION ast_refuse_change();
                                 END IF;
                               END $$""")
            self._run(build)

    def drop_schema_for_tests(self) -> None:
        if self.backend == "postgres" and self.schema and self.schema.startswith("ast_test_"):
            import psycopg
            with psycopg.connect(self.url, connect_timeout=10) as c:
                c.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    # ── helpers ────────────────────────────────────────────────────────────
    _COLS = ("v.dataset_id, v.version_id, v.version_no, v.parent_version_id, v.kind, v.content_sha256, "
             "v.row_count, v.col_count, v.note, v.created_at")

    @staticmethod
    def _ref(row) -> VersionRef:
        return VersionRef(*row[:10])

    def _owned(self, ex, session_id: str, version_id: str, extra: str = ""):
        """The version row if it exists AND belongs to this session; else NotFound."""
        if not isinstance(session_id, str) or not session_id or not isinstance(version_id, str):
            raise NotFound(NOT_FOUND)
        row = ex(f"SELECT {self._COLS}{extra} FROM dataset_versions v JOIN datasets d ON d.dataset_id = v.dataset_id "
                 "WHERE v.version_id = ? AND d.session_id = ?", (version_id, session_id)).fetchone()
        if row is None:
            raise NotFound(NOT_FOUND)
        return row

    def _payload(self, data: bytes) -> Optional[bytes]:
        return data if len(data) <= self.max_payload_bytes else None

    # ── writing ────────────────────────────────────────────────────────────
    def register_upload(self, session_id: str, name: str, raw: bytes) -> VersionRef:
        """Version 1 of a dataset: the bytes exactly as uploaded. The same bytes
        from the same session return the existing version."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("a session id is required")
        sha = hashlib.sha256(raw).hexdigest()

        def work(ex):
            row = ex(f"SELECT {self._COLS} FROM dataset_versions v JOIN datasets d ON d.dataset_id = v.dataset_id "
                     "WHERE d.session_id = ? AND v.kind = 'upload' AND v.content_sha256 = ?", (session_id, sha)).fetchone()
            if row:
                return self._ref(row)
            did, vid, now = uuid.uuid4().hex, uuid.uuid4().hex, _now()
            ex("INSERT INTO datasets (dataset_id, session_id, name, created_at) VALUES (?, ?, ?, ?)",
               (did, session_id, str(name or "")[:200], now))
            ex("INSERT INTO dataset_versions (version_id, dataset_id, version_no, parent_version_id, kind, content_sha256, "
               "row_count, col_count, payload_format, payload, note, created_at) VALUES (?, ?, 1, NULL, 'upload', ?, NULL, NULL, 'raw', ?, ?, ?)",
               (vid, did, sha, self._payload(raw), "uploaded file", now))
            return VersionRef(did, vid, 1, None, "upload", sha, None, None, "uploaded file", now)
        return self._run(work)

    def add_version(self, session_id: str, parent_version_id: str, df: pd.DataFrame, kind: str, note: str = "") -> VersionRef:
        """A new version derived from `parent_version_id`. The same table from
        the same parent and step returns the existing version."""
        sha = frame_sha256(df)
        fmt, data = _encode_frame(df)
        last_error: Optional[Exception] = None
        for _ in range(4):   # two writers can pick the same next number; the UNIQUE constraint makes one retry
            def work(ex):
                parent = self._ref(self._owned(ex, session_id, parent_version_id))
                row = ex(f"SELECT {self._COLS} FROM dataset_versions v WHERE v.dataset_id = ? AND v.parent_version_id = ? "
                         "AND v.kind = ? AND v.content_sha256 = ?", (parent.dataset_id, parent.version_id, kind, sha)).fetchone()
                if row:
                    return self._ref(row)
                nxt = ex("SELECT COALESCE(MAX(version_no), 0) + 1 FROM dataset_versions WHERE dataset_id = ?",
                         (parent.dataset_id,)).fetchone()[0]
                vid, now = uuid.uuid4().hex, _now()
                ex("INSERT INTO dataset_versions (version_id, dataset_id, version_no, parent_version_id, kind, content_sha256, "
                   "row_count, col_count, payload_format, payload, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (vid, parent.dataset_id, int(nxt), parent.version_id, str(kind)[:40], sha, int(df.shape[0]), int(df.shape[1]),
                    fmt, self._payload(data), str(note or "")[:500], now))
                return VersionRef(parent.dataset_id, vid, int(nxt), parent.version_id, str(kind)[:40], sha,
                                  int(df.shape[0]), int(df.shape[1]), str(note or "")[:500], now)
            try:
                return self._run(work)
            except (NotFound, ImmutableError):
                raise
            except Exception as e:
                if "unique" not in str(e).lower():
                    raise
                last_error = e
        raise last_error  # type: ignore[misc]

    def record_answer(self, session_id: str, version_id: str, question: str, method: str, verification: str) -> str:
        def work(ex):
            self._owned(ex, session_id, version_id)
            aid, q = uuid.uuid4().hex, str(question or "")
            ex("INSERT INTO answers (answer_id, version_id, session_id, question, question_sha256, method, verification, created_at) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
               (aid, version_id, session_id, q[:500], hashlib.sha256(q.encode("utf-8")).hexdigest(),
                str(method or "")[:40], str(verification or "")[:40], _now()))
            return aid
        return self._run(work)

    # ── reading ────────────────────────────────────────────────────────────
    def get_version(self, session_id: str, version_id: str) -> tuple[VersionRef, Any]:
        """(metadata, content). Content is the original bytes for an upload, a
        DataFrame for a derived version, or None if it was over the size cap."""
        def work(ex):
            return self._owned(ex, session_id, version_id, ", v.payload_format, v.payload")
        row = self._run(work)
        payload = bytes(row[11]) if row[11] is not None else None
        return self._ref(row), _decode(row[10], payload)

    def lineage(self, session_id: str, version_id: str) -> list[VersionRef]:
        """This version and its ancestors, oldest first."""
        def work(ex):
            chain = [self._ref(self._owned(ex, session_id, version_id))]
            while chain[-1].parent_version_id and len(chain) < 10_000:
                chain.append(self._ref(self._owned(ex, session_id, chain[-1].parent_version_id)))
            return chain[::-1]
        return self._run(work)

    def answers_for(self, session_id: str, version_id: str) -> list[dict]:
        def work(ex):
            self._owned(ex, session_id, version_id)
            rows = ex("SELECT answer_id, version_id, question, method, verification, created_at FROM answers "
                      "WHERE version_id = ? AND session_id = ? ORDER BY created_at, answer_id", (version_id, session_id)).fetchall()
            return [dict(zip(("answer_id", "version_id", "question", "method", "verification", "created_at"), r)) for r in rows]
        return self._run(work)

    def verify_integrity(self, session_id: str, version_id: str) -> bool:
        """True if the stored content still hashes to the recorded SHA-256
        (also True when the content was over the size cap and never stored)."""
        ref, content = self.get_version(session_id, version_id)
        if content is None:
            return True
        actual = hashlib.sha256(content).hexdigest() if isinstance(content, bytes) else frame_sha256(content)
        return actual == ref.sha256


# ── the app's single store, chosen by environment ──────────────────────────
_store: Optional[DatasetStore] = None
_store_error: str = ""
_store_lock = threading.Lock()
_resolved = False


def configured() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").startswith(("postgresql://", "postgres://"))
                or os.environ.get("AST_DATASET_DB"))


def get_store() -> Optional[DatasetStore]:
    """The configured store, or None if none is configured or it cannot be
    reached (store_error() then says why; the app keeps answering)."""
    global _store, _store_error, _resolved
    with _store_lock:
        if _resolved:
            return _store
        _resolved = True
        url = os.environ.get("DATABASE_URL", "")
        try:
            if url.startswith(("postgresql://", "postgres://")):
                _store = DatasetStore(url)
            elif os.environ.get("AST_DATASET_DB"):
                _store = DatasetStore("sqlite:///" + os.environ["AST_DATASET_DB"])
        except Exception as e:
            _store, _store_error = None, type(e).__name__
            print(f"[dataset_store] configured but unavailable ({type(e).__name__}); answers will not carry a "
                  f"dataset version until this is fixed", flush=True)
        return _store


def store_error() -> str:
    return _store_error


def reset_for_tests() -> None:
    global _store, _store_error, _resolved
    with _store_lock:
        _store, _store_error, _resolved = None, "", False
