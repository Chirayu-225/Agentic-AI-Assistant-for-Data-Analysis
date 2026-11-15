"""
local_store.py — on-disk store for uploaded files, so the app runs with NO
third-party storage (no Vercel Blob, no token).

Flow: the browser sends the file to the Next.js app, which streams it to the
backend's POST /upload. The bytes are written to a temp folder and the backend
returns an id of the form  local://<32 hex chars>.  Every other endpoint keeps
accepting a `fileUrl`; api._fetch_blob() resolves a local:// id through load().

Design choices
  * Content-addressed: the id is the first 128 bits of the file's SHA-256, so
    uploading the same bytes twice reuses one file, and an id can never contain
    a path separator (it is validated against ^[0-9a-f]{32}$ before any
    filesystem access — no path traversal).
  * Written to disk, not memory: survives an uvicorn restart mid-session and
    does not count against the in-memory cache budgets in cache_layer.py.
  * Streamed with a hard size cap (MAX_UPLOAD_MB, default 200) so a huge or
    endless upload cannot fill the disk; a partial file is deleted on failure.
  * Old files are removed after LOCAL_UPLOAD_TTL_HOURS (default 24).

Env vars (all optional):
  LOCAL_UPLOAD_DIR          default: <system temp dir>/ast_uploads
  MAX_UPLOAD_MB             default: 200
  LOCAL_UPLOAD_TTL_HOURS    default: 24
"""

import hashlib
import os
import re
import tempfile
import time

SCHEME = "local://"
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def upload_dir() -> str:
    d = os.environ.get("LOCAL_UPLOAD_DIR") or os.path.join(tempfile.gettempdir(), "ast_uploads")
    os.makedirs(d, exist_ok=True)
    return d


def max_upload_bytes() -> int:
    return int(float(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024)


def ttl_seconds() -> float:
    return float(os.environ.get("LOCAL_UPLOAD_TTL_HOURS", "24")) * 3600


class UploadTooLarge(Exception):
    pass


def is_local_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith(SCHEME)


def _path_for(file_id: str) -> str:
    return os.path.join(upload_dir(), file_id + ".bin")


def cleanup_old(now: float | None = None) -> int:
    """Delete stored files older than the TTL. Returns how many were removed."""
    now = time.time() if now is None else now
    removed = 0
    try:
        for name in os.listdir(upload_dir()):
            p = os.path.join(upload_dir(), name)
            try:
                if now - os.path.getmtime(p) > ttl_seconds():
                    os.remove(p)
                    removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed


class UploadWriter:
    """Streams chunks to a temp file while hashing and enforcing the size cap."""

    def __init__(self, limit: int | None = None):
        self.limit = max_upload_bytes() if limit is None else limit
        self.size = 0
        self._hash = hashlib.sha256()
        fd, self._tmp = tempfile.mkstemp(prefix="up_", suffix=".part", dir=upload_dir())
        self._fh = os.fdopen(fd, "wb")
        self._done = False

    def write(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > self.limit:
            self.abort()
            raise UploadTooLarge(f"File is larger than the {self.limit // (1024 * 1024)} MB limit.")
        self._hash.update(chunk)
        self._fh.write(chunk)

    def finish(self) -> str:
        """Finalize and return the local:// id. Empty uploads are rejected."""
        if self.size == 0:
            self.abort()
            raise ValueError("The uploaded file is empty.")
        self._fh.close()
        self._done = True
        file_id = self._hash.hexdigest()[:32]
        os.replace(self._tmp, _path_for(file_id))   # atomic; same bytes -> same file
        os.utime(_path_for(file_id), None)          # re-upload refreshes the TTL
        cleanup_old()
        return SCHEME + file_id

    def abort(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._fh.close()
        except Exception:
            pass
        try:
            os.remove(self._tmp)
        except OSError:
            pass


def save_bytes(data: bytes) -> str:
    """Convenience for small in-memory payloads (and tests)."""
    w = UploadWriter()
    w.write(data)
    return w.finish()


def load(url: str) -> bytes | None:
    """Bytes for a local:// id, or None if it is malformed, unknown or expired."""
    if not is_local_url(url):
        return None
    file_id = url[len(SCHEME):]
    if not _ID_RE.match(file_id):
        return None
    p = _path_for(file_id)
    try:
        if time.time() - os.path.getmtime(p) > ttl_seconds():
            return None
        with open(p, "rb") as fh:
            return fh.read()
    except OSError:
        return None
