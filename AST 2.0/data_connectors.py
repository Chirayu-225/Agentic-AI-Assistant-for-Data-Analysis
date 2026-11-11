"""
data_connectors.py - bring a dataset into the Join SQL Analysis tab from Google Drive or MySQL.

Both connectors do the same thing in the end: fetch the data ONCE, write it to local_store as a
file, and return {"url": "local://<id>", "filename": ...}. From there it is an ordinary uploaded
dataset, so the rest of the pipeline (relationship check, DuckDB, sandbox, PII handling, audit)
is unchanged and nothing downstream knows or cares where the data came from.

Safety design
  Google Drive
    * only a Drive/Docs link is accepted; the file ID is parsed out and the request is built by
      us against FIXED Google hosts. The user's string is never fetched as a URL (no SSRF).
    * redirects are followed by hand and every hop must stay on a Google host.
    * no sign-in: the file must be shared as "Anyone with the link can view". A private file
      returns Google's login page, which we detect and explain instead of parsing.
    * size is capped by local_store.max_upload_bytes() while streaming.
  MySQL
    * credentials are used for this one request and are never stored, logged or audited
      (only a short hash of the host is). Error text is rewritten so it cannot echo them.
    * the table name is validated against a strict identifier pattern and back-quoted;
      the statement is a fixed SELECT * ... LIMIT built here - the user cannot send SQL.
    * the session is set READ ONLY, local_infile is off, and connect/read timeouts apply.
    * row count is capped (MYSQL_IMPORT_MAX_ROWS, default 200000); the result says if it was cut.
    * when the service is hosted (RENDER / APP_ENV=production) connections to private, loopback
      and link-local addresses are refused so the server cannot be used to probe its own
      network. Locally that is exactly what you want to reach, so it is allowed. Override with
      ALLOW_PRIVATE_DB_HOSTS=1 / =0.

Env vars (all optional): MYSQL_IMPORT_MAX_ROWS, ALLOW_PRIVATE_DB_HOSTS, CONNECTOR_TIMEOUT_S
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import local_store


class ConnectorError(Exception):
    """A problem the user can act on. The message is safe to show as is."""


# --------------------------------------------------------------------------- #
# Google Drive
# --------------------------------------------------------------------------- #
_DRIVE_HOSTS = {"drive.google.com", "docs.google.com", "drive.usercontent.google.com"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,120}$")
_MAX_REDIRECTS = 5


def _timeout() -> float:
    return float(os.environ.get("CONNECTOR_TIMEOUT_S", "30"))


def parse_drive_link(link: str) -> tuple[str, str, str | None]:
    """-> (kind, id, gid) where kind is "sheet" or "file". Raises ConnectorError otherwise."""
    link = (link or "").strip()
    if not link:
        raise ConnectorError("Paste the Google Drive link to the file.")
    u = urlparse(link if "://" in link else "https://" + link)
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or host not in _DRIVE_HOSTS:
        raise ConnectorError("That is not a Google Drive or Google Sheets link "
                             "(it should start with drive.google.com or docs.google.com).")
    path, qs = u.path, parse_qs(u.query)
    frag_gid = parse_qs(u.fragment).get("gid", [None])[0]
    gid = (qs.get("gid") or [frag_gid])[0]
    if gid is not None and not re.fullmatch(r"\d{1,20}", gid):
        gid = None

    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", path)
    if m and _ID_RE.match(m.group(1)):
        return "sheet", m.group(1), gid
    if "/folders/" in path or "/drive/folders" in path:
        raise ConnectorError("That link is to a folder. Open the file itself, choose Share, and paste its link.")
    if re.search(r"/(document|presentation|forms)/d/", path):
        raise ConnectorError("Only CSV, Excel and Google Sheets files can be used here.")
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", path)
    file_id = m.group(1) if m else (qs.get("id") or [None])[0]
    if file_id and _ID_RE.match(file_id):
        return "file", file_id, None
    raise ConnectorError("Could not find a file ID in that link. Use the link from Share -> Copy link.")


def _download_url(kind: str, file_id: str, gid: str | None) -> str:
    if kind == "sheet":
        return (f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
                + (f"&gid={gid}" if gid else ""))
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _is_google_host(host: str | None) -> bool:
    host = (host or "").lower()
    return host in _DRIVE_HOSTS or host.endswith(".googleusercontent.com")


def _sniff(data: bytes) -> str:
    """csv | xlsx | xls | html | unknown, from the bytes (the name Google sends can't be trusted)."""
    head = data[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if data[:4] == b"PK\x03\x04":
        return "xlsx"
    if data[:4] == b"\xd0\xcf\x11\xe0":
        return "xls"
    if head.startswith((b"<!doctype html", b"<html")) or b"<html" in head[:500]:
        return "html"
    if b"\x00" in data[:2048]:
        return "unknown"
    return "csv"


def _name_from_headers(headers: Any, fallback: str) -> str:
    cd = headers.get("content-disposition", "") if headers else ""
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd, re.I) or re.search(r'filename="?([^";]+)"?', cd, re.I)
    name = unquote(m.group(1)) if m else fallback
    name = re.sub(r"[^\w.\- ]+", "_", os.path.basename(name)).strip(" ._") or fallback
    return name[:80]


def fetch_drive(link: str, *, http: Any = None) -> tuple[bytes, str]:
    """Download a shared Drive file / Google Sheet. -> (bytes, filename with a correct extension)."""
    kind, file_id, gid = parse_drive_link(link)
    return _fetch_by_id(kind, file_id, gid, http=http)


def _fetch_by_id(kind: str, file_id: str, gid: str | None, *, http: Any = None, name_hint: str | None = None) -> tuple[bytes, str]:
    """The download itself. `name_hint` (a file name from a folder listing) is used for the file name when given."""
    import requests
    http = http or requests
    url = _download_url(kind, file_id, gid)
    limit = local_store.max_upload_bytes()

    resp = None
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            resp = http.get(url, stream=True, allow_redirects=False, timeout=_timeout(),
                            headers={"User-Agent": "Mozilla/5.0 (AST dataset import)"})
            if resp.status_code in (301, 302, 303, 307, 308):
                nxt = resp.headers.get("location", "")
                if not nxt.startswith("https://") or not _is_google_host(urlparse(nxt).hostname):
                    raise ConnectorError("Google redirected somewhere unexpected; the import was stopped.")
                url = nxt
                resp.close()
                continue
            break
        else:
            raise ConnectorError("Google redirected too many times; try again or use a different link.")

        if resp.status_code in (401, 403):
            raise ConnectorError(_NOT_SHARED)
        if resp.status_code == 404:
            raise ConnectorError("Google could not find that file. Check the link.")
        if resp.status_code == 429:
            raise ConnectorError("Google is limiting downloads of this file right now. Try again later.")
        if resp.status_code != 200:
            raise ConnectorError(f"Google Drive answered with an error ({resp.status_code}).")

        chunks, size = [], 0
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            size += len(chunk)
            if size > limit:
                raise ConnectorError(f"The file is larger than the {limit // (1024 * 1024)} MB limit.")
            chunks.append(chunk)
        data = b"".join(chunks)
        headers = resp.headers
    except ConnectorError:
        raise
    except requests.RequestException:
        raise ConnectorError("Could not reach Google Drive. Check your connection and try again.") from None
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    if not data:
        raise ConnectorError("The file is empty.")
    fmt = _sniff(data)
    if fmt == "html":
        raise ConnectorError(_NOT_SHARED)
    if fmt == "unknown":
        raise ConnectorError("That file is not a CSV or Excel file.")
    if name_hint:
        base = os.path.splitext(_name_from_headers({}, name_hint))[0]
    else:
        base = "google_sheet" if kind == "sheet" else os.path.splitext(_name_from_headers(headers, "drive_file"))[0]
    return data, f"{base or 'drive_file'}.{fmt}"


_NOT_SHARED = ("Google asked for a sign-in, so the file is private. In Drive choose Share -> General access -> "
               "'Anyone with the link' (Viewer), then paste the link again.")


# --------------------------------------------------------------------------- #
# MySQL
# --------------------------------------------------------------------------- #
_HOST_RE = re.compile(r"^[A-Za-z0-9._:-]{1,253}$")
_IDENT_RE = re.compile(r"^[A-Za-z0-9_$]{1,64}$")


def _hosted() -> bool:
    return bool(os.environ.get("RENDER")) or os.environ.get("APP_ENV", "").lower() == "production"


def _private_allowed() -> bool:
    v = os.environ.get("ALLOW_PRIVATE_DB_HOSTS")
    return (not _hosted()) if v is None else v == "1"


def _check_host(host: str, port: int) -> None:
    if _private_allowed():
        return
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ConnectorError("Could not find that MySQL host name.") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ConnectorError("This server only allows connections to databases on the public internet.")


def max_rows() -> int:
    return max(int(os.environ.get("MYSQL_IMPORT_MAX_ROWS", "200000")), 1)


def validate_mysql_params(p: dict) -> dict:
    host = str(p.get("host") or "").strip()
    database = str(p.get("database") or "").strip()
    table = str(p.get("table") or "").strip()
    user = str(p.get("user") or "")
    if not host or not _HOST_RE.match(host):
        raise ConnectorError("Enter the MySQL host (a name or IP address, without 'mysql://').")
    try:
        port = int(p.get("port") or 3306)
    except (TypeError, ValueError):
        raise ConnectorError("The port must be a number (MySQL's default is 3306).") from None
    if not 1 <= port <= 65535:
        raise ConnectorError("The port must be between 1 and 65535.")
    if not user:
        raise ConnectorError("Enter the MySQL username.")
    if not database or not _IDENT_RE.match(database):
        raise ConnectorError("Enter the database name (letters, digits, _ and $ only).")
    if not table or not _IDENT_RE.match(table):
        raise ConnectorError("Enter the table name (letters, digits, _ and $ only). "
                             "Write the table only, not a query.")
    try:
        limit = int(p.get("limit") or max_rows())
    except (TypeError, ValueError):
        raise ConnectorError("The row limit must be a whole number.") from None
    return {"host": host, "port": port, "user": user, "password": str(p.get("password") or ""),
            "database": database, "table": table, "ssl": bool(p.get("ssl")),
            "limit": min(max(limit, 1), max_rows())}


def _friendly_mysql_error(exc: Exception, host: str) -> ConnectorError:
    code = exc.args[0] if getattr(exc, "args", None) and isinstance(exc.args[0], int) else None
    msg = {
        1045: "MySQL refused the username or password.",
        1044: "That user is not allowed to use this database.",
        1049: "That database does not exist on the server.",
        1142: "That user is not allowed to read this table.",
        1146: "That table does not exist in the database.",
        2003: f"Could not connect to {host}. Check the host, the port, and that the server accepts remote connections.",
        2005: "Could not find that MySQL host name.",
        2006: "The MySQL server closed the connection.",
        2013: "The connection to MySQL was lost while reading.",
    }.get(code or 0)
    return ConnectorError(msg or "Could not read from MySQL. Check the connection details.")


def fetch_mysql(params: dict, *, connect: Any = None):
    """Read one table. -> (DataFrame, truncated: bool). `connect` is injectable for tests."""
    import pandas as pd
    cfg = validate_mysql_params(params)
    _check_host(cfg["host"], cfg["port"])
    if connect is None:
        try:
            import pymysql  # type: ignore
        except ImportError:
            raise ConnectorError("MySQL support is not installed on the server (pip install pymysql).") from None
        connect = pymysql.connect
        errors: tuple = (pymysql.MySQLError, OSError)
    else:
        errors = (Exception,)

    t = int(_timeout())
    conn = None
    try:
        kwargs: dict[str, Any] = dict(host=cfg["host"], port=cfg["port"], user=cfg["user"],
                                      password=cfg["password"], database=cfg["database"],
                                      connect_timeout=min(t, 15), read_timeout=t, write_timeout=t,
                                      local_infile=False, charset="utf8mb4")
        if cfg["ssl"]:
            kwargs["ssl"] = {}  # verify with the system CA bundle
        conn = connect(**kwargs)
        cur = conn.cursor()
        cur.execute("SET SESSION TRANSACTION READ ONLY")
        # Identifiers were validated against ^[A-Za-z0-9_$]+$ above, so back-quoting is safe.
        cur.execute(f"SELECT * FROM `{cfg['database']}`.`{cfg['table']}` LIMIT {cfg['limit'] + 1}")
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchall()
    except ConnectorError:
        raise
    except errors as exc:
        raise _friendly_mysql_error(exc, cfg["host"]) from None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    truncated = len(rows) > cfg["limit"]
    df = pd.DataFrame([tuple(r) for r in rows[:cfg["limit"]]], columns=cols)
    return _tidy(df), truncated


def _tidy(df):
    """Make MySQL's Python types CSV-friendly: Decimal -> float, bytes -> text."""
    import decimal
    for c in df.columns:
        if df[c].dtype != object:
            continue
        nn = df[c].dropna()
        if nn.empty:
            continue
        if all(isinstance(v, decimal.Decimal) for v in nn):
            df[c] = df[c].map(lambda v: None if v is None else float(v))
        elif any(isinstance(v, (bytes, bytearray)) for v in nn):
            df[c] = df[c].map(lambda v: v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else v)
    return df


# --------------------------------------------------------------------------- #
# Several files at once: a folder link, or several links
# --------------------------------------------------------------------------- #
# A folder is listed one of two ways. With GOOGLE_API_KEY set on the server, through the official Drive API (files.list;
# the folder must be shared "Anyone with the link"). Without a key, through the public embedded folder view
# (drive.google.com/embeddedfolderview), the same page tools such as gdown read: undocumented, so best-effort. Neither
# signs in to anyone's account, so private folders stay unreadable. Every request goes to a FIXED Google host; the folder
# id is validated and the user's text is never fetched as a URL.
_API_LIST_URL = "https://www.googleapis.com/drive/v3/files"
_EMBED_URL = "https://drive.google.com/embeddedfolderview"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
_TABLE_EXT = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
_TABLE_MIME = {"text/csv", "text/tab-separated-values", "text/plain", "application/vnd.ms-excel",
               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/octet-stream"}
_ENTRY_RE = re.compile(r'id="entry-([A-Za-z0-9_-]{10,120})"')
_TITLE_RE = re.compile(r'class="flip-entry-title"[^>]*>(.*?)</div>', re.S)


def max_files() -> int:
    try:
        return min(max(int(os.environ.get("DRIVE_FOLDER_MAX_FILES", "12")), 1), 50)
    except ValueError:
        return 12


def _budget_s() -> float:
    try:
        return max(float(os.environ.get("DRIVE_IMPORT_BUDGET_S", "90")), 0.0)
    except ValueError:
        return 90.0


def _api_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "").strip()


def split_links(text: str) -> list:
    """Links separated by whitespace, commas or semicolons; blanks dropped, repeats removed, order kept."""
    out, seen = [], set()
    for tok in re.split(r"[\s,;]+", (text or "").strip()):
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def folder_id_from_link(link: str) -> str | None:
    u = urlparse(link if "://" in link else "https://" + link)
    if u.scheme not in ("http", "https") or (u.hostname or "").lower() not in _DRIVE_HOSTS:
        return None
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", u.path)
    return m.group(1) if m and _ID_RE.match(m.group(1)) else None


def _clean_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", os.path.basename(str(name))).strip(" ._")[:120]


def _entry(kind: str, file_id: str, name: str, skip: str | None = None) -> dict:
    return {"kind": kind, "id": file_id, "gid": None, "name": _clean_name(name) or file_id[:8], "skip": skip}


def _api_error(resp: Any) -> ConnectorError:
    """A user-safe message for a failed files.list call. Never includes the API key or any URL."""
    reason = ""
    try:
        reason = str((resp.json().get("error", {}).get("errors") or [{}])[0].get("reason", ""))
    except Exception:  # noqa: BLE001
        pass
    if reason in ("keyInvalid", "badRequest") or resp.status_code == 400:
        return ConnectorError("Google rejected the server's GOOGLE_API_KEY. Check the key in the backend's .env.")
    if reason in ("accessNotConfigured", "API_KEY_SERVICE_BLOCKED", "ipRefererBlocked"):
        return ConnectorError("The Google Drive API is not enabled for the server's API key. Enable it in Google Cloud "
                              "(APIs & Services -> Google Drive API) and try again.")
    if resp.status_code == 404:
        return ConnectorError("Google could not find that folder. Check the link, and that it is shared with "
                              "'Anyone with the link' (Viewer).")
    if resp.status_code in (401, 403):
        return ConnectorError("Google would not list that folder: it is private. Share it with 'Anyone with the link' "
                              "(Viewer) and try again.")
    if resp.status_code == 429:
        return ConnectorError("Google is limiting requests right now. Try again later.")
    return ConnectorError(f"Google Drive answered with an error ({resp.status_code}).")


def _classify(name: str, mime: str, size) -> tuple:
    """-> (kind, skip reason or None) for one folder entry, from its mime type and file name."""
    low = name.lower()
    if mime == _FOLDER_MIME:
        return "file", "a subfolder (only the files directly in the folder are imported)"
    if mime == _SHEET_MIME:
        return "sheet", None
    if mime.startswith("application/vnd.google-apps."):
        return "file", "not a spreadsheet (only CSV, Excel and Google Sheets are imported)"
    if low.endswith(_TABLE_EXT) or mime in _TABLE_MIME - {"application/octet-stream", "text/plain"}:
        try:
            if size is not None and int(size) > local_store.max_upload_bytes():
                return "file", f"larger than the {local_store.max_upload_bytes() // (1024 * 1024)} MB limit"
        except (TypeError, ValueError):
            pass
        return "file", None
    return "file", "not a CSV or Excel file"


def _list_via_api(folder_id: str, key: str, http: Any) -> list:
    import requests
    http = http or requests
    entries, token = [], None
    for _ in range(5):                                          # at most 5 pages of 100
        params = {"q": f"'{folder_id}' in parents and trashed = false", "key": key, "pageSize": 100,
                  "fields": "nextPageToken,files(id,name,mimeType,size)", "supportsAllDrives": "true",
                  "includeItemsFromAllDrives": "true", "orderBy": "name"}
        if token:
            params["pageToken"] = token
        try:
            resp = http.get(_API_LIST_URL, params=params, allow_redirects=False, timeout=_timeout(),
                            headers={"User-Agent": "AST dataset import"})
        except requests.RequestException:                      # its message can contain the URL, and with it the key
            raise ConnectorError("Could not reach Google Drive. Check your connection and try again.") from None
        if resp.status_code != 200:
            raise _api_error(resp)
        try:
            body = resp.json()
        except ValueError:
            raise ConnectorError("Google Drive sent a reply that could not be read.") from None
        for f in body.get("files", []):
            fid, name = str(f.get("id", "")), str(f.get("name", ""))
            if not _ID_RE.match(fid):
                continue
            kind, skip = _classify(name, str(f.get("mimeType", "")), f.get("size"))
            entries.append(_entry(kind, fid, name, skip))
        token = body.get("nextPageToken")
        if not token:
            break
    return entries


def _list_via_embedded_view(folder_id: str, http: Any) -> list:
    import requests
    http = http or requests
    try:
        resp = http.get(_EMBED_URL, params={"id": folder_id}, allow_redirects=False, timeout=_timeout(),
                        headers={"User-Agent": "Mozilla/5.0 (AST dataset import)"})
    except requests.RequestException:
        raise ConnectorError("Could not reach Google Drive. Check your connection and try again.") from None
    if resp.status_code in (401, 403, 404) or resp.status_code in (301, 302, 303, 307, 308):
        raise ConnectorError("Google would not show that folder. Check the link, and that it is shared with "
                             "'Anyone with the link' (Viewer).")
    if resp.status_code != 200:
        raise ConnectorError(f"Google Drive answered with an error ({resp.status_code}).")
    import html as _html
    page = resp.content[:2_000_000].decode("utf-8", "replace") if hasattr(resp, "content") else ""
    ids = [(m.start(), m.group(1)) for m in _ENTRY_RE.finditer(page)]
    entries = []
    for i, (pos, fid) in enumerate(ids):
        end = ids[i + 1][0] if i + 1 < len(ids) else len(page)
        t = _TITLE_RE.search(page, pos, end)
        name = _html.unescape(re.sub(r"<[^>]+>", "", t.group(1))).strip() if t else fid[:8]
        low = name.lower()
        if low.endswith(_TABLE_EXT):
            entries.append(_entry("file", fid, name))
        elif "." not in os.path.basename(name):                # no extension: possibly a Google Sheet, tried as one
            entries.append(_entry("maybe_sheet", fid, name))
        else:
            entries.append(_entry("file", fid, name, "not a CSV or Excel file"))
    return entries


def list_folder(folder_id: str, *, http: Any = None) -> list:
    """The entries directly inside a shared folder: [{"kind","id","gid","name","skip"}]. Raises ConnectorError."""
    key = _api_key()
    entries = _list_via_api(folder_id, key, http) if key else _list_via_embedded_view(folder_id, http)
    if not entries:
        raise ConnectorError("That folder has no files that could be listed. Check the link, and that the folder is shared "
                             "with 'Anyone with the link' (Viewer).")
    return entries


def _import_entry(e: dict, deadline: float, http: Any) -> dict:
    """Download one entry and store it -> {"url","filename","bytes"}. Raises ConnectorError (caught by the caller)."""
    import time
    if time.monotonic() >= deadline:
        raise ConnectorError("Ran out of time before this file; import again to get the rest.")
    kinds = ("sheet", "file") if e["kind"] == "maybe_sheet" else (e["kind"],)
    last = None
    for kind in kinds:
        try:
            data, filename = _fetch_by_id(kind, e["id"], e["gid"], http=http, name_hint=e["name"])
            return store(data, filename)
        except ConnectorError as exc:
            last = exc
    raise last or ConnectorError("That file could not be imported.")


def import_drive_many(text: str, *, http: Any = None) -> dict:
    """One link to a file or Google Sheet behaves exactly like import_drive. A folder link, or several links, import every
    CSV / Excel / Google Sheet they point to (at most max_files(); one that fails is skipped and reported, the rest still
    import). -> {"success", "url", "filename", "bytes", "files": [...], "skipped": [{"name","reason"}], "note"}."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    links = split_links(text)
    if not links:
        raise ConnectorError("Paste the Google Drive link to the file or folder.")
    if len(links) == 1 and folder_id_from_link(links[0]) is None:
        return import_drive(links[0], http=http)
    if len(links) > 50:
        raise ConnectorError("Too many links at once (the limit is 50).")

    entries, skipped, seen = [], [], set()
    for n, link in enumerate(links, 1):
        fid = folder_id_from_link(link)
        try:
            found = list_folder(fid, http=http) if fid else None
            if found is None:
                kind, file_id, gid = parse_drive_link(link)
                found = [{"kind": kind, "id": file_id, "gid": gid, "name": None, "skip": None}]
        except ConnectorError as exc:
            if len(links) == 1:
                raise
            skipped.append({"name": f"link {n}", "reason": str(exc)})
            continue
        for e in found:
            if e["skip"]:
                skipped.append({"name": e["name"], "reason": e["skip"]})
            elif (e["id"], e["gid"]) not in seen:
                seen.add((e["id"], e["gid"]))
                entries.append(e)

    cap = max_files()
    for e in entries[cap:]:
        skipped.append({"name": e["name"] or e["id"][:8], "reason": f"over the {cap}-file limit; import it separately"})
    entries = entries[:cap]

    deadline = time.monotonic() + _budget_s()
    results = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=min(4, max(len(entries), 1))) as pool:
        futures = [pool.submit(_import_entry, e, deadline, http) for e in entries]
        for i, fut in enumerate(futures):
            try:
                results[i] = fut.result()
            except ConnectorError as exc:
                skipped.append({"name": entries[i]["name"] or entries[i]["id"][:8], "reason": str(exc)})
            except Exception:  # noqa: BLE001  one odd file must not sink the rest, and no raw error text reaches the user
                skipped.append({"name": entries[i]["name"] or entries[i]["id"][:8], "reason": "That file could not be imported."})

    files = [r for r in results if r]
    if not files:
        reason = skipped[0]["reason"] if skipped else "No CSV, Excel or Google Sheet files were found."
        raise ConnectorError(reason)
    note = f"Imported {len(files)} file{'s' if len(files) != 1 else ''}."
    if skipped:
        shown = "; ".join(f"{s['name']} ({s['reason']})" for s in skipped[:4])
        note += f" Skipped {len(skipped)}: {shown}{'; ...' if len(skipped) > 4 else ''}"
    return {"success": True, "url": files[0]["url"], "filename": files[0]["filename"],
            "bytes": sum(f["bytes"] for f in files), "count": len(files), "files": files, "skipped": skipped, "note": note}


# --------------------------------------------------------------------------- #
# Hand-off to local_store
# --------------------------------------------------------------------------- #
def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:10]


def store(data: bytes, filename: str) -> dict:
    try:
        url = local_store.save_bytes(data)
    except local_store.UploadTooLarge as exc:
        raise ConnectorError(str(exc)) from None
    except ValueError as exc:
        raise ConnectorError(str(exc)) from None
    return {"success": True, "url": url, "filename": filename, "bytes": len(data)}


def import_mysql(params: dict, *, connect: Any = None) -> dict:
    df, truncated = fetch_mysql(params, connect=connect)
    if df.empty and len(df.columns) == 0:
        raise ConnectorError("The table has no columns.")
    if df.empty:
        raise ConnectorError("The table has no rows.")
    table = validate_mysql_params(params)["table"]
    out = store(df.to_csv(index=False).encode("utf-8"), f"{table}.csv")
    out.update(rows=int(len(df)), columns=int(len(df.columns)), truncated=truncated)
    if truncated:
        out["note"] = f"Only the first {len(df):,} rows were imported (the import limit)."
    return out


def import_drive(link: str, *, http: Any = None) -> dict:
    data, filename = fetch_drive(link, http=http)
    return store(data, filename)
