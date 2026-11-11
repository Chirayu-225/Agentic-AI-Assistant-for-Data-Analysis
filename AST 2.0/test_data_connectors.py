"""
test_data_connectors.py - Google Drive / MySQL import for the Join SQL Analysis tab.

No network and no MySQL server needed: Google is replaced by a fake HTTP client and MySQL by a
fake connection. Run: python3 test_data_connectors.py
"""
import io, os, sys, tempfile

os.environ["LOCAL_UPLOAD_DIR"] = tempfile.mkdtemp(prefix="dc_test_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(tempfile.mkdtemp(prefix="dc_audit_"), "audit.jsonl")
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
os.environ.pop("RENDER", None); os.environ.pop("APP_ENV", None); os.environ.pop("ALLOW_PRIVATE_DB_HOSTS", None)

import decimal
import data_connectors as dc
import local_store

FAIL = []
def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: FAIL.append(label)

def raises(fn, *a, **k):
    try: fn(*a, **k)
    except dc.ConnectorError as e: return str(e)
    return None


# ---- link parsing -----------------------------------------------------------
ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
check("file/d link", dc.parse_drive_link(f"https://drive.google.com/file/d/{ID}/view?usp=sharing") == ("file", ID, None))
check("open?id link", dc.parse_drive_link(f"https://drive.google.com/open?id={ID}")[:2] == ("file", ID))
check("uc?id link", dc.parse_drive_link(f"https://drive.google.com/uc?export=download&id={ID}")[:2] == ("file", ID))
check("sheet link with gid", dc.parse_drive_link(f"https://docs.google.com/spreadsheets/d/{ID}/edit#gid=42") == ("sheet", ID, "42"))
check("sheet link without scheme", dc.parse_drive_link(f"docs.google.com/spreadsheets/d/{ID}/edit")[:2] == ("sheet", ID))
for bad, why in [("", "empty"), ("https://evil.com/file/d/" + ID, "other host"),
                 ("https://drive.google.com.evil.com/file/d/" + ID, "look-alike host"),
                 ("http://169.254.169.254/latest/meta-data", "metadata ip"),
                 (f"https://drive.google.com/drive/folders/{ID}", "folder"),
                 (f"https://docs.google.com/document/d/{ID}/edit", "google doc"),
                 ("https://drive.google.com/file/d/x/view", "short id"),
                 ("file:///etc/passwd", "file scheme")]:
    check(f"rejected: {why}", raises(dc.parse_drive_link, bad) is not None)


# ---- fake Google --------------------------------------------------------------
class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}
        self.closed = False
    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._b), chunk_size):
            yield self._b[i:i + chunk_size]
    def close(self): self.closed = True

class Http:
    def __init__(self, *resps): self.resps, self.urls = list(resps), []
    def get(self, url, **kw):
        assert kw.get("allow_redirects") is False, "redirects must be followed by hand"
        self.urls.append(url)
        return self.resps.pop(0)

CSV = b"id,name\n1,a\n2,b\n"
d, name = dc.fetch_drive(f"https://drive.google.com/file/d/{ID}/view",
                         http=Http(Resp(200, CSV, {"content-disposition": 'attachment; filename="orders 2024.csv"'})))
check("csv download + filename", d == CSV and name == "orders 2024.csv", name)

h = Http(Resp(200, CSV))
dc.fetch_drive(f"https://docs.google.com/spreadsheets/d/{ID}/edit#gid=7", http=h)
check("sheet exported as csv with gid, fixed host", h.urls == [f"https://docs.google.com/spreadsheets/d/{ID}/export?format=csv&gid=7"], h.urls)

_, name = dc.fetch_drive(f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(200, b"PK\x03\x04rest", {})))
check("xlsx detected from bytes whatever the name says", name.endswith(".xlsx"), name)

h = Http(Resp(303, headers={"location": "https://abc.googleusercontent.com/x"}), Resp(200, CSV))
dc.fetch_drive(f"https://drive.google.com/file/d/{ID}/view", http=h)
check("follows redirect to googleusercontent", h.urls[-1] == "https://abc.googleusercontent.com/x")

h = Http(Resp(302, headers={"location": "https://evil.example/x"}))
check("redirect off Google refused", raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=h) is not None)
h = Http(Resp(302, headers={"location": "http://169.254.169.254/"}))
check("redirect to plain http/metadata refused", raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=h) is not None)

msg = raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(200, b"<!DOCTYPE html><html>Sign in</html>")))
check("login page -> 'not shared' message", msg and "Anyone with the link" in msg, msg)
msg = raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(403)))
check("403 -> 'not shared' message", msg and "Anyone with the link" in msg, msg)
check("404 message", "could not find" in (raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(404))) or "").lower())
check("empty file refused", raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(200, b""))) is not None)
check("binary non-spreadsheet refused", raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(200, b"\x00\x01\x02binary"))) is not None)

os.environ["MAX_UPLOAD_MB"] = "0.0001"  # ~100 bytes
msg = raises(dc.fetch_drive, f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(200, b"a,b\n" * 200)))
check("size limit enforced while streaming", msg and "limit" in msg, msg)
os.environ["MAX_UPLOAD_MB"] = "200"

out = dc.import_drive(f"https://drive.google.com/file/d/{ID}/view", http=Http(Resp(200, CSV)))
check("import_drive stores a local:// file", out["success"] and local_store.load(out["url"]) == CSV and out["filename"].endswith(".csv"), out)


# ---- MySQL ---------------------------------------------------------------------
class Cur:
    def __init__(self, conn): self.c = conn; self.description = None; self._rows = []
    def execute(self, sql):
        self.c.sql.append(sql)
        if sql.startswith("SELECT"):
            self.description = [(n,) for n in self.c.cols]; self._rows = self.c.rows
    def fetchall(self): return self._rows

class Conn:
    def __init__(self, cols, rows, **kw): self.cols, self.rows, self.kw, self.sql, self.closed = cols, rows, kw, [], False
    def cursor(self): return Cur(self)
    def close(self): self.closed = True

def connector(cols, rows, holder):
    def connect(**kw):
        holder.append(Conn(cols, rows, **kw)); return holder[-1]
    return connect

P = dict(host="localhost", port=3306, user="reader", password="hunter2", database="shop", table="orders")
h = []
rows = [(1, decimal.Decimal("9.50"), b"x"), (2, None, None)]
out = dc.import_mysql(P, connect=connector(["id", "total", "blob"], rows, h))
csv = local_store.load(out["url"]).decode()
check("mysql import -> csv in local store", csv.splitlines()[0] == "id,total,blob" and out["rows"] == 2 and out["filename"] == "orders.csv", out)
check("decimal -> float, bytes -> text", "9.5" in csv and ",x" in csv, csv)
check("read-only session + fixed back-quoted select",
      h[0].sql[0] == "SET SESSION TRANSACTION READ ONLY" and h[0].sql[1].startswith("SELECT * FROM `shop`.`orders` LIMIT"), h[0].sql)
check("local_infile off, timeouts set", h[0].kw["local_infile"] is False and h[0].kw["connect_timeout"] <= 15)
check("connection closed", h[0].closed)
check("password never in the result", "hunter2" not in repr(out))

for field, val, why in [("table", "orders; DROP TABLE x", "sql in table"), ("table", "a`b", "backtick"), ("table", "", "empty table"),
                        ("table", "db.orders", "dotted table"), ("database", "x y", "space in db"),
                        ("host", "", "empty host"), ("host", "mysql://h", "url as host"), ("host", "a b", "space in host"),
                        ("port", "abc", "port text"), ("port", 70000, "port range"), ("user", "", "empty user")]:
    check(f"rejected: {why}", raises(dc.validate_mysql_params, {**P, field: val}) is not None)

h = []
out = dc.import_mysql({**P, "limit": 2}, connect=connector(["id"], [(1,), (2,), (3,)], h))
check("row cap -> truncated flag + note", out["rows"] == 2 and out["truncated"] and "first 2" in out["note"], out)
check("cap asks the server for limit+1 only", h[0].sql[1].endswith("LIMIT 3"), h[0].sql)

check("empty table refused", "no rows" in (raises(dc.import_mysql, P, connect=connector(["id"], [], [])) or ""))

class MySQLError(Exception): pass
def bad_login(**kw): raise MySQLError(1045, "Access denied for user 'reader'@'host' (using password: YES) hunter2")
msg = raises(dc.import_mysql, P, connect=bad_login)
check("login error is friendly and cannot echo the password", msg == "MySQL refused the username or password." and "hunter2" not in msg, msg)
def no_table(**kw): raise MySQLError(1146, "Table 'shop.orders' doesn't exist")
check("missing table message", "does not exist" in (raises(dc.import_mysql, P, connect=no_table) or ""))
def weird(**kw): raise MySQLError("secret hunter2 detail")
msg = raises(dc.import_mysql, P, connect=weird)
check("unknown error is generic", msg and "hunter2" not in msg, msg)

# hosted: private/loopback targets refused
os.environ["APP_ENV"] = "production"
for host in ("127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.169.254", "::1"):
    check(f"hosted refuses {host}", raises(dc.fetch_mysql, {**P, "host": host}, connect=connector(["a"], [(1,)], [])) is not None)
os.environ["ALLOW_PRIVATE_DB_HOSTS"] = "1"
check("hosted + ALLOW_PRIVATE_DB_HOSTS=1 allows", raises(dc.fetch_mysql, {**P, "host": "10.0.0.5"}, connect=connector(["a"], [(1,)], [])) is None)
os.environ.pop("APP_ENV"); os.environ.pop("ALLOW_PRIVATE_DB_HOSTS")
check("local run allows localhost", raises(dc.fetch_mysql, P, connect=connector(["a"], [(1,)], [])) is None)


# ---- API endpoints (secret, no password in audit log) ----------------------------
import api
api.SERVICE_SECRET = "s3cret-for-tests"  # a local .env would otherwise win over the variable above
from fastapi.testclient import TestClient
c = TestClient(api.app)
H = {"X-Service-Secret": "s3cret-for-tests"}
r = c.post("/connect/mysql", json={**P, "table": "bad;name", "sessionId": "t1"}, headers=H).json()
check("endpoint: bad input -> success false + message", r["success"] is False and "table name" in r["error"], r)
check("endpoint: needs the service secret", c.post("/connect/mysql", json=P).status_code in (401, 403))
r = c.post("/connect/google-drive", json={"link": "https://evil.com/x", "sessionId": "t1"}, headers=H).json()
check("endpoint: non-google link refused", r["success"] is False, r)
r = c.post("/connect/mysql", json={**P, "port": {"x": 1}, "sessionId": "t1"}, headers=H)
check("endpoint: malformed body does not echo the password", "hunter2" not in r.text, r.text[:200])
log = open(os.environ["AUDIT_LOG_PATH"], encoding="utf-8").read() if os.path.exists(os.environ["AUDIT_LOG_PATH"]) else ""
check("audit log has the events but no password/host", "connect_failed" in log and "hunter2" not in log and "localhost" not in log)

print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
sys.exit(1 if FAIL else 0)
