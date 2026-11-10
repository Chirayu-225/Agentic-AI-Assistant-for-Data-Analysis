"""
test_data_checks_endpoint.py — POST /data-checks, driven through the real FastAPI app with TestClient (no server, no
network, no Groq). File loading is stubbed; the check itself is the real data_checks module.

Run: python test_data_checks_endpoint.py
"""
import os
import tempfile

TMP = tempfile.mkdtemp(prefix="dc_endpoint_test_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(TMP, "audit.jsonl")
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
for k in ("RATE_LIMIT_ENABLED", "RATE_LIMIT_MULTIPLIER", "CACHE_ENABLED"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient
import api

os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
api.SERVICE_SECRET = "s3cret-for-tests"

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def make_csv() -> bytes:
    rows = ["id,phone,note"]
    for i in range(120):
        phone = "555-0100" if i == 7 else f"(555) 867-{1000 + i}"
        rows.append(f"{i},{phone},ok")
    return ("\n".join(rows) + "\n").encode()


GOOD = make_csv()
BAD = b"\x00\x01\x02 not a table \xff\xfe"
SECRET_PATH = "C:\\Users\\Test\\secret_dir\\leaky.py"


def fake_load(url):
    if "dead" in url:
        return None
    return BAD if "bad" in url else GOOD


api.local_store.load = fake_load
api.requests.get = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network allowed"))

client = TestClient(api.app)
H = {"x-service-secret": "s3cret-for-tests"}


def post(url="local://abc", headers=H, filename="t.csv"):
    return client.post("/data-checks", json={"fileUrl": url, "filename": filename}, headers=headers)


# success
r = post()
j = r.json()
check("200 + success", r.status_code == 200 and j.get("success") is True, str(j)[:200])
check("response keys", set(j) == {"success", "filename", "nRows", "cards", "columns"}, str(sorted(j)))
check("filename echoed, nRows", j.get("filename") == "t.csv" and j.get("nRows") == 120)
check("drift card found for phone", any(c["column"] == "phone" and c["reason"] == "format_drift"
                                        for c in j["cards"]), str(j["cards"])[:300])
check("one column record per column", [c["column"] for c in j["columns"]] == ["id", "phone", "note"])

# cached second call is identical
check("second call identical", post().json() == j)

# auth
check("missing secret -> 401", post(headers={}).status_code == 401)
check("wrong secret -> 401", post(headers={"x-service-secret": "nope"}).status_code == 401)

# blob errors use the existing shape
d = post(url="local://dead").json()
check("expired upload -> existing error shape", d.get("success") is False and "no longer available" in d["error"])
d = post(url="https://example.com/x.csv").json()
check("remote url refused", d.get("success") is False)

# failure -> fixed safe message, no exception text, detail only in server log
import io, contextlib
orig = api._data_checks_payload


def boom(content, filename):
    raise RuntimeError(f"disk read failed at {SECRET_PATH}")


api._data_checks_payload = boom
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = post(filename="fail.csv")
api._data_checks_payload = orig
d = r.json()
check("failure shape {success:false,error}", set(d) == {"success", "error"} and d["success"] is False, str(d))
check("fixed message", d.get("error") == api.DATA_CHECKS_ERROR)
check("no exception text / path in response", SECRET_PATH not in r.text and "RuntimeError" not in r.text
      and "disk read" not in r.text)
check("detail logged server-side", SECRET_PATH in buf.getvalue())

# unparseable file through the real payload function: also fixed message
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    d = post(url="local://bad", filename="bad.csv").json()
check("unparseable or empty file -> never leaks", d.get("success") is True or d.get("error") == api.DATA_CHECKS_ERROR,
      str(d)[:200])

# existing routes untouched
paths = {r.path for r in api.app.routes}
check("existing routes still present", {"/clean", "/profile", "/analyze", "/upload", "/clean/agent"} <= paths)

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    raise SystemExit(1)
print("all passed")
