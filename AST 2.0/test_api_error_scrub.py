"""
test_api_error_scrub.py — the API boundary scrubs every service result: no "Traceback", no ".py", no "C:\\" reaches the
client, while the final exception line stays. Services are stubbed (no network, no Groq).

  * one case per route family: analyze, rag/index, rag/query, rag/insights, profile, clean, data-checks, clean/agent,
    auto-analyze, auto-analyze/follow-up, multi-table/query
  * a nested case ("charts_error" inside an auto-analyze result)
  * a success response is returned unchanged
  * the upload error ("Upload failed: ...") no longer carries a disk path; the 413 / 400 wording is unchanged

Run: python test_api_error_scrub.py
"""
import os
import shutil
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="api_scrub_test_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(TMP, "audit.jsonl")
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
for k in ("RATE_LIMIT_ENABLED", "RATE_LIMIT_MULTIPLIER", "CACHE_ENABLED"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient

import api
import audit_log
import cache_layer
import rate_limiter

os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"   # api.py's .env load can override it on import
api.SERVICE_SECRET = "s3cret-for-tests"

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


H = {"x-service-secret": "s3cret-for-tests"}
WIN = r"C:\Users\x\secret.py"
TRACEBACK = (
    "Traceback (most recent call last):\n"
    f'  File "{WIN}", line 12, in run\n'
    "    result = df['sales'].sum()\n"
    "             ^^^^^^^^^^^^^^^^^\n"
    '  File "/home/u/app/x.py", line 40, in helper\n'
    "    return self._data[key]\n"
    "KeyError: 'sales'"
)
FILE = {"fileUrl": "local://abc", "filename": "t.csv"}
CSV = b"a,b\n1,2\n3,4\n"
BAD = ("Traceback", ".py", "C:\\", "/home/", 'File "')


def leaks(text):
    return [b for b in BAD if b in text]


def post(client, path, body, key):
    rate_limiter.limiter.reset(); cache_layer.clear_all()
    return client.post(path, headers=H, json=body)


def main():
    audit_log.reset_for_tests(); rate_limiter.limiter.reset(); cache_layer.clear_all()
    client = TestClient(api.app, raise_server_exceptions=False)
    api.local_store.load = lambda url: CSV

    fail_result = lambda *a, **k: {"success": False, "error": TRACEBACK}
    cases = [
        ("analyze", "run_analysis", "/analyze", {**FILE, "query": "q", "sessionId": "s1"}),
        ("rag/index", "index_document", "/rag/index", {**FILE, "sessionId": "s2"}),
        ("rag/query", "answer_question", "/rag/query", {"docId": "d", "filename": "t.csv", "query": "q"}),
        ("rag/insights", "get_insights", "/rag/insights", {"docId": "d", "filename": "t.csv"}),
        ("profile", "get_profile", "/profile", FILE),
        ("clean", "get_cleaning_report", "/clean", FILE),
        ("data-checks", "_data_checks_response", "/data-checks", FILE),
        ("clean/agent", "run_cleaning_agent", "/clean/agent", {**FILE, "sessionId": "s3"}),
        ("auto-analyze", "run_auto_analyze", "/auto-analyze", {**FILE, "sessionId": "s4"}),
        ("auto-analyze/follow-up", "answer_followup", "/auto-analyze/follow-up",
         {**FILE, "sessionId": "s5", "insights": [{"finding": "f", "action": "a"}], "question": "q", "priorQa": []}),
        ("multi-table/query", "run_multi_table_query", "/multi-table/query",
         {"query": "q", "sessionId": "s6", "datasets": [{"alias": "a", "fileUrl": "local://1", "filename": "a.csv"},
                                                        {"alias": "b", "fileUrl": "local://2", "filename": "b.csv"}]}),
    ]
    for family, attr, path, body in cases:
        original = getattr(api, attr)
        setattr(api, attr, fail_result)
        try:
            r = post(client, path, body, family)
        finally:
            setattr(api, attr, original)
        j = r.json() if r.status_code == 200 else {}
        check(f"{family}: HTTP 200 and success false", r.status_code == 200 and j.get("success") is False, f"{r.status_code} {r.text[:120]}")
        check(f"{family}: no Traceback, no .py, no C:\\, no frame lines", not leaks(r.text), f"{leaks(r.text)} {r.text[:200]}")
        check(f"{family}: the final exception line is kept", j.get("error") == "KeyError: 'sales'", repr(j.get("error")))

    # the key set of a service result is not changed by the scrub
    api.run_analysis = lambda *a, **k: {"success": False, "resultType": "error", "resultData": None,
                                        "error": TRACEBACK, "code": "result = 1", "method": "m", "retried": True}
    r = post(client, "/analyze", {**FILE, "query": "q", "sessionId": "s7"}, "shape")
    check("analyze: every response key is still there",
          set(r.json()) == {"success", "resultType", "resultData", "error", "code", "method", "retried"}, str(sorted(r.json())))

    # nested: charts_error inside an auto-analyze result, and strings that are NOT errors stay as they were
    nested = {"success": True, "insights": [{"finding": f"see {WIN}", "action": "keep"}],
              "charts": [{"title": "t", "charts_error": TRACEBACK, "resultData": [{"error_rate": f"{WIN}"}]}],
              "charts_error": f"Chart planning failed: boom in {WIN}"}
    api.run_auto_analyze = lambda *a, **k: nested
    r = post(client, "/auto-analyze", {**FILE, "sessionId": "s8"}, "nested")
    j = r.json()
    check("nested charts_error is scrubbed (list item and top level)",
          j["charts"][0]["charts_error"] == "KeyError: 'sales'" and "C:" not in j["charts"][0 if False else 0]["charts_error"]
          and "C:\\" not in j["charts_error"] and ".py" not in j["charts_error"]
          and j["charts_error"].startswith("Chart planning failed: boom in"), str(j)[:300])
    check("nested: table data and non-error strings are not rewritten",
          j["charts"][0]["resultData"] == [{"error_rate": WIN}] and j["insights"][0]["finding"] == f"see {WIN}")
    check("nested: the service's own object was not mutated", nested["charts"][0]["charts_error"] == TRACEBACK)

    # a success response is returned unchanged
    ok = {"success": True, "resultType": "scalar", "resultData": 42, "error": None, "code": "result = 42",
          "method": "generated_code", "basis": {"rows": 2}}
    api.run_analysis = lambda *a, **k: ok
    r = post(client, "/analyze", {**FILE, "query": "q", "sessionId": "s9"}, "ok")
    check("a success response is unchanged", r.status_code == 200 and r.json() == ok, r.text[:200])

    # ---- upload --------------------------------------------------------------------------------------------------
    class BrokenWriter:
        size = 0
        def write(self, chunk): pass
        def finish(self): raise OSError(f"[Errno 28] No space left on device: '{WIN}'")
        def abort(self): pass
    real_writer = api.local_store.UploadWriter
    api.local_store.UploadWriter = BrokenWriter
    rate_limiter.limiter.reset()
    r = client.post("/upload", headers={**H, "x-filename": "t.csv"}, content=b"abc")
    api.local_store.UploadWriter = real_writer
    j = r.json()
    check("upload: an OSError with a disk path gives the safe message + reference id",
          j.get("success") is False and j["error"].startswith("Something went wrong. Reference: ") and set(j) == {"success", "error"}, str(j))
    check("upload: same status code as before (500)", r.status_code == 500, str(r.status_code))
    check("upload: no path, no .py, no Errno text in the response", not leaks(r.text) and "Errno" not in r.text and "secret" not in r.text, r.text)

    rate_limiter.limiter.reset()
    r = client.post("/upload", headers={**H, "x-filename": "empty.csv"}, content=b"")
    check("upload: the 400 message is still our own wording", r.status_code == 400 and r.json().get("error") == "The uploaded file is empty.", r.text)
    rate_limiter.limiter.reset()
    old_max = api.local_store.max_upload_bytes
    api.local_store.max_upload_bytes = lambda: 2
    r = client.post("/upload", headers={**H, "x-filename": "big.csv"}, content=b"abcdef")
    api.local_store.max_upload_bytes = old_max
    check("upload: the 413 message is still our own wording", r.status_code == 413 and r.json().get("error") == "File is larger than the upload limit.", r.text)

    print()
    shutil.rmtree(TMP, ignore_errors=True)
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
