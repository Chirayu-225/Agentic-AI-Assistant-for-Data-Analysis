"""
test_api_error_handler.py — the API never shows an exception, a path or a file name to the user.

 1. a route that catches its own exception (/analyze) answers with the safe text + a reference id, in the SAME
    response shape (success / resultType / resultData / error), HTTP 200;
 2. a route with no try/except of its own (/admin/stats) is caught by the top-level handler: HTTP 200 + safe text;
    the full trace is logged server-side under the same reference id;
 3. HTTPException (401 bad secret) is untouched;
 4. end to end: model-written code that hits numpy's re-raised TypeError (np.issubdtype on a text column with
    np.object_, which the sandbox shim deliberately does not paper over) gets one repair attempt (LLM stubbed, no
    Groq), and the final user-facing error has no "Traceback", no ".py" and no "C:\\".

No LLM, no network. Run: python test_api_error_handler.py
"""
import io
import os
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

TMP = tempfile.mkdtemp(prefix="api_err_test_")
os.environ["AUDIT_LOG_PATH"] = os.path.join(TMP, "audit.jsonl")
os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"
for k in ("RATE_LIMIT_ENABLED", "RATE_LIMIT_MULTIPLIER", "CACHE_ENABLED"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient

import api
import audit_log
import cache_layer
import query_engine
import rate_limiter

os.environ["PYTHON_SERVICE_SECRET"] = "s3cret-for-tests"   # api.py's .env load can override it on import
api.SERVICE_SECRET = "s3cret-for-tests"

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


H = {"x-service-secret": "s3cret-for-tests"}
SECRET_TEXT = "boom C:\\secret\\x.py line 7"
REF = re.compile(r"^Something went wrong\. Reference: [0-9a-f]{8}$")
CSV = b"team,pts\na,3\nb,4\na,5\nc,6\n"


def leaks(text):
    return [bad for bad in ("Traceback", ".py", "C:\\", "secret", "boom") if bad in text]


def main():
    audit_log.reset_for_tests(); rate_limiter.limiter.reset(); cache_layer.clear_all()
    client = TestClient(api.app, raise_server_exceptions=False)
    api.local_store.load = lambda url: CSV

    # 1) /analyze: the route's own catch-all uses the helper
    def boom(*a, **k):
        raise RuntimeError(SECRET_TEXT)
    original = api.run_analysis
    api.run_analysis = boom
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = client.post("/analyze", headers=H, json={"fileUrl": "local://abc", "filename": "t.csv",
                                                       "query": "anything", "sessionId": "e1"})
    api.run_analysis = original
    body = r.json()
    check("/analyze: HTTP 200", r.status_code == 200, str(r.status_code))
    check("/analyze: same shape (success, resultType, resultData, error)",
          set(body) == {"success", "resultType", "resultData", "error"} and body["success"] is False
          and body["resultType"] == "error" and body["resultData"] is None, str(body))
    check("/analyze: safe text with a reference id", bool(REF.match(body.get("error", ""))), str(body))
    check("/analyze: no exception text, no path, no .py", not leaks(r.text), str(leaks(r.text)))
    ref = body["error"].rsplit(" ", 1)[-1]
    check("/analyze: the full trace is logged server-side under the same id",
          f"ref={ref}" in buf.getvalue() and SECRET_TEXT in buf.getvalue() and "Traceback" in buf.getvalue())

    # 2) a route with no try/except: the top-level handler
    original_stats = api.cache_layer.stats
    api.cache_layer.stats = boom
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = client.get("/admin/stats", headers=H)
    api.cache_layer.stats = original_stats
    body = r.json()
    check("handler: HTTP 200, not 500", r.status_code == 200, str(r.status_code))
    check("handler: {success: false, error: 'Something went wrong. Reference: <id>'}",
          body.get("success") is False and bool(REF.match(body.get("error", ""))) and set(body) == {"success", "error"}, str(body))
    check("handler: no exception text, no path, no .py", not leaks(r.text), str(leaks(r.text)))
    ref = body["error"].rsplit(" ", 1)[-1]
    check("handler: trace logged under the same id", f"ref={ref}" in buf.getvalue() and SECRET_TEXT in buf.getvalue())

    # 3) HTTPException is left alone
    r = client.get("/admin/stats", headers={"x-service-secret": "wrong"})
    check("a bad secret is still a 401, not the generic handler", r.status_code == 401, str(r.status_code))

    # 4) end to end: model code hits the sandbox shim's re-raise -> one repair attempt -> safe final error
    prompts = []
    answers = ["result = [c for c in df.columns if np.issubdtype(df[c].dtype, np.object_)]",
               "result = [c for c in df.columns if np.issubdtype(df[c].dtype, np.bool_)]"]

    def fake_llm(prompt, temperature=0.1, session_id=""):
        prompts.append(prompt)
        return answers[min(len(prompts) - 1, len(answers) - 1)]
    query_engine.llm_with_fallback = fake_llm
    cache_layer.clear_all()
    r = client.post("/analyze", headers=H, json={"fileUrl": "local://abc", "filename": "t.csv",
                                                   "query": "which columns hold labels rather than amounts, per team",
                                                   "sessionId": "e2e-1"})
    body = r.json()
    err = body.get("error") or ""
    check("e2e: the repair attempt ran (two model calls; the second prompt carries the first error)",
          len(prompts) == 2 and "Error:" in prompts[1], f"{len(prompts)} calls")
    check("e2e: the request failed cleanly", body.get("success") is False and body.get("resultType") == "error", str(body)[:300])
    check("e2e: the user-facing error has no Traceback, no .py, no C:\\", not leaks(err), f"{leaks(err)} in {err!r}")
    check("e2e: the error is numpy's own TypeError text (the shim did not hide it)", "Cannot interpret" in err, err[:200])

    print()
    shutil.rmtree(TMP, ignore_errors=True)
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("DONE — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
