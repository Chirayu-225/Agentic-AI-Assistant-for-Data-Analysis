"""
api.py — FastAPI service exposing /analyze to the Next.js frontend.

Run locally:
    uvicorn api:app --reload --port 8000

Run in production (e.g. Render):
    uvicorn api:app --host 0.0.0.0 --port $PORT

Required environment variables:
    PYTHON_SERVICE_SECRET — must match the Next.js app's PYTHON_SERVICE_SECRET.
                             Every request without a matching
                             X-Service-Secret header is rejected with 401.
                             This is the ONLY thing standing between this
                             sandboxed-exec endpoint and the open internet
                             on a free tier with no private networking
                             between services — see the Next.js repo's
                             lib/pythonClient.ts for the other side of this.
    GROQ_API_KEY           — required by llm_provider.py, unchanged from
                             the Streamlit app.

All the actual analysis logic lives in analysis_service.py, which has no
FastAPI dependency and can be unit tested directly — this file is
intentionally thin.
"""

from __future__ import annotations

import hmac
import math
import os
import time

from dotenv import load_dotenv
load_dotenv(override=True)  # .env always wins over a stale shell variable (e.g. an old `set PYTHON_SERVICE_SECRET=...`); reads .env in this folder — see .env.example for what's needed

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from analysis_service import run_analysis, run_analysis_from_cleaned_df
from rag_service import index_document, answer_question, get_insights
from profile_service import get_profile, get_cleaning_report
from cleaning_agent import run_cleaning_agent
from auto_analyze_service import run_auto_analyze, answer_followup
from multi_table_service import run_multi_table_query
import session_cache
import audit_log
import cache_layer
import rate_limiter

SERVICE_SECRET = os.environ.get("PYTHON_SERVICE_SECRET", "")

def _fp(value: str) -> str:
    """Short, non-reversible fingerprint so two sides can confirm they hold the
    same secret without ever printing it."""
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8] if value else "none"

print(f"[api] PYTHON_SERVICE_SECRET fingerprint: {_fp(SERVICE_SECRET) if SERVICE_SECRET else 'NOT SET'}", flush=True)
# Was 15s — sized for small test CSVs, not for the 200MB nominal upload
# ceiling this app actually advertises. On a realistic connection, a
# large CSV or a multi-hundred-page PDF can legitimately take longer
# than 15s just to *download* from Blob storage, well before parsing
# ever starts — that made this a silent, much-lower-than-advertised
# intake ceiling. 120s gives real headroom for large files without
# hanging forever on a genuinely dead/unreachable blob URL.
FETCH_TIMEOUT_SECONDS = 120

app = FastAPI(title="Analyst Assistant API")


def _client_ip(request: Request) -> str:
    # Behind a proxy (Render) the real caller is the first X-Forwarded-For hop.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def audit_and_ip_backstop(request: Request, call_next):
    """Every request gets an id (returned as X-Request-ID and stamped on
    every audit record written while handling it), a per-source-address
    rate-limit backstop, and one `request` audit record on completion.
    /health is skipped: it is polled constantly and reveals nothing."""
    path = request.url.path
    ip = _client_ip(request)
    ctx, token = audit_log.begin_request(method=request.method, path=path, ip=audit_log.sha_short(ip, 10))
    t0 = time.time()
    status = 500
    try:
        if path != "/health":
            ok, retry = rate_limiter.check("ip", ip)
            if not ok:
                status = 429
                wait = math.ceil(retry)
                audit_log.record("rate_limited", scope="ip", retry_after=wait)
                return JSONResponse(
                    {"success": False, "error": f"Too many requests. Please wait {wait}s and try again."},
                    status_code=429, headers={"Retry-After": str(wait), "X-Request-ID": ctx["request_id"]},
                )
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = ctx["request_id"]
        return response
    finally:
        if path != "/health":
            audit_log.record("request", status=status, duration_ms=round((time.time() - t0) * 1000, 1))
        audit_log.end_request(token)


class AnalyzeRequest(BaseModel):
    fileUrl: str
    filename: str
    query: str
    sessionId: str


class RagIndexRequest(BaseModel):
    fileUrl: str
    filename: str
    sessionId: str


class RagQueryRequest(BaseModel):
    docId: str
    filename: str
    query: str


class RagInsightsRequest(BaseModel):
    docId: str
    filename: str


class ProfileRequest(BaseModel):
    fileUrl: str
    filename: str


class CleanRequest(BaseModel):
    fileUrl: str
    filename: str


class CleanAgentRequest(BaseModel):
    fileUrl: str
    filename: str
    sessionId: str


class AutoAnalyzeRequest(BaseModel):
    fileUrl: str
    filename: str
    sessionId: str


class InsightItem(BaseModel):
    finding: str
    action: str


class QaPair(BaseModel):
    question: str
    answer: str


class FollowUpRequest(BaseModel):
    fileUrl: str
    filename: str
    sessionId: str
    insights: list[InsightItem]
    question: str
    priorQa: list[QaPair] = []


class DatasetRef(BaseModel):
    alias: str
    fileUrl: str
    filename: str


class MultiTableQueryRequest(BaseModel):
    datasets: list[DatasetRef]
    query: str
    sessionId: str


def _check_secret(x_service_secret: str = Header(default="")) -> None:
    if not SERVICE_SECRET:
        # Fail loudly rather than silently accepting every request —
        # a missing secret on THIS side is a misconfiguration, not "open by
        # default".
        raise HTTPException(
            status_code=500,
            detail="PYTHON_SERVICE_SECRET is not configured on the server.",
        )
    # Constant-time comparison: a plain != leaks, via response timing,
    # how many leading characters of a guess were right.
    if not hmac.compare_digest(x_service_secret.encode("utf-8"), SERVICE_SECRET.encode("utf-8")):
        audit_log.record("auth_failure", reason="bad_or_missing_secret")
        print(f"[api] auth rejected: client sent secret fingerprint {_fp(x_service_secret)}, server expects {_fp(SERVICE_SECRET)}", flush=True)
        raise HTTPException(status_code=401, detail="Invalid or missing service secret.")


def _limit(group: str, key: str, session_id: str | None = None, filename: str | None = None) -> None:
    """Per-caller rate limit (see rate_limiter.py for the policy table and
    why it keys on sessionId, not IP). Also binds who/what to the audit
    context so every record for this request carries it."""
    audit_log.bind(session_id=session_id, filename=(filename or "")[:80] or None)
    ok, retry = rate_limiter.check(group, key or "anonymous")
    if not ok:
        wait = math.ceil(retry)
        audit_log.record("rate_limited", scope=group, key_hash=audit_log.sha_short(key), retry_after=wait)
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {wait}s and try again.",
            headers={"Retry-After": str(wait)},
        )


def _fetch_blob(file_url: str) -> tuple[bytes | None, dict | None]:
    """Shared by /analyze and /rag/index — fetch the uploaded file from
    Blob storage. Returns (bytes, None) on success or (None, error_dict)."""
    cached = cache_layer.blob_cache.get(file_url)
    if cached is not None:
        return cached, None
    try:
        resp = requests.get(file_url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        # Blob URLs are unique per upload, so (url -> bytes) can't go stale;
        # see cache_layer.py. Files too big for the cache just aren't stored.
        cache_layer.blob_cache.put(file_url, resp.content, len(resp.content))
        return resp.content, None
    except Exception as e:
        return None, {"success": False, "error": f"Could not fetch file from storage: {e}"}


@app.get("/health")
def health():
    # Cheap endpoint for Render's health checks / manual "is it up" pings —
    # deliberately requires no secret, since it reveals nothing sensitive.
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("medium", req.sessionId, req.sessionId, req.filename)

    # session_cache: a follow-up question against the exact same upload
    # (same sessionId + same fileUrl — see session_cache.py's module
    # docstring for why that pair is a safe, sufficient key) skips the
    # Blob fetch AND the parse/clean entirely, straight to query routing.
    # A cache miss here is always just "proceed as before" — never an
    # error — so caching being cold, expired, or unavailable degrades to
    # the exact pre-caching behavior below, not a broken one.
    cached = session_cache.get(req.sessionId, req.fileUrl)
    if cached is not None:
        try:
            return run_analysis_from_cleaned_df(cached.cleaned_df, req.query, req.sessionId)
        except Exception as e:
            return {
                "success": False, "resultType": "error", "resultData": None,
                "error": f"Unexpected server error: {e}", "confidence": "LOW",
            }

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return {**err, "resultType": "error", "resultData": None, "confidence": "LOW"}

    try:
        return run_analysis(content, req.filename, req.query, req.sessionId, file_url=req.fileUrl)
    except Exception as e:
        # Last-resort catch-all — analysis_service.py has its own internal
        # guards, but a public endpoint should never return a raw 500
        # regardless. If this ever fires, it means something slipped past
        # those internal guards and is worth investigating directly.
        return {
            "success": False, "resultType": "error", "resultData": None,
            "error": f"Unexpected server error: {e}", "confidence": "LOW",
        }


@app.post("/rag/index")
def rag_index(req: RagIndexRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("light", req.sessionId, req.sessionId, req.filename)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return index_document(content, req.filename, req.sessionId)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/rag/query")
def rag_query(req: RagQueryRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("medium", req.docId, filename=req.filename)

    try:
        return answer_question(req.docId, req.query, req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/rag/insights")
def rag_insights(req: RagInsightsRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("medium", req.docId, filename=req.filename)

    try:
        return get_insights(req.docId, req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/profile")
def profile(req: ProfileRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("light", "file:" + audit_log.sha_short(req.fileUrl), filename=req.filename)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return cache_layer.cached_result(
            "profile", content, lambda: get_profile(content, req.filename), extra=req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/clean")
def clean(req: CleanRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("light", "file:" + audit_log.sha_short(req.fileUrl), filename=req.filename)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return cache_layer.cached_result(
            "clean", content, lambda: get_cleaning_report(content, req.filename), extra=req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/clean/agent")
def clean_agent(req: CleanAgentRequest, x_service_secret: str = Header(default="")):
    # Separate, explicitly user-triggered endpoint — see cleaning_agent.py's
    # docstring for why this is never called automatically the way /clean is.
    _check_secret(x_service_secret)
    _limit("heavy", req.sessionId, req.sessionId, req.filename)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return run_cleaning_agent(content, req.filename, req.sessionId)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/auto-analyze")
def auto_analyze(req: AutoAnalyzeRequest, x_service_secret: str = Header(default="")):
    # Separate, explicitly user-triggered endpoint — costs several LLM
    # calls (insights + chart plan + up to 3 chart code-gens), see
    # auto_analyze_service.py's docstring for why this is never automatic.
    _check_secret(x_service_secret)
    _limit("heavy", req.sessionId, req.sessionId, req.filename)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return run_auto_analyze(content, req.filename, req.sessionId)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/auto-analyze/follow-up")
def auto_analyze_followup(req: FollowUpRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("medium", req.sessionId, req.sessionId, req.filename)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return answer_followup(
            content, req.filename,
            [i.model_dump() for i in req.insights],
            req.question,
            [qa.model_dump() for qa in req.priorQa],
            req.sessionId,
        )
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/multi-table/query")
def multi_table_query(req: MultiTableQueryRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)
    _limit("medium", req.sessionId, req.sessionId, "multi-table")

    if len(req.datasets) < 2:
        return {"success": False, "error": "At least 2 datasets are needed for a multi-table query."}

    datasets = []
    for ds in req.datasets:
        content, err = _fetch_blob(ds.fileUrl)
        if err is not None:
            return {"success": False, "error": f"Could not fetch '{ds.filename}': {err.get('error', 'unknown error')}"}
        datasets.append({"alias": ds.alias, "csv_bytes": content, "filename": ds.filename})

    try:
        return run_multi_table_query(datasets, req.query, req.sessionId)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.get("/admin/stats")
def admin_stats(x_service_secret: str = Header(default="")):
    """Operational view for the owner: cache hit rates, rate-limiter
    counters, and whether the audit log's hash chain is still intact.
    Same shared-secret protection as every other endpoint."""
    _check_secret(x_service_secret)
    return {
        "cache": cache_layer.stats(),
        "rateLimiter": {"enabled": rate_limiter.enabled(), **rate_limiter.limiter.stats()},
        "audit": {
            "chain": audit_log.verify_chain(),
            "eventCounts": audit_log.summarize(),
        },
    }
