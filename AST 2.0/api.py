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

import os

from dotenv import load_dotenv
load_dotenv()  # reads .env in this folder — see .env.example for what's needed

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from analysis_service import run_analysis
from rag_service import index_document, answer_question, get_insights
from profile_service import get_profile, get_cleaning_report
from cleaning_agent import run_cleaning_agent
from auto_analyze_service import run_auto_analyze, answer_followup
from multi_table_service import run_multi_table_query

SERVICE_SECRET = os.environ.get("PYTHON_SERVICE_SECRET", "")
FETCH_TIMEOUT_SECONDS = 15

app = FastAPI(title="Analyst Assistant API")


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
    if x_service_secret != SERVICE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing service secret.")


def _fetch_blob(file_url: str) -> tuple[bytes | None, dict | None]:
    """Shared by /analyze and /rag/index — fetch the uploaded file from
    Blob storage. Returns (bytes, None) on success or (None, error_dict)."""
    try:
        resp = requests.get(file_url, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
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

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return {**err, "resultType": "error", "resultData": None, "confidence": "LOW"}

    try:
        return run_analysis(content, req.filename, req.query, req.sessionId)
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

    try:
        return answer_question(req.docId, req.query, req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/rag/insights")
def rag_insights(req: RagInsightsRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)

    try:
        return get_insights(req.docId, req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/profile")
def profile(req: ProfileRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return get_profile(content, req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/clean")
def clean(req: CleanRequest, x_service_secret: str = Header(default="")):
    _check_secret(x_service_secret)

    content, err = _fetch_blob(req.fileUrl)
    if err is not None:
        return err

    try:
        return get_cleaning_report(content, req.filename)
    except Exception as e:
        return {"success": False, "error": f"Unexpected server error: {e}"}


@app.post("/clean/agent")
def clean_agent(req: CleanAgentRequest, x_service_secret: str = Header(default="")):
    # Separate, explicitly user-triggered endpoint — see cleaning_agent.py's
    # docstring for why this is never called automatically the way /clean is.
    _check_secret(x_service_secret)

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
