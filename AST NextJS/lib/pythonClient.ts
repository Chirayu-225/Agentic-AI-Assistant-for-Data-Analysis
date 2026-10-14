/**
 * lib/pythonClient.ts — the ONLY place that talks to the Python service.
 *
 * Two env vars required (see .env.example):
 *   PYTHON_SERVICE_URL   — e.g. https://your-python-service.onrender.com
 *   PYTHON_SERVICE_SECRET — a shared secret, checked by the Python service
 *
 * WHY THE SHARED SECRET
 *   Render's free tier has no private networking between services — the
 *   Python service is reachable on the open internet the same as this
 *   Next.js app is. Without a secret, anyone who finds that URL could call
 *   your sandboxed exec endpoint directly, bypassing this app's auth/rate
 *   limiting entirely. The Python service should reject any request
 *   missing or mismatching this header — see the FastAPI wrapper's
 *   dependency for the corresponding check.
 */

import { createHash } from "crypto";
import { NextResponse } from "next/server";

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL?.replace(/\/+$/, "");
const PYTHON_SERVICE_SECRET = process.env.PYTHON_SERVICE_SECRET;

// Non-reversible fingerprint (first 8 hex chars of SHA-256) so you can confirm the
// frontend and backend hold the same secret without printing it.
console.log(
  "[pythonClient] PYTHON_SERVICE_SECRET fingerprint:",
  PYTHON_SERVICE_SECRET
    ? createHash("sha256").update(PYTHON_SERVICE_SECRET).digest("hex").slice(0, 8)
    : "NOT SET"
);

if (!PYTHON_SERVICE_URL || !PYTHON_SERVICE_SECRET) {
  // Fails loudly at import time in dev rather than silently 500ing later.
  console.warn(
    "[pythonClient] PYTHON_SERVICE_URL or PYTHON_SERVICE_SECRET is not set — " +
      "requests to the Python service will fail. See .env.example."
  );
}

// Default timeout for most calls. analysis_service.py gives chart-generation
// queries up to 25s (Windows spawn cold-start overhead in local dev) — this
// MUST stay above that, or Next.js aborts the fetch first and you see a
// generic timeout error here instead of Python's own, more specific one.
const FETCH_TIMEOUT_MS = 35_000;

// Document indexing (parse + chunk + embed) needs more room than a chart
// query: embedding generation is CPU-bound, and the very first call after
// any uvicorn restart pays a one-time cold-load cost for the
// sentence-transformers model (plus, per the terminal warnings you've seen,
// an unauthenticated Hugging Face Hub call). 35s was cutting this off
// mid-request — this is what "The operation was aborted due to timeout"
// actually meant: our own client-side abort firing, not a Python error.
const INDEX_TIMEOUT_MS = 90_000;

export interface AnalyzeRequest {
  fileUrl: string; // Vercel Blob URL — the Python service fetches the CSV from here
  filename: string;
  query: string;
  sessionId: string; // per-user namespace — see the isolation work in sandbox_proxy/rag_engine
}

export interface AnalyzeResponse {
  success: boolean;
  resultType: "table" | "scalar" | "chart" | "error";
  resultData: unknown; // shape depends on resultType — table rows, a number/string, or a Plotly figure JSON
  error?: string;
  confidence?: "HIGH" | "MEDIUM" | "LOW";
}

export interface RagIndexRequest {
  fileUrl: string;
  filename: string;
  sessionId: string;
}

export interface ExtractedTable {
  title: string;
  page: number;
  row_count: number;
  col_count: number;
  preview: Record<string, unknown>[];
  csv_base64?: string; // present only if the Blob upload step failed
  url?: string; // Vercel Blob URL — present on success, ready to feed straight into the CSV Analysis pipeline
}

export interface RagIndexResponse {
  success: boolean;
  doc_id?: string;
  filename?: string;
  char_count?: number;
  chunk_count?: number;
  word_count?: number;
  tables?: ExtractedTable[];
  error?: string;
}

export interface RagQueryRequest {
  docId: string;
  filename: string;
  query: string;
}

export interface RagQueryResponse {
  success: boolean;
  answer?: string;
  error?: string;
}

export interface RagInsightsRequest {
  docId: string;
  filename: string;
}

export interface RagInsightsResponse {
  success: boolean;
  insights?: string;
  error?: string;
}

export interface SchemaColumn {
  column: string;
  dtype: string;
  nulls: number;
}

export interface ProfileRequest {
  fileUrl: string;
  filename: string;
}

export interface ProfileResponse {
  success: boolean;
  filename?: string;
  totalRows?: number;
  totalColumns?: number;
  numericCols?: number;
  nullValues?: number;
  schema?: SchemaColumn[];
  preview?: Record<string, unknown>[];
  previewCount?: number;
  piiReport?: PiiReport | null;
  error?: string;
}

export interface PiiFlaggedColumn {
  column: string;
  severity: "high" | "medium" | "low";
  categories: Record<string, { count: number; pct: number }>;
  nameHint: "personal" | "review" | null;
  withheldFromLlm: boolean;
}

export interface PiiReport {
  risk: "none" | "low" | "medium" | "high";
  flaggedColumns: PiiFlaggedColumn[];
  summary: string;
  scannedValuesPerColumn: number;
}

export interface CleanRequest {
  fileUrl: string;
  filename: string;
}

export interface CleanResponse {
  success: boolean;
  filename?: string;
  log?: string[];
  before?: { rows: number; columns: number; nullValues: number };
  after?: { rows: number; columns: number; nullValues: number };
  schema?: SchemaColumn[];
  preview?: Record<string, unknown>[];
  previewCount?: number;
  error?: string;
}

export class RateLimitedError extends Error {
  retryAfterSeconds: number;
  constructor(retryAfterSeconds: number) {
    super(`Too many requests. Please wait ${retryAfterSeconds}s and try again.`);
    this.name = "RateLimitedError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

/**
 * Shared catch-block response for every route that forwards to the Python
 * service. A rate-limit rejection becomes a real HTTP 429 with Retry-After
 * (the user did nothing wrong and the service is fine); anything else keeps
 * the original 502 "<label>: <message>" behavior.
 */
export function upstreamErrorResponse(err: unknown, label: string): NextResponse {
  if (err instanceof RateLimitedError) {
    return NextResponse.json(
      { error: err.message },
      { status: 429, headers: { "Retry-After": String(err.retryAfterSeconds) } }
    );
  }
  return NextResponse.json(
    { error: `${label}: ${(err as Error).message}` },
    { status: 502 }
  );
}

async function callPythonService<TResponse>(
  path: string,
  body: unknown,
  timeoutMs: number = FETCH_TIMEOUT_MS
): Promise<TResponse> {
  const res = await fetch(`${PYTHON_SERVICE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Service-Secret": PYTHON_SERVICE_SECRET ?? "",
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });

  if (res.status === 429) {
    // The Python service's rate limiter (rate_limiter.py) said slow down.
    // This is a normal, expected outcome — not an outage — so it gets its
    // own error type and a plain-language message instead of a raw
    // "Python service error (429): {...json...}" string.
    const retryAfter = Number(res.headers.get("Retry-After")) || 30;
    throw new RateLimitedError(retryAfter);
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Python service error (${res.status}): ${text || res.statusText}`);
  }

  return res.json() as Promise<TResponse>;
}

export function analyzeCsv(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  return callPythonService<AnalyzeResponse>("/analyze", req);
}

export function indexDocument(req: RagIndexRequest): Promise<RagIndexResponse> {
  return callPythonService<RagIndexResponse>("/rag/index", req, INDEX_TIMEOUT_MS);
}

export function queryDocument(req: RagQueryRequest): Promise<RagQueryResponse> {
  return callPythonService<RagQueryResponse>("/rag/query", req);
}

export function getDocumentInsights(req: RagInsightsRequest): Promise<RagInsightsResponse> {
  return callPythonService<RagInsightsResponse>("/rag/insights", req);
}

// /profile and /clean both do real work before any LLM call: read_tabular_file()
// on the Python side, which for a multi-sheet Excel workbook means real
// openpyxl/pandas parse time (was much worse before file_reader.py's
// sheet-selection fix — see that file's _pick_best_excel_sheet docstring
// for the live bug this was filed from: a 5-sheet, 18MB workbook took
// 42s just to pick which sheet to use, well past the old default 35s
// here, which is exactly what "Profiling service unavailable: The
// operation was aborted due to timeout" meant — our own client-side
// abort firing before Python ever got to answer, same as INDEX_TIMEOUT_MS's
// note above). Even after that fix, a large single sheet still takes
// real time to parse (that same workbook's winning sheet alone: ~24s) —
// this constant gives that genuine parse time headroom instead of
// assuming a CSV-speed read.
const FILE_READ_TIMEOUT_MS = 150_000; // large Excel workbooks can take 60s+ to parse; was 60_000

export function getProfile(req: ProfileRequest): Promise<ProfileResponse> {
  return callPythonService<ProfileResponse>("/profile", req, FILE_READ_TIMEOUT_MS);
}

export interface CleanAgentRequest {
  fileUrl: string;
  filename: string;
  sessionId: string;
}

export interface DataQualityDefect {
  column: string;
  type: string;
  evidence: string;
  action: string;
}

export interface CleanAgentResponse {
  success: boolean;
  filename?: string;
  ruleLog?: string[];
  defects?: DataQualityDefect[];
  agentUsed?: boolean;
  reasoning?: string[];
  before?: { rows: number; columns: number; nullValues: number };
  after?: { rows: number; columns: number; nullValues: number };
  preview?: Record<string, unknown>[];
  previewCount?: number;
  error?: string;
}

export function getCleaningReport(req: CleanRequest): Promise<CleanResponse> {
  // Same read_tabular_file() cost as /profile, same reasoning — see
  // FILE_READ_TIMEOUT_MS's comment above.
  return callPythonService<CleanResponse>("/clean", req, FILE_READ_TIMEOUT_MS);
}

// LIVE BUG: a 5-sheet, 18MB workbook (hotel_revenue_historical_full-2.xlsx,
// the same file that filed the /profile timeout above) took 46253ms here
// and got aborted at the old 45s ceiling — "Cleaning agent service
// unavailable: The operation was aborted due to timeout", again our own
// client-side abort, not a Python crash (backend log showed the request
// reached the LLM stage and printed "[cleaning-agent] 0 defects
// diagnosed" — it was still running, just past our deadline).
//
// WHY THIS ROUTE IS SLOWER THAN /profile'S FILE_READ_TIMEOUT_MS ALONE:
// cleaning_agent.py's run_cleaning_agent() does NOT share session_cache
// (that cache only covers /analyze) and does NOT reuse the /profile or
// /clean call the user already made — it calls read_tabular_file() on
// its own, fresh, every time. So this endpoint pays that same ~24s
// large-sheet parse cost (see file_reader.py's _pick_best_excel_sheet
// docstring) EVERY time it's invoked, THEN does rule_based_clean(),
// THEN one LLM call to diagnose defects, and — only if defects are
// found — a SECOND LLM call to generate a fix plus a sandboxed exec
// (run_sandboxed(..., timeout_seconds=20)) on top of that. 45s was sized
// as if this were one LLM call over a small in-memory frame; it doesn't
// have room for a real file parse plus up to two LLM calls plus a
// sandbox exec, stacked sequentially. 120s covers the worst realistic
// case (large-sheet parse + diagnose call + fix call + a full 20s
// sandbox timeout) with headroom, same order of magnitude as this app's
// other worst-case budget (FETCH_TIMEOUT_SECONDS on the Python side is
// already 120s for exactly this class of large-file cost).
const CLEAN_AGENT_TIMEOUT_MS = 120_000;

export function runCleaningAgent(req: CleanAgentRequest): Promise<CleanAgentResponse> {
  return callPythonService<CleanAgentResponse>("/clean/agent", req, CLEAN_AGENT_TIMEOUT_MS);
}

export interface AutoAnalyzeRequest {
  fileUrl: string;
  filename: string;
  sessionId: string;
}

export interface AutoAnalyzeInsight {
  finding: string;
  action: string;
}

export interface AutoAnalyzeChart {
  title: string;
  why: string;
  resultType: "chart" | "error";
  resultData: unknown; // Plotly figure JSON when resultType is "chart"
  error?: string | null;
}

export interface AutoAnalyzeResponse {
  success: boolean;
  filename?: string;
  insights?: AutoAnalyzeInsight[];
  charts?: AutoAnalyzeChart[];
  chartsError?: string | null;
  error?: string;
}

// Costs the most of any endpoint here: 1 LLM call for insights, 1 for the
// chart plan, then up to 3 more LLM calls + sandboxed execs (one per
// chart), all sequential. Give it more room than the cleaning agent's
// single LLM call + single sandbox exec.
const AUTO_ANALYZE_TIMEOUT_MS = 75_000;

export function runAutoAnalyze(req: AutoAnalyzeRequest): Promise<AutoAnalyzeResponse> {
  return callPythonService<AutoAnalyzeResponse>("/auto-analyze", req, AUTO_ANALYZE_TIMEOUT_MS);
}

export interface FollowUpQaPair {
  question: string;
  answer: string;
}

export interface FollowUpRequest {
  fileUrl: string;
  filename: string;
  sessionId: string;
  insights: AutoAnalyzeInsight[];
  question: string;
  priorQa: FollowUpQaPair[];
}

export interface FollowUpResponse {
  success: boolean;
  answer?: string;
  error?: string;
}

// LIVE BUG: same file as the /clean/agent and /profile bugs above
// (hotel_revenue_historical_full-2.xlsx) 502'd here too — "Follow-up
// service unavailable: The operation was aborted due to timeout" at
// 30522ms, just past the old 30s ceiling. Same root cause as
// /clean/agent: auto_analyze_service.answer_followup() calls
// _read_csv()/read_tabular_file() fresh on every single follow-up
// question — it does NOT share session_cache (/analyze-only) and does
// NOT reuse the file the user already loaded for the auto-analyze run
// that produced the insights being asked about. So every follow-up
// pays the same ~24s large-sheet parse cost (see file_reader.py's
// _pick_best_excel_sheet docstring) on top of up to 2 LLM calls
// (decide, then answer) plus up to 3 sandboxed code execs for the
// "explanatory angles" the decision step can request. 30s was sized
// for that LLM+sandbox work alone, assuming a cheap, already-parsed
// dataset — it was never going to survive a large multi-sheet Excel
// file's own read cost. 90s gives room for the read plus that full
// LLM/sandbox chain, mirroring the /clean/agent fix above for the
// same reason.
const FOLLOWUP_TIMEOUT_MS = 90_000;

export function askFollowUp(req: FollowUpRequest): Promise<FollowUpResponse> {
  return callPythonService<FollowUpResponse>("/auto-analyze/follow-up", req, FOLLOWUP_TIMEOUT_MS);
}

export interface DatasetRef {
  alias: string;
  fileUrl: string;
  filename: string;
}

export interface MultiTableQueryRequest {
  datasets: DatasetRef[];
  query: string;
  sessionId: string;
}

export interface MultiTableQueryResponse {
  success: boolean;
  reasoning?: string;
  tables?: string[];
  resultType?: "table";
  resultData?: Record<string, unknown>[];
  rowCount?: number;
  columns?: string[];
  error?: string;
}

// Not live-bug-confirmed yet, but bumped preemptively for the same
// reason as /clean/agent and /auto-analyze/follow-up above: every
// dataset in the join that ISN'T routed through the large-CSV/DuckDB
// path (should_use_large_csv_path is CSV/TXT-only — see file_reader.py
// — so ANY Excel file in a join always goes through pandas/openpyxl
// regardless of size) gets a fresh, uncached read_tabular_file() call
// here too. Join a workbook shaped like hotel_revenue_historical_full-2
// .xlsx (the file behind all three confirmed timeouts above) into a
// multi-table query, and this endpoint pays that same ~24s parse cost
// on top of the SQL-generation LLM call and the sandboxed join exec —
// 45s was sized like the other endpoints were, without that cost in
// mind. Matching the other fixes' order of magnitude rather than
// waiting for this to also 502 in the field.
const MULTI_TABLE_TIMEOUT_MS = 90_000;

export function runMultiTableQuery(req: MultiTableQueryRequest): Promise<MultiTableQueryResponse> {
  return callPythonService<MultiTableQueryResponse>("/multi-table/query", req, MULTI_TABLE_TIMEOUT_MS);
}
