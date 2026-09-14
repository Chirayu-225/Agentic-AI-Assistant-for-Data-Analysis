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

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL?.replace(/\/+$/, "");
const PYTHON_SERVICE_SECRET = process.env.PYTHON_SERVICE_SECRET;

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
  error?: string;
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

export function getProfile(req: ProfileRequest): Promise<ProfileResponse> {
  return callPythonService<ProfileResponse>("/profile", req);
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
  return callPythonService<CleanResponse>("/clean", req);
}

// The cleaning agent involves an LLM call plus a sandboxed exec — same
// order-of-magnitude latency as chart generation, not a plain fetch.
const CLEAN_AGENT_TIMEOUT_MS = 45_000;

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

// Up to 2 LLM calls (decide-if-code-needed, then answer) plus a possible
// sandbox exec — smaller than a full auto-analyze run but still real
// LLM latency, not a cheap lookup.
const FOLLOWUP_TIMEOUT_MS = 30_000;

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

// One LLM call (SQL generation) + one sandboxed exec, same order of
// magnitude as a single-table analyze query — but joins/aggregations
// across real files can take a bit longer than a single-table query.
const MULTI_TABLE_TIMEOUT_MS = 45_000;

export function runMultiTableQuery(req: MultiTableQueryRequest): Promise<MultiTableQueryResponse> {
  return callPythonService<MultiTableQueryResponse>("/multi-table/query", req, MULTI_TABLE_TIMEOUT_MS);
}
