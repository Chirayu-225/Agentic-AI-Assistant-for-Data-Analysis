"use client";

import { useEffect, useState } from "react";
import { UploadedFile } from "@/components/FileUpload";
import QueryPanel from "@/components/QueryPanel";
import ResultDisplay from "@/components/ResultDisplay";
import InfoCards from "@/components/InfoCards";
import SystemCapabilities from "@/components/SystemCapabilities";
import DatasetProfile from "@/components/DatasetProfile";
import CleaningReport from "@/components/CleaningReport";
import CleaningAgentReport from "@/components/CleaningAgentReport";
import AutoAnalyzeReport from "@/components/AutoAnalyzeReport";
import type {
  AnalyzeResponse,
  ProfileResponse,
  CleanResponse,
  CleanAgentResponse,
  AutoAnalyzeResponse,
} from "@/lib/pythonClient";
import { getOrCreateSessionId } from "@/lib/sessionId";

const STEPS = [
  { number: "01", label: "INGEST", description: "Upload your dataset via the sidebar panel" },
  { number: "02", label: "QUERY", description: "Enter a natural language analytical question" },
  { number: "03", label: "ANALYZE", description: "Groq's open models generate and execute sandboxed Python code" },
];

const EXAMPLE_QUERIES = [
  "What are the top 5 rows by sales?",
  "Show correlation between price and volume",
  "Find all rows where revenue > 1000",
  "Group by category and sum profit",
  "What columns have null values?",
  "Calculate mean and std of numeric cols",
];

type PipelineStage = "idle" | "profiling" | "awaiting-clean-consent" | "cleaning" | "ready";
type AgentState = "idle" | "running" | "done";

export default function CsvAnalysisTab({ file }: { file: UploadedFile | null }) {
  const [stage, setStage] = useState<PipelineStage>("idle");
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [cleaning, setCleaning] = useState<CleanResponse | null>(null);
  const [agentState, setAgentState] = useState<AgentState>("idle");
  const [agentReport, setAgentReport] = useState<CleanAgentResponse | null>(null);
  const [autoAnalyzeState, setAutoAnalyzeState] = useState<AgentState>("idle");
  const [autoAnalyzeReport, setAutoAnalyzeReport] = useState<AutoAnalyzeResponse | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [querying, setQuerying] = useState(false);

  // Runs the profile step automatically on upload (it's read-only —
  // nothing about the user's data changes), then STOPS and waits for
  // explicit consent before the clean step, which rewrites values
  // (fills nulls, reparses types, etc.) — see handleConfirmClean below.
  // The cleaning AGENT is a separate, further opt-in on top of this —
  // see CleaningAgentReport / handleRunAgent.
  useEffect(() => {
    if (!file) {
      setStage("idle");
      setProfile(null);
      setCleaning(null);
      setAgentState("idle");
      setAgentReport(null);
      setAutoAnalyzeState("idle");
      setAutoAnalyzeReport(null);
      setResult(null);
      return;
    }

    let cancelled = false;

    async function runProfile() {
      setStage("profiling");
      setProfile(null);
      setCleaning(null);
      setAgentState("idle");
      setAgentReport(null);
      setAutoAnalyzeState("idle");
      setAutoAnalyzeReport(null);
      setResult(null);

      try {
        const profileRes = await fetch("/api/profile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fileUrl: file!.url, filename: file!.filename }),
        });
        const profileData = (await profileRes.json()) as ProfileResponse;
        if (cancelled) return;
        setProfile(profileData);
        setStage(profileData.success ? "awaiting-clean-consent" : "ready");
      } catch (err) {
        if (cancelled) return;
        setProfile({ success: false, error: (err as Error).message });
        setStage("ready");
      }
    }

    runProfile();
    return () => {
      cancelled = true;
    };
  }, [file]);

  async function handleConfirmClean() {
    if (!file) return;
    setStage("cleaning");
    try {
      const cleanRes = await fetch("/api/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fileUrl: file.url, filename: file.filename }),
      });
      const cleanData = (await cleanRes.json()) as CleanResponse;
      setCleaning(cleanData);
    } catch (err) {
      setCleaning({ success: false, error: (err as Error).message });
    } finally {
      setStage("ready");
    }
  }

  function handleSkipClean() {
    setStage("ready");
  }

  async function handleRunAgent() {
    if (!file) return;
    setAgentState("running");
    setAgentReport(null);

    try {
      const res = await fetch("/api/clean/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fileUrl: file.url,
          filename: file.filename,
          sessionId: getOrCreateSessionId(),
        }),
      });
      const data = (await res.json()) as CleanAgentResponse;
      setAgentReport(data);
    } catch (err) {
      setAgentReport({ success: false, error: (err as Error).message });
    } finally {
      setAgentState("done");
    }
  }

  async function handleRunAutoAnalyze() {
    if (!file) return;
    setAutoAnalyzeState("running");
    setAutoAnalyzeReport(null);

    try {
      const res = await fetch("/api/auto-analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fileUrl: file.url,
          filename: file.filename,
          sessionId: getOrCreateSessionId(),
        }),
      });
      const data = (await res.json()) as AutoAnalyzeResponse;
      setAutoAnalyzeReport(data);
    } catch (err) {
      setAutoAnalyzeReport({ success: false, error: (err as Error).message });
    } finally {
      setAutoAnalyzeState("done");
    }
  }

  async function handleQuery(query: string) {
    if (!file) return;
    setQuerying(true);
    setResult(null);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fileUrl: file.url,
          filename: file.filename,
          query,
          sessionId: getOrCreateSessionId(),
        }),
      });
      const data = (await res.json()) as AnalyzeResponse;
      setResult(data);
    } catch (err) {
      setResult({
        success: false,
        resultType: "error",
        resultData: null,
        error: (err as Error).message,
      });
    } finally {
      setQuerying(false);
    }
  }


  return (
    <div className="flex flex-col gap-6">
      <InfoCards steps={STEPS} />

      {!file && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <SystemCapabilities />
          <div className="border border-border rounded-lg bg-panel p-5">
            <p className="font-mono text-xs text-accent tracking-widest mb-4">// QUERY EXAMPLES</p>
            <div className="flex flex-col gap-2">
              {EXAMPLE_QUERIES.map((ex) => (
                <p key={ex} className="font-mono text-xs text-accent/80">&gt; {ex}</p>
              ))}
            </div>
          </div>
        </div>
      )}

      {file && (
        <>
          {stage === "profiling" && (
            <p className="font-mono text-sm text-accent animate-pulse">
              ▸ Profiling dataset...
            </p>
          )}
          {profile && <DatasetProfile profile={profile} />}

          {/* The ask, explicitly — rule-based cleaning rewrites values
              (fills nulls, reparses currency/dates/etc.), so it doesn't
              run until the user says so. */}
          {stage === "awaiting-clean-consent" && (
            <div className="border border-accent/40 rounded-lg bg-panel p-5">
              <p className="font-mono text-sm text-text mb-1">
                Clean this dataset before analyzing it?
              </p>
              <p className="font-mono text-xs text-muted mb-4">
                Deterministic, rule-based cleaning: normalizes column names, parses
                currency/percentages/dates, and fills remaining nulls (numeric columns get
                the column median, text columns get &apos;Unknown&apos;). This changes the
                values you&apos;ll be querying — you can skip it and query the raw data instead.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={handleConfirmClean}
                  className="px-4 py-2 rounded-md bg-accent text-base font-mono text-sm font-semibold"
                >
                  Clean Data
                </button>
                <button
                  onClick={handleSkipClean}
                  className="px-4 py-2 rounded-md border border-border font-mono text-sm text-muted"
                >
                  Skip — use raw data
                </button>
              </div>
            </div>
          )}
          {stage === "cleaning" && (
            <p className="font-mono text-sm text-accent animate-pulse">
              ▸ Running cleaning engine...
            </p>
          )}
          {cleaning && <CleaningReport report={cleaning} />}

          {/* Gated on reaching "ready" (post clean/skip decision), NOT on
              nulls remaining — the agent looks for general data quality
              defects (inconsistent categories, outliers, meaningful
              nulls) independent of whether any nulls are actually left,
              so it's still worth offering on a dataset with zero nulls. */}
          {stage === "ready" && agentState === "idle" && (
            <div className="border border-accent/40 rounded-lg bg-panel p-5">
              <p className="font-mono text-sm text-text mb-1">
                Let the model audit this dataset for defects rule-based cleaning can&apos;t catch.
              </p>
              <p className="font-mono text-xs text-muted mb-4">
                The data quality agent looks for things pattern-matching structurally
                can&apos;t — inconsistent category spellings (&quot;USA&quot; / &quot;U.S.&quot;),
                implausible-but-not-malformed values, and meaningful missing values — then
                proposes and applies fixes. This calls the LLM and may take longer than the
                instant rule-based pass above.
              </p>
              <button
                onClick={handleRunAgent}
                className="px-4 py-2 rounded-md bg-accent text-base font-mono text-sm font-semibold"
              >
                Run Data Quality Agent
              </button>
            </div>
          )}
          {agentState === "running" && (
            <p className="font-mono text-sm text-accent animate-pulse">
              ▸ Auditing dataset for data quality defects...
            </p>
          )}
          {agentReport && <CleaningAgentReport report={agentReport} />}

          {/* User-triggered, same reasoning as the cleaning agent above —
              costs several LLM calls, so it never runs automatically. */}
          {stage === "ready" && autoAnalyzeState === "idle" && (
            <div className="border border-accent/40 rounded-lg bg-panel p-5">
              <p className="font-mono text-sm text-text mb-1">
                Let the model find what matters in this dataset on its own.
              </p>
              <p className="font-mono text-xs text-muted mb-4">
                Auto-Analyze reads the whole dataset, ranks the most impactful business
                insights, and builds up to 3 charts chosen specifically to make those
                insights visible at a glance. This calls the LLM several times and can take
                20&ndash;40s.
              </p>
              <button
                onClick={handleRunAutoAnalyze}
                className="px-4 py-2 rounded-md bg-accent text-base font-mono text-sm font-semibold"
              >
                Run Auto-Analyze
              </button>
            </div>
          )}
          {autoAnalyzeState === "running" && (
            <p className="font-mono text-sm text-accent animate-pulse">
              ▸ Analyzing dataset and generating insights...
            </p>
          )}
          {autoAnalyzeReport && file && (
            <AutoAnalyzeReport
              report={autoAnalyzeReport}
              fileUrl={file.url}
              filename={file.filename}
            />
          )}

          <QueryPanel disabled={stage !== "ready" || querying} onSubmit={handleQuery} />

          {querying && (
            <p className="font-mono text-sm text-accent animate-pulse">
              ▸ Executing in secure sandbox...
            </p>
          )}

          {result && <ResultDisplay result={result} />}
        </>
      )}
    </div>
  );
}
