"use client";

import { useEffect, useState } from "react";
import { UploadedFile } from "@/components/FileUpload";
import RagQueryPanel from "@/components/RagQueryPanel";
import RagResultDisplay from "@/components/RagResultDisplay";
import { getOrCreateSessionId } from "@/lib/sessionId";
import type { RagIndexResponse, RagQueryResponse, RagInsightsResponse, ExtractedTable } from "@/lib/pythonClient";

type IndexState =
  | { status: "idle" }
  | { status: "indexing" }
  | { status: "ready"; docId: string; filename: string; chunkCount: number; tables: ExtractedTable[] }
  | { status: "error"; message: string };

type ResultState =
  | { kind: "answer"; text?: string; error?: string; question: string }
  | { kind: "insights"; text?: string; error?: string }
  | null;

export default function DocumentIntelligenceTab({
  file,
  onAnalyzeTable,
}: {
  file: UploadedFile | null;
  onAnalyzeTable: (table: UploadedFile) => void;
}) {
  const [indexState, setIndexState] = useState<IndexState>({ status: "idle" });
  const [result, setResult] = useState<ResultState>(null);
  const [asking, setAsking] = useState(false);

  useEffect(() => {
    if (!file) {
      setIndexState({ status: "idle" });
      setResult(null);
      return;
    }

    let cancelled = false;

    async function runIndex() {
      setIndexState({ status: "indexing" });
      setResult(null);

      try {
        const res = await fetch("/api/rag/index", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fileUrl: file!.url,
            filename: file!.filename,
            sessionId: getOrCreateSessionId(),
          }),
        });
        const data = (await res.json()) as RagIndexResponse;
        if (cancelled) return;

        if (data.success && data.doc_id) {
          setIndexState({
            status: "ready",
            docId: data.doc_id,
            filename: file!.filename,
            chunkCount: data.chunk_count ?? 0,
            tables: (data.tables ?? []).filter((t) => t.url), // only tables that made it to Blob successfully are actionable
          });
        } else {
          setIndexState({ status: "error", message: data.error ?? "Indexing failed" });
        }
      } catch (err) {
        if (cancelled) return;
        setIndexState({ status: "error", message: (err as Error).message });
      }
    }

    runIndex();
    return () => {
      cancelled = true;
    };
  }, [file]);

  async function handleAsk(query: string) {
    if (indexState.status !== "ready") return;
    setAsking(true);
    setResult(null);

    try {
      const res = await fetch("/api/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          docId: indexState.docId,
          filename: indexState.filename,
          query,
        }),
      });
      const data = (await res.json()) as RagQueryResponse;
      setResult({ kind: "answer", text: data.answer, error: data.error, question: query });
    } catch (err) {
      setResult({ kind: "answer", error: (err as Error).message, question: query });
    } finally {
      setAsking(false);
    }
  }

  async function handleGetInsights() {
    if (indexState.status !== "ready") return;
    setAsking(true);
    setResult(null);

    try {
      const res = await fetch("/api/rag/insights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          docId: indexState.docId,
          filename: indexState.filename,
        }),
      });
      const data = (await res.json()) as RagInsightsResponse;
      setResult({ kind: "insights", text: data.insights, error: data.error });
    } catch (err) {
      setResult({ kind: "insights", error: (err as Error).message });
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {!file && (
        <p className="font-mono text-sm text-muted">
          ▸ Upload a document via the sidebar to get started.
        </p>
      )}

      {indexState.status === "indexing" && (
        <p className="font-mono text-xs text-accent animate-pulse">
          ▸ Parsing, chunking, and embedding document...
        </p>
      )}
      {indexState.status === "ready" && (
        <p className="font-mono text-xs text-accent">
          ✓ Indexed: {indexState.filename} ({indexState.chunkCount} chunks)
        </p>
      )}
      {indexState.status === "ready" && indexState.tables.length > 0 && (
        <div className="border border-accent/40 rounded-lg bg-panel p-4">
          <p className="font-mono text-[11px] text-muted tracking-widest mb-1">
            STRUCTURED DATA DETECTED
          </p>
          <p className="font-mono text-xs text-muted mb-3">
            {indexState.tables.length} real table(s) found — analyzed as data (profiling,
            cleaning, visualizations, insights), not summarized as text, so no numbers here
            are guessed from a text description.
          </p>
          <div className="flex flex-col gap-2">
            {indexState.tables.map((table, i) => (
              <div
                key={i}
                className="flex items-center justify-between border border-border rounded-md px-3 py-2"
              >
                <div>
                  <p className="font-mono text-xs text-text">{table.title}</p>
                  <p className="font-mono text-[11px] text-muted">
                    {table.row_count} rows × {table.col_count} columns
                  </p>
                </div>
                <button
                  onClick={() =>
                    onAnalyzeTable({
                      url: table.url as string,
                      filename: `${table.title.replace(/[^a-z0-9]+/gi, "_")}.csv`,
                    })
                  }
                  className="px-3 py-1.5 rounded-md bg-accent text-base font-mono text-[11px] font-semibold shrink-0"
                >
                  Analyze as Data →
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      {indexState.status === "error" && (
        <p className="font-mono text-xs text-red-400">✗ {indexState.message}</p>
      )}

      <RagQueryPanel
        disabled={indexState.status !== "ready" || asking}
        onSubmit={handleAsk}
        onGetInsights={handleGetInsights}
      />

      {asking && (
        <p className="font-mono text-sm text-accent animate-pulse">
          ▸ Retrieving relevant passages and generating answer...
        </p>
      )}

      {result && result.kind === "answer" && (
        <div>
          <p className="font-mono text-[11px] text-muted mb-2">Q: {result.question}</p>
          <RagResultDisplay kind="answer" text={result.text} error={result.error} />
        </div>
      )}
      {result && result.kind === "insights" && (
        <RagResultDisplay kind="insights" text={result.text} error={result.error} />
      )}
    </div>
  );
}
