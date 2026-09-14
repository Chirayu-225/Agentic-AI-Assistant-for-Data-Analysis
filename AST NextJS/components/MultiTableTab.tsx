"use client";

import { useState } from "react";
import FileUpload, { UploadedFile } from "@/components/FileUpload";
import { TableResult } from "@/components/ResultDisplay";
import { getOrCreateSessionId } from "@/lib/sessionId";
import type { MultiTableQueryResponse } from "@/lib/pythonClient";

interface RegisteredDataset extends UploadedFile {
  alias: string;
}

function defaultAliasFromFilename(filename: string): string {
  return filename
    .replace(/\.[^.]+$/, "") // drop extension
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase() || "table";
}

export default function MultiTableTab() {
  const [datasets, setDatasets] = useState<RegisteredDataset[]>([]);
  const [showUploadSlot, setShowUploadSlot] = useState(true);
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<MultiTableQueryResponse | null>(null);

  function handleDatasetUploaded(file: UploadedFile) {
    setDatasets((prev) => [...prev, { ...file, alias: defaultAliasFromFilename(file.filename) }]);
    setShowUploadSlot(false);
  }

  function updateAlias(index: number, alias: string) {
    setDatasets((prev) => prev.map((d, i) => (i === index ? { ...d, alias } : d)));
  }

  function removeDataset(index: number) {
    setDatasets((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleRunQuery() {
    if (datasets.length < 2 || !query.trim()) return;
    setRunning(true);
    setResult(null);
    try {
      const response = await fetch("/api/multi-table/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datasets: datasets.map((d) => ({ alias: d.alias, fileUrl: d.url, filename: d.filename })),
          query: query.trim(),
          sessionId: getOrCreateSessionId(),
        }),
      });
      const data = (await response.json()) as MultiTableQueryResponse;
      setResult(data);
    } catch (err) {
      setResult({ success: false, error: (err as Error).message });
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="border border-border rounded-lg bg-panel p-5">
        <p className="font-mono text-xs text-accent tracking-widest mb-1">
          // MULTI-DATASET QUERY
        </p>
        <p className="font-mono text-xs text-muted mb-4">
          Upload 2 or more datasets, give each a short name, then ask a question that spans
          them — joins, aggregations across files, anything a SQL query could answer. Runs on
          real SQL (DuckDB) inside the same sandboxed pipeline as every other query in this
          app, not a text summary — no numbers here are guessed.
        </p>

        {datasets.length > 0 && (
          <div className="flex flex-col gap-2 mb-4">
            {datasets.map((d, i) => (
              <div
                key={i}
                className="flex items-center gap-3 border border-border rounded-md px-3 py-2"
              >
                <span className="font-mono text-[11px] text-muted shrink-0">table alias:</span>
                <input
                  type="text"
                  value={d.alias}
                  onChange={(e) => updateAlias(i, e.target.value)}
                  className="bg-base border border-border rounded px-2 py-1 font-mono text-xs text-accent w-32"
                />
                <span className="font-mono text-[11px] text-text truncate flex-1">
                  {d.filename}
                </span>
                <button
                  onClick={() => removeDataset(i)}
                  className="font-mono text-[11px] text-red-400 hover:text-red-300 shrink-0"
                >
                  remove
                </button>
              </div>
            ))}
          </div>
        )}

        {showUploadSlot ? (
          <FileUpload
            compact
            onUploaded={handleDatasetUploaded}
            accept=".csv,.xlsx,.xls"
            label={`// ADD DATASET ${datasets.length + 1}`}
            helperText="CSV, XLSX, XLS"
          />
        ) : (
          <button
            onClick={() => setShowUploadSlot(true)}
            className="font-mono text-xs text-accent border border-accent/40 rounded-md px-3 py-1.5"
          >
            + Add another dataset
          </button>
        )}
      </div>

      {datasets.length >= 2 && (
        <div className="border border-border rounded-lg bg-panel p-5">
          <p className="font-mono text-[11px] text-muted tracking-widest mb-2">
            ASK A QUESTION ACROSS {datasets.length} DATASETS
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleRunQuery()}
              disabled={running}
              placeholder={`e.g. "top 5 ${datasets[0]?.alias ?? "rows"} joined with ${datasets[1]?.alias ?? "the other table"} by shared key"`}
              className="flex-1 bg-base border border-border rounded-md px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-50"
            />
            <button
              onClick={handleRunQuery}
              disabled={running || !query.trim()}
              className="px-4 py-2 rounded-md bg-accent text-base font-mono text-xs font-semibold disabled:opacity-40 shrink-0"
            >
              {running ? "Running..." : "Run Query"}
            </button>
          </div>
        </div>
      )}

      {running && (
        <p className="font-mono text-sm text-accent animate-pulse">
          ▸ Generating SQL and querying {datasets.length} dataset(s)...
        </p>
      )}

      {result && !result.success && (
        <div className="border border-red-900 bg-red-950/30 rounded-lg p-4">
          <p className="font-mono text-xs text-red-400 mb-1">// QUERY ERROR</p>
          <p className="text-red-300 text-sm">{result.error ?? "Query failed."}</p>
        </div>
      )}

      {result && result.success && (
        <div className="border border-accent/40 rounded-lg bg-panel p-5">
          {result.reasoning && (
            <p className="font-mono text-xs text-[#818CF8] mb-3">{result.reasoning}</p>
          )}
          <p className="font-mono text-[11px] text-muted tracking-widest mb-3">
            {result.rowCount} ROW(S)
          </p>
          <TableResult data={result.resultData} />
        </div>
      )}
    </div>
  );
}
