"use client";

import type { CleanResponse } from "@/lib/pythonClient";

function DeltaStat({
  label,
  before,
  after,
  lowerIsBetter = false,
}: {
  label: string;
  before: number;
  after: number;
  lowerIsBetter?: boolean;
}) {
  const changed = before !== after;
  const improved = lowerIsBetter ? after < before : after !== before;
  const diff = after - before;

  return (
    <div className="border border-border rounded-lg bg-panel px-5 py-4">
      <p className="font-mono text-[11px] text-muted tracking-widest mb-1">{label}</p>
      <div className="flex items-baseline gap-2">
        <p className="font-mono text-lg text-muted line-through opacity-60">
          {before.toLocaleString()}
        </p>
        <span className="text-muted text-sm">→</span>
        <p className={`font-mono text-xl font-bold transition-all ${changed && improved ? "text-accent" : "text-text"}`}>
          {after.toLocaleString()}
        </p>
        {changed && (
          <span className={`font-mono text-xs ${improved ? "text-accent" : "text-amber-400"}`}>
            {diff < 0 ? "▼" : "▲"} {Math.abs(diff).toLocaleString()}
          </span>
        )}
      </div>
    </div>
  );
}

// What the table looks like NOW, after the user's own choices (removing duplicate rows, filling blanks). Kept by the parent.
export interface LiveStats {
  rows: number;
  nullValues: number;
  events: string[];
}

export default function CleaningReport({ report, live }: { report: CleanResponse; live?: LiveStats | null }) {
  if (!report.success) {
    return (
      <div className="border border-red-900 bg-red-950/30 rounded-lg p-4 mb-6">
        <p className="font-mono text-xs text-red-400 mb-1">// CLEANING ERROR</p>
        <p className="text-red-300 text-sm">{report.error ?? "Cleaning failed."}</p>
      </div>
    );
  }

  const { before, after, log } = report;

  return (
    <div className="border border-border rounded-lg bg-panel p-5 mb-6">
      <p className="font-mono text-xs text-accent tracking-widest mb-4">
        // CLEANING REPORT — {report.filename}
      </p>

      {before && after && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-5">
          <DeltaStat label="ROWS" before={before.rows} after={live ? live.rows : after.rows} />
          <DeltaStat label="COLUMNS" before={before.columns} after={after.columns} />
          <DeltaStat
            label="NULL VALUES"
            before={before.nullValues}
            after={live ? live.nullValues : after.nullValues}
            lowerIsBetter
          />
        </div>
      )}

      {live && live.events.length > 0 && (
        <div className="mb-5 border border-accent/30 rounded-md bg-base px-4 py-3">
          <p className="font-mono text-[11px] text-accent tracking-widest mb-2">AFTER YOUR CHOICES</p>
          <div className="flex flex-col gap-1 font-mono text-xs text-text">
            {live.events.map((e, i) => (
              <p key={i}>• {e}</p>
            ))}
          </div>
        </div>
      )}

      <p className="font-mono text-[11px] text-muted tracking-widest mb-2">ACTIONS TAKEN</p>
      <div className="flex flex-col gap-1.5 font-mono text-xs">
        {(log ?? []).map((line: string, i: number) => (
          <p
            key={i}
            className={line.startsWith("⚠") ? "text-amber-400" : "text-text"}
          >
            {line}
          </p>
        ))}
      </div>
    </div>
  );
}
