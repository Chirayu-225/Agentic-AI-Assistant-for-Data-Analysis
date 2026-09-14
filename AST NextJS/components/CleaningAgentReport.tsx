"use client";

import type { CleanAgentResponse, DataQualityDefect } from "@/lib/pythonClient";

function DeltaStat({
  label,
  before,
  after,
}: {
  label: string;
  before: number;
  after: number;
}) {
  const improved = after < before;
  return (
    <div className="border border-border rounded-lg bg-panel px-5 py-4">
      <p className="font-mono text-[11px] text-muted tracking-widest mb-1">{label}</p>
      <div className="flex items-baseline gap-2">
        <p className="font-mono text-lg text-muted line-through opacity-60">
          {before.toLocaleString()}
        </p>
        <span className="text-muted text-sm">→</span>
        <p className={`font-mono text-xl font-bold ${improved ? "text-accent" : "text-text"}`}>
          {after.toLocaleString()}
        </p>
      </div>
    </div>
  );
}

const DEFECT_TYPE_LABELS: Record<string, string> = {
  missing_values: "MISSING VALUES",
  inconsistent_categories: "INCONSISTENT CATEGORIES",
  outlier: "OUTLIER",
  mixed_format: "MIXED FORMAT",
  structural: "STRUCTURAL",
};

function DefectCard({ defect }: { defect: DataQualityDefect }) {
  return (
    <div className="border-l-2 border-[#818CF8]/40 pl-3 border border-border rounded-lg bg-panel p-3">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-accent/15 text-accent tracking-widest">
          {DEFECT_TYPE_LABELS[defect.type] ?? defect.type.toUpperCase()}
        </span>
        <span className="font-mono text-xs text-text">{defect.column}</span>
      </div>
      <p className="text-[#818CF8] text-xs mb-1">{defect.evidence}</p>
      <p className="font-mono text-xs text-[#FBBF24]">→ {defect.action}</p>
    </div>
  );
}

export default function CleaningAgentReport({ report }: { report: CleanAgentResponse }) {
  if (!report.success) {
    return (
      <div className="border border-red-900 bg-red-950/30 rounded-lg p-4 mb-6">
        <p className="font-mono text-xs text-red-400 mb-1">// DATA QUALITY AGENT ERROR</p>
        <p className="text-red-300 text-sm">{report.error ?? "Data quality agent failed."}</p>
        {(report.defects ?? []).length > 0 && (
          <>
            <p className="font-mono text-[11px] text-red-400/80 tracking-widest mt-3 mb-2">
              DIAGNOSIS COMPLETED BEFORE THE FAILURE
            </p>
            <div className="flex flex-col gap-2">
              {(report.defects ?? []).map((d, i) => (
                <DefectCard key={i} defect={d} />
              ))}
            </div>
          </>
        )}
      </div>
    );
  }

  const defects = report.defects ?? [];

  if (!report.agentUsed) {
    return (
      <div className="border border-border rounded-lg bg-panel p-4 mb-6">
        <p className="font-mono text-xs text-accent">
          ✓ No data quality defects found — rule-based cleaning and the agent&apos;s own audit
          both came back clean.
        </p>
      </div>
    );
  }

  const { before, after, reasoning } = report;

  return (
    <div className="border border-accent/40 rounded-lg bg-panel p-5 mb-6">
      <p className="font-mono text-xs text-accent tracking-widest mb-4">
        // DATA QUALITY AGENT REPORT — {report.filename}
      </p>

      {before && after && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-5">
          <DeltaStat label="NULL VALUES (before agent)" before={before.nullValues} after={after.nullValues} />
          <div className="border border-border rounded-lg bg-panel px-5 py-4">
            <p className="font-mono text-[11px] text-muted tracking-widest mb-1">DEFECTS DIAGNOSED</p>
            <p className="font-mono text-xl font-bold text-accent">{defects.length}</p>
          </div>
        </div>
      )}

      {defects.length > 0 && (
        <>
          <p className="font-mono text-[11px] text-muted tracking-widest mb-2">
            DIAGNOSED DEFECTS
          </p>
          <div className="flex flex-col gap-2 mb-5">
            {defects.map((d, i) => (
              <DefectCard key={i} defect={d} />
            ))}
          </div>
        </>
      )}

      <p className="font-mono text-[11px] text-muted tracking-widest mb-2">AGENT FIX REASONING</p>
      <div className="flex flex-col gap-2 font-mono text-xs">
        {(reasoning ?? []).map((line: string, i: number) => (
          <p key={i} className="text-text border-b border-border/50 pb-2">
            {line.replace(/^-\s*/, "")}
          </p>
        ))}
      </div>
    </div>
  );
}
