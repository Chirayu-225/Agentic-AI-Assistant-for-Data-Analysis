"use client";

import type { ProfileResponse, PiiReport } from "@/lib/pythonClient";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-border rounded-lg bg-panel px-5 py-4">
      <p className="font-mono text-[11px] text-muted tracking-widest mb-1">{label}</p>
      <p className="font-mono text-2xl font-bold text-accent">{value}</p>
    </div>
  );
}

const SEVERITY_STYLE: Record<string, string> = {
  high: "text-red-400 border-red-900/60",
  medium: "text-amber-400 border-amber-900/60",
  low: "text-muted border-border",
};

/**
 * Shown right after upload, BEFORE any AI call: which columns look like
 * personal data and exactly how each will be treated. The scan itself runs
 * on the Python side (pii_redaction.scan_dataframe) and changes nothing —
 * it exists so the user can see and trust what will leave the server.
 */
function PiiNotice({ report }: { report: PiiReport }) {
  if (report.risk === "none" || report.flaggedColumns.length === 0) return null;

  return (
    <div className="border border-amber-900/60 bg-amber-950/20 rounded-lg p-4">
      <p className="font-mono text-xs text-amber-400 tracking-widest mb-2">// PERSONAL DATA CHECK</p>
      <p className="text-sm text-text mb-3">{report.summary}</p>
      <div className="flex flex-col divide-y divide-border/50">
        {report.flaggedColumns.map((col) => {
          const found = Object.entries(col.categories).map(
            ([label, info]) => `${label.replace(/_/g, " ").toLowerCase()} (${info.pct}%)`
          );
          return (
            <div key={col.column} className="flex flex-wrap items-center justify-between gap-2 py-1.5 font-mono text-xs">
              <span className="text-accent">{col.column}</span>
              <span className="flex flex-wrap items-center gap-2">
                {found.length > 0 && <span className="text-muted">{found.join(", ")}</span>}
                {col.nameHint === "personal" && <span className="text-muted">name looks personal</span>}
                {col.nameHint === "review" && <span className="text-muted">generic name — review</span>}
                <span className={`border rounded px-1.5 py-0.5 ${SEVERITY_STYLE[col.severity]}`}>
                  {col.severity}
                </span>
                {col.withheldFromLlm && (
                  <span className="border border-border rounded px-1.5 py-0.5 text-muted">withheld from AI</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-muted text-[11px] mt-3">
        Your data is never altered. Emails, phone numbers, card and ID numbers are masked in everything sent to the
        AI model, and withheld columns are not sent at all.
      </p>
    </div>
  );
}

export default function DatasetProfile({ profile }: { profile: ProfileResponse }) {
  if (!profile.success) {
    return (
      <div className="border border-red-900 bg-red-950/30 rounded-lg p-4 mb-6">
        <p className="font-mono text-xs text-red-400 mb-1">// PROFILING ERROR</p>
        <p className="text-red-300 text-sm">{profile.error ?? "Could not profile file."}</p>
      </div>
    );
  }

  const preview = profile.preview ?? [];
  const columns = preview.length > 0 ? Object.keys(preview[0]) : [];

  return (
    <div className="flex flex-col gap-4 mb-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard label="TOTAL ROWS" value={profile.totalRows?.toLocaleString() ?? "—"} />
        <StatCard label="TOTAL COLUMNS" value={profile.totalColumns ?? "—"} />
        <StatCard label="NUMERIC COLS" value={profile.numericCols ?? "—"} />
        <StatCard label="NULL VALUES" value={profile.nullValues?.toLocaleString() ?? "—"} />
      </div>

      {profile.piiReport && <PiiNotice report={profile.piiReport} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="border border-border rounded-lg bg-panel p-4 overflow-hidden">
          <p className="font-mono text-xs text-accent tracking-widest mb-3">// DATA STREAM PREVIEW</p>
          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-border text-muted sticky top-0 bg-panel">
                  {columns.map((col: string) => (
                    <th key={col} className="text-left py-1.5 pr-4 whitespace-nowrap">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((row: Record<string, unknown>, i: number) => (
                  <tr key={i} className="border-b border-border/50">
                    {columns.map((col: string) => (
                      <td key={col} className="py-1.5 pr-4 text-text whitespace-nowrap">
                        {String(row[col] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-muted text-[11px] mt-3">
            Showing {profile.previewCount ?? 0} / {profile.totalRows?.toLocaleString() ?? 0} records — {profile.filename}
          </p>
        </div>

        <div className="border border-border rounded-lg bg-panel p-4">
          <p className="font-mono text-xs text-accent tracking-widest mb-3">// SCHEMA MANIFEST</p>
          <div className="max-h-80 overflow-y-auto flex flex-col divide-y divide-border/50">
            {(profile.schema ?? []).map((col: { column: string; dtype: string; nulls: number }) => (
              <div key={col.column} className="flex items-center justify-between py-1.5 font-mono text-xs">
                <span className="text-accent">{col.column}</span>
                <span className="flex items-center gap-3">
                  <span className="text-muted">{col.dtype}</span>
                  <span className={col.nulls > 0 ? "text-amber-400" : "text-muted"}>
                    [{col.nulls} null]
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
