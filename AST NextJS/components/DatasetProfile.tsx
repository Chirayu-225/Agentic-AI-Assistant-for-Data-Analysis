"use client";

import type { ProfileResponse } from "@/lib/pythonClient";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-border rounded-lg bg-panel px-5 py-4">
      <p className="font-mono text-[11px] text-muted tracking-widest mb-1">{label}</p>
      <p className="font-mono text-2xl font-bold text-accent">{value}</p>
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
