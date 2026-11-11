"use client";

import type { FillMissingResponse } from "@/lib/pythonClient";

export default function FillMissingReport({
  report,
  onFillUnknown,
}: {
  report: FillMissingResponse;
  onFillUnknown?: () => void;
}) {
  if (!report.success) {
    return (
      <div className="border border-red-900 bg-red-950/30 rounded-lg p-4 mb-6">
        <p className="font-mono text-xs text-red-400 mb-1">// FILL MISSING VALUES ERROR</p>
        <p className="text-red-300 text-sm">{report.error ?? "Could not fill the missing values."}</p>
      </div>
    );
  }

  const before = report.before?.nullValues ?? 0;
  const after = report.after?.nullValues ?? 0;
  const filled = report.filled ?? [];
  const skipped = report.skipped ?? [];
  const remainingText = report.remainingText ?? 0;

  return (
    <div className="border border-accent/40 rounded-lg bg-panel p-5 mb-6">
      <p className="font-mono text-xs text-accent tracking-widest mb-3">// MISSING VALUES FILLED</p>
      <div className="flex items-baseline gap-2 mb-3">
        <p className="font-mono text-lg text-muted line-through opacity-60">{before.toLocaleString()}</p>
        <span className="text-muted text-sm">→</span>
        <p className={`font-mono text-xl font-bold ${after < before ? "text-accent" : "text-text"}`}>
          {after.toLocaleString()}
        </p>
        <span className="font-mono text-[11px] text-muted tracking-widest">NULL VALUES</span>
      </div>
      {report.message && <p className="font-mono text-xs text-text mb-3">{report.message}</p>}

      {filled.length > 0 && (
        <ul className="font-mono text-xs text-text flex flex-col gap-1 mb-3">
          {filled.map((f, i) => (
            <li key={`${f.column}-${i}`}>
              ✓ {f.column}: {f.count.toLocaleString()} filled{" "}
              {f.method === "inferred"
                ? `from the other rows with the same ${f.from ?? "key"}`
                : f.method === "derived"
                  ? `by exact arithmetic: ${String(f.value)}`
                  : f.method === "median"
                  ? `with the column median (${String(f.value)})`
                  : `with "${String(f.value)}"`}
            </li>
          ))}
        </ul>
      )}

      {skipped.length > 0 && (
        <>
          <p className="font-mono text-[11px] text-muted tracking-widest mb-1">LEFT BLANK</p>
          <ul className="font-mono text-xs text-[#FBBF24] flex flex-col gap-1">
            {skipped.map((s, i) => (
              <li key={`${s.column}-${i}`}>
                ⚠ {s.column}: {s.count.toLocaleString()} blank, {s.reason}
              </li>
            ))}
          </ul>
        </>
      )}

      {onFillUnknown && remainingText > 0 && (
        <button
          onClick={onFillUnknown}
          className="mt-4 px-3 py-1.5 rounded-md border border-accent/40 text-accent font-mono text-xs"
        >
          Label the remaining {remainingText.toLocaleString()} text blanks &quot;Unknown&quot;
        </button>
      )}
    </div>
  );
}
