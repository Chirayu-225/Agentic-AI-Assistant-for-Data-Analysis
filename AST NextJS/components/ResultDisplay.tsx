"use client";

import dynamic from "next/dynamic";
import type { AnalyzeResponse } from "@/lib/pythonClient";

// Plotly touches `window` at import time, so it can only load client-side —
// dynamic() with ssr:false is required here, not just a style preference.
// Exported so other chart-rendering components (e.g. AutoAnalyzeReport)
// reuse this same client-only load instead of re-declaring it.
export const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// Brand palette, teal-first to match the accent color used everywhere else.
export const CHART_COLORS = ["#2DD4BF", "#818CF8", "#F472B6", "#FBBF24", "#34D399", "#60A5FA"];

// The Python side sends raw fig.to_json() output through untouched — a
// trace with no explicit marker/line color renders solid black in
// plotly.js when only `data` + a partial `layout` are supplied without a
// `layout.template` (this is what caused the black bars). Fixing it at
// render time here, rather than on the Python side, keeps the chart data
// itself provider-agnostic and puts all frontend theming in one place.
export function themeTraces(traces: any[]): any[] {
  return traces.map((trace, i) => {
    const color = CHART_COLORS[i % CHART_COLORS.length];
    const themed = { ...trace };

    if (themed.type === "bar" || themed.type === "histogram") {
      themed.marker = { ...themed.marker, color: themed.marker?.color ?? color };
    } else if (themed.type === "scatter" || themed.type === "scatterpolar") {
      themed.marker = { ...themed.marker, color: themed.marker?.color ?? color };
      themed.line = { ...themed.line, color: themed.line?.color ?? color };
    } else if (themed.type === "pie") {
      // Pie charts want one color per slice, not per trace.
      const n = (themed.labels?.length as number) ?? CHART_COLORS.length;
      themed.marker = {
        ...themed.marker,
        colors: themed.marker?.colors ?? Array.from({ length: n }, (_, j) => CHART_COLORS[j % CHART_COLORS.length]),
      };
    } else {
      themed.marker = { ...themed.marker, color: themed.marker?.color ?? color };
    }

    return themed;
  });
}

export default function ResultDisplay({ result }: { result: AnalyzeResponse }) {
  if (!result.success || result.resultType === "error") {
    return (
      <div className="border border-red-900 bg-red-950/30 rounded-lg p-4">
        <p className="font-mono text-xs text-red-400 mb-1">// ERROR</p>
        <p className="text-red-300 text-sm">{result.error ?? "Unknown error"}</p>
      </div>
    );
  }

  return (
    <div className="border border-border rounded-lg p-6 bg-panel">
      <div className="flex items-center justify-between mb-4">
        <p className="font-mono text-sm text-muted">// RESULT</p>
        {result.confidence && (
          <span className="font-mono text-xs text-accent">
            CONFIDENCE: {result.confidence}
          </span>
        )}
      </div>

      {result.resultType === "scalar" && (
        <p className="text-2xl font-mono text-text">{String(result.resultData)}</p>
      )}

      {result.resultType === "table" && <TableResult data={result.resultData} />}

      {result.resultType === "chart" && (
        <Plot
          data={themeTraces((result.resultData as { data: any[] }).data)}
          layout={{
            ...(result.resultData as { layout: object }).layout,
            paper_bgcolor: "#11151F",
            plot_bgcolor: "#161B27",
            font: { color: "#DDE1E8" },
            colorway: CHART_COLORS,
            xaxis: { ...(result.resultData as any).layout?.xaxis, gridcolor: "#242B3D", linecolor: "#242B3D" },
            yaxis: { ...(result.resultData as any).layout?.yaxis, gridcolor: "#242B3D", linecolor: "#242B3D" },
          }}
          style={{ width: "100%" }}
          config={{ displayModeBar: false }}
        />
      )}
    </div>
  );
}

export function TableResult({ data }: { data: unknown }) {
  const rows = Array.isArray(data) ? (data as Record<string, unknown>[]) : [];
  if (rows.length === 0) return <p className="text-muted text-sm">No rows returned.</p>;

  const columns = Object.keys(rows[0]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm font-mono">
        <thead>
          <tr className="border-b border-border text-muted">
            {columns.map((col) => (
              <th key={col} className="text-left py-2 pr-4">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-border/50">
              {columns.map((col) => (
                <td key={col} className="py-2 pr-4 text-text">
                  {String(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
