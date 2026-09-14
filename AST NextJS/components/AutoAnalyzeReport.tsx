"use client";

import { useRef, useState } from "react";
import type { AutoAnalyzeResponse, AutoAnalyzeChart, FollowUpQaPair, FollowUpResponse } from "@/lib/pythonClient";
import { getOrCreateSessionId } from "@/lib/sessionId";
import { Plot, CHART_COLORS, themeTraces } from "@/components/ResultDisplay";

function ChartPanel({
  chart,
  onGraphDiv,
}: {
  chart: AutoAnalyzeChart;
  onGraphDiv: (el: HTMLElement | null) => void;
}) {
  return (
    <div className="border border-border rounded-lg bg-panel p-4">
      <p className="font-mono text-sm text-text mb-1">{chart.title}</p>
      {chart.why && (
        <p className="font-mono text-[11px] text-muted mb-3">supports insight {chart.why}</p>
      )}

      {chart.resultType === "error" || !chart.resultData ? (
        <div className="border border-red-900 bg-red-950/30 rounded-lg p-3">
          <p className="text-red-300 text-xs">{chart.error ?? "This chart could not be generated."}</p>
        </div>
      ) : (
        <Plot
          data={themeTraces((chart.resultData as { data: any[] }).data)}
          layout={{
            ...(chart.resultData as { layout: object }).layout,
            paper_bgcolor: "#11151F",
            plot_bgcolor: "#161B27",
            font: { color: "#DDE1E8", size: 11 },
            colorway: CHART_COLORS,
            margin: { t: 24, l: 40, r: 16, b: 40 },
            xaxis: { ...(chart.resultData as any).layout?.xaxis, gridcolor: "#242B3D", linecolor: "#242B3D" },
            yaxis: { ...(chart.resultData as any).layout?.yaxis, gridcolor: "#242B3D", linecolor: "#242B3D" },
          }}
          style={{ width: "100%", height: "280px" }}
          config={{ displayModeBar: false }}
          // Captures the actual rendered chart DOM node so the PDF
          // export (lib/pdfReport.ts) can turn it into a PNG via
          // Plotly.toImage() — see that file's docstring for why this
          // runs client-side instead of a backend render.
          onInitialized={(_figure: unknown, graphDiv: HTMLElement) => onGraphDiv(graphDiv)}
          onUpdate={(_figure: unknown, graphDiv: HTMLElement) => onGraphDiv(graphDiv)}
        />
      )}
    </div>
  );
}

function FollowUpSection({
  insights,
  fileUrl,
  filename,
  onHistoryChange,
}: {
  insights: { finding: string; action: string }[];
  fileUrl: string;
  filename: string;
  onHistoryChange: (history: FollowUpQaPair[]) => void;
}) {
  const [qaHistory, setQaHistory] = useState<FollowUpQaPair[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  async function handleAsk() {
    const q = question.trim();
    if (!q || asking) return;
    setAsking(true);
    setAskError(null);
    try {
      // Direct fetch to this app's OWN /api route, not the pythonClient.ts
      // helper — that module holds the Python service's URL/secret and is
      // meant to run server-side only, inside the /api route handler
      // (app/api/auto-analyze/follow-up/route.ts), never from a client
      // component like this one.
      const response = await fetch("/api/auto-analyze/follow-up", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fileUrl,
          filename,
          sessionId: getOrCreateSessionId(),
          insights,
          question: q,
          priorQa: qaHistory,
        }),
      });
      const res = (await response.json()) as FollowUpResponse;
      if (res.success && res.answer) {
        const updated = [...qaHistory, { question: q, answer: res.answer! }];
        setQaHistory(updated);
        onHistoryChange(updated);
        setQuestion("");
      } else {
        setAskError(res.error ?? "Could not answer that question.");
      }
    } catch (err) {
      setAskError((err as Error).message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="mb-6">
      <p className="font-mono text-[11px] text-muted tracking-widest mb-2">
        FOLLOW-UP QUESTIONS
      </p>

      {qaHistory.length > 0 && (
        <div className="flex flex-col gap-3 mb-3">
          {qaHistory.map((qa, i) => (
            <div key={i} className="border border-border rounded-lg bg-panel p-3">
              <p className="font-mono text-xs text-text mb-1">Q: {qa.question}</p>
              <p className="text-muted text-sm">{qa.answer}</p>
            </div>
          ))}
        </div>
      )}

      <div className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          disabled={asking}
          placeholder="Ask about an insight or recommended action..."
          className="flex-1 bg-base border border-border rounded-md px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-50"
        />
        <button
          onClick={handleAsk}
          disabled={asking || !question.trim()}
          className="px-4 py-2 rounded-md bg-accent text-base font-mono text-xs font-semibold disabled:opacity-40"
        >
          {asking ? "Asking..." : "Ask"}
        </button>
      </div>
      {askError && <p className="text-red-400 text-xs mt-2">{askError}</p>}
    </div>
  );
}

export default function AutoAnalyzeReport({
  report,
  fileUrl,
  filename,
}: {
  report: AutoAnalyzeResponse;
  fileUrl: string;
  filename: string;
}) {
  const graphDivsRef = useRef<Map<number, HTMLElement>>(new Map());
  const [qaHistory, setQaHistory] = useState<FollowUpQaPair[]>([]);
  const [downloading, setDownloading] = useState(false);

  if (!report.success) {
    return (
      <div className="border border-red-900 bg-red-950/30 rounded-lg p-4 mb-6">
        <p className="font-mono text-xs text-red-400 mb-1">// AUTO-ANALYZE ERROR</p>
        <p className="text-red-300 text-sm">{report.error ?? "Auto-analyze failed."}</p>
      </div>
    );
  }

  const insights = report.insights ?? [];
  const charts = report.charts ?? [];

  async function handleDownloadReport() {
    setDownloading(true);
    try {
      // Dynamically imported, not a top-level import — jsPDF (~200KB+)
      // has no reason to be in the page's initial bundle when this
      // button might never be clicked. This alone moved the page's
      // First Load JS from ~146KB to ~275KB when it was a static
      // import; code-splitting it back out was a real, measured fix,
      // not just tidiness.
      const { generateAutoAnalyzeReport, captureChartImage } = await import("@/lib/pdfReport");
      const chartResults = await Promise.all(
        charts.map(async (chart, i) => {
          const graphDiv = graphDivsRef.current.get(i);
          const imageDataUrl = graphDiv ? await captureChartImage(graphDiv) : null;
          return { title: chart.title, why: chart.why, imageDataUrl };
        })
      );
      generateAutoAnalyzeReport({
        filename: report.filename ?? filename,
        insights,
        charts: chartResults,
        followUpQa: qaHistory,
      });
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="border border-accent/40 rounded-lg bg-panel p-5 mb-6">
      <div className="flex items-center justify-between mb-4">
        <p className="font-mono text-xs text-accent tracking-widest">
          // AUTO-ANALYZE REPORT — {report.filename}
        </p>
        <button
          onClick={handleDownloadReport}
          disabled={downloading}
          className="px-3 py-1.5 rounded-md border border-accent/60 text-accent font-mono text-[11px] font-semibold disabled:opacity-40 shrink-0"
        >
          {downloading ? "Building PDF..." : "Download PDF Report"}
        </button>
      </div>

      <p className="font-mono text-[11px] text-muted tracking-widest mb-2">
        BUSINESS INSIGHTS
      </p>
      <div className="flex flex-col gap-3 mb-6">
        {insights.map((ins, i) => (
          <div key={i} className="border-l-2 border-[#818CF8]/40 pl-3 border-b border-b-border/50 pb-3">
            <p className="text-[#818CF8] text-sm">{ins.finding}</p>
            <p className="font-mono text-xs text-[#FBBF24] mt-1">→ {ins.action}</p>
          </div>
        ))}
        {insights.length === 0 && (
          <p className="text-muted text-sm">No insights were generated for this dataset.</p>
        )}
      </div>

      {insights.length > 0 && (
        <FollowUpSection
          insights={insights}
          fileUrl={fileUrl}
          filename={filename}
          onHistoryChange={setQaHistory}
        />
      )}

      {charts.length > 0 && (
        <>
          <p className="font-mono text-[11px] text-muted tracking-widest mb-2">
            VISUALIZATIONS
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {charts.map((chart, i) => (
              <ChartPanel
                key={i}
                chart={chart}
                onGraphDiv={(el) => {
                  if (el) graphDivsRef.current.set(i, el);
                  else graphDivsRef.current.delete(i);
                }}
              />
            ))}
          </div>
        </>
      )}
      {charts.length === 0 && report.chartsError && (
        <div className="border border-yellow-900 bg-yellow-950/20 rounded-lg p-3">
          <p className="font-mono text-[11px] text-yellow-500 tracking-widest mb-1">
            VISUALIZATIONS UNAVAILABLE
          </p>
          <p className="text-yellow-200/80 text-xs">{report.chartsError} Insights above are still valid — try running Auto-Analyze again for charts.</p>
        </div>
      )}
    </div>
  );
}
