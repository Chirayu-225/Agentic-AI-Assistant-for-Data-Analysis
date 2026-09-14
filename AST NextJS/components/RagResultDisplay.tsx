"use client";

interface RagResultDisplayProps {
  kind: "answer" | "insights";
  text?: string;
  error?: string;
}

// Insights come back formatted as "1• finding → implication" per line,
// per rag_engine.py's build_insight_extraction_prompt — parse and style
// that structure instead of dumping it as a flat paragraph. Same
// indigo/amber split as AutoAnalyzeReport.tsx's finding/action cards,
// for one consistent visual language across both insight-generating
// features rather than two different treatments for the same pattern.
function InsightLines({ text }: { text: string }) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);

  return (
    <div className="flex flex-col gap-3">
      {lines.map((line, i) => {
        const match = line.match(/^(\d+)[•.\-:]\s*(.*)/);
        if (!match) {
          return (
            <p key={i} className="text-text text-sm leading-relaxed">
              {line}
            </p>
          );
        }
        const [, rank, body] = match;
        const arrowIndex = body.indexOf("→");
        const finding = arrowIndex === -1 ? body : body.slice(0, arrowIndex).trim();
        const implication = arrowIndex === -1 ? null : body.slice(arrowIndex + 1).trim();

        return (
          <div
            key={i}
            className="flex gap-3 items-start border-l-2 border-[#818CF8]/40 pl-3 border-b border-b-border/50 pb-3"
          >
            <span className="font-mono text-accent font-bold text-sm shrink-0">{rank}</span>
            <div>
              <span className="text-[#818CF8] text-sm leading-relaxed">{finding}</span>
              {implication && (
                <p className="font-mono text-xs text-[#FBBF24] mt-1">→ {implication}</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function RagResultDisplay({ kind, text, error }: RagResultDisplayProps) {
  if (error || !text) {
    return (
      <div className="border border-red-900 bg-red-950/30 rounded-lg p-4">
        <p className="font-mono text-xs text-red-400 mb-1">// ERROR</p>
        <p className="text-red-300 text-sm">{error ?? "No answer returned."}</p>
      </div>
    );
  }

  return (
    <div className="border border-border rounded-lg p-6 bg-panel">
      <p className="font-mono text-sm text-muted mb-4">
        {kind === "insights" ? "// KEY INSIGHTS" : "// ANSWER"}
      </p>
      {kind === "insights" ? (
        <InsightLines text={text} />
      ) : (
        <p className="text-text text-base leading-relaxed whitespace-pre-wrap">{text}</p>
      )}
    </div>
  );
}
