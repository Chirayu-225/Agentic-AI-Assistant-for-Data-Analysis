"use client";

import type { CleanAgentResponse, DataQualityDefect, ScaleReviewItem } from "@/lib/pythonClient";

// The user's answer to one question from the agent: fill that column's blanks with "Unknown", or leave them empty.
export type FillChoice = "unknown" | "empty" | "working" | "failed";
export type ScaleChoice = "fixed" | "kept" | "working" | "failed";

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
        <span className="font-mono text-[10px] text-muted ml-auto">model-suggested, unverified</span>
      </div>
      <p className="text-[#818CF8] text-xs mb-1">{defect.evidence}</p>
      <p className="font-mono text-xs text-[#FBBF24]">→ {defect.action}</p>
    </div>
  );
}

function AgentFillSection({
  fill,
  choices,
  onChoose,
}: {
  fill: CleanAgentResponse["fill"];
  choices?: Record<string, FillChoice>;
  onChoose?: (column: string, choice: "unknown" | "empty") => void;
}) {
  if (!fill) return null;
  const filled = fill.filled ?? [];
  const skipped = fill.skipped ?? [];
  const questions = skipped.filter((s) => s.kind === "no_evidence" && !s.numeric);
  // number columns cannot be "Unknown": no question, just an honest note that they stay empty
  const never = skipped.filter((s) => s.kind !== "no_evidence" || s.numeric);
  if (filled.length === 0 && questions.length === 0 && never.length === 0) return null;

  return (
    <div className="mb-5">
      {filled.length > 0 && (
        <>
          <p className="font-mono text-[11px] text-muted tracking-widest mb-2">
            FILLED BY THE AGENT (no AI, nothing invented)
          </p>
          <ul className="font-mono text-xs text-text flex flex-col gap-1 mb-4">
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
        </>
      )}

      {questions.length > 0 && (
        <>
          <p className="font-mono text-[11px] text-[#FBBF24] tracking-widest mb-2">
            QUESTIONS FOR YOU — NOTHING IN THE FILE TELLS ME THESE VALUES
          </p>
          <div className="flex flex-col gap-2 mb-4">
            {questions.map((q) => {
              const choice = choices?.[q.column];
              return (
                <div key={q.column} className="border border-[#FBBF24]/40 rounded-lg bg-[#FBBF24]/5 p-3">
                  <p className="font-mono text-xs text-text mb-2">
                    {q.column}: {q.count.toLocaleString()} blank. Fill them with &quot;Unknown&quot;, or leave them
                    empty?
                  </p>
                  {choice === "unknown" && (
                    <p className="font-mono text-xs text-accent">
                      ✓ filled with &quot;Unknown&quot;, used for your questions
                    </p>
                  )}
                  {choice === "empty" && <p className="font-mono text-xs text-muted">✓ left empty</p>}
                  {choice === "working" && (
                    <p className="font-mono text-xs text-accent animate-pulse">▸ filling...</p>
                  )}
                  {(choice === undefined || choice === "failed") && onChoose && (
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        onClick={() => onChoose(q.column, "unknown")}
                        className="px-3 py-1.5 rounded-md bg-accent text-base font-mono text-xs font-semibold"
                      >
                        Fill with &quot;Unknown&quot;
                      </button>
                      <button
                        onClick={() => onChoose(q.column, "empty")}
                        className="px-3 py-1.5 rounded-md border border-border text-text font-mono text-xs"
                      >
                        Leave empty
                      </button>
                      {choice === "failed" && (
                        <span className="font-mono text-xs text-red-400">could not fill, try again</span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}

      {never.length > 0 && (
        <>
          <p className="font-mono text-[11px] text-muted tracking-widest mb-1">LEFT BLANK ON PURPOSE</p>
          <ul className="font-mono text-xs text-muted flex flex-col gap-1">
            {never.map((s, i) => (
              <li key={`${s.column}-${i}`}>
                ⚠ {s.column}: {s.count.toLocaleString()} blank, {s.reason}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

// What the deterministic repairs did (value_validators.py). Every line says what in the table justified it, with examples,
// so nothing is changed silently. Blanks are never filled back in: these run after the fill step.
function AgentRepairSection({ validators }: { validators: CleanAgentResponse["validators"] }) {
  const entries = validators?.entries ?? [];
  if (!validators || entries.length === 0) return null;
  return (
    <div className="mb-5">
      <p className="font-mono text-[11px] text-muted tracking-widest mb-2">
        REPAIRED BY THE AGENT (no AI — each repair needed evidence in the table)
      </p>
      <ul className="font-mono text-xs text-text flex flex-col gap-3">
        {entries.map((e, i) => (
          <li key={`${e.column}-${e.rule}-${i}`}>
            <span className="text-accent">✓</span> <span className="font-semibold">{e.column}</span>:{" "}
            {e.count.toLocaleString()} {e.action}
            <p className="text-muted mt-0.5">{e.evidence}</p>
            {e.examples.length > 0 && (
              <p className="text-muted mt-0.5">
                e.g.{" "}
                {e.examples
                  .map((x) => `${String(x.before)} → ${x.after === null ? "left blank" : String(x.after)}`)
                  .join("; ")}
              </p>
            )}
          </li>
        ))}
      </ul>
      {validators.kept === false && (
        <p className="font-mono text-[11px] text-[#FBBF24] mt-2">
          These repairs could not be kept for your questions.
        </p>
      )}
    </div>
  );
}

function ScaleReviewSection({
  items,
  choices,
  onChoose,
}: {
  items?: ScaleReviewItem[];
  choices?: Record<string, ScaleChoice>;
  onChoose?: (column: string, choice: "fix" | "keep") => void;
}) {
  if (!items || items.length === 0) return null;
  return (
    <>
      <p className="font-mono text-[11px] text-[#FBBF24] tracking-widest mb-2">
        PLEASE CHECK — THESE NUMBERS LOOK 1000× TOO LARGE
      </p>
      <div className="flex flex-col gap-2 mb-4">
        {items.map((it) => {
          const choice = choices?.[it.column];
          const ex = it.examples
            .map((e) => `${e.value.toLocaleString()}${e.row ? ` (row ${e.row})` : ""}`)
            .join(", ");
          return (
            <div key={it.column} className="border border-[#FBBF24]/40 rounded-lg bg-[#FBBF24]/5 p-3">
              <p className="font-mono text-xs text-text mb-1">
                {it.column}: {it.count.toLocaleString()} value{it.count === 1 ? "" : "s"} are exact multiples of 1000
                {it.typical !== null ? ` while most are around ${it.typical.toLocaleString()}` : ""}, e.g. {ex}.
                A typo, or a real bulk order? Nothing in the file says.
              </p>
              {choice === "fixed" && <p className="font-mono text-xs text-accent">✓ divided by 1000, used for your questions</p>}
              {choice === "kept" && <p className="font-mono text-xs text-muted">✓ kept as they are</p>}
              {choice === "working" && <p className="font-mono text-xs text-accent animate-pulse">▸ applying...</p>}
              {(choice === undefined || choice === "failed") && onChoose && (
                <div className="flex flex-wrap items-center gap-2 mt-2">
                  <button
                    onClick={() => onChoose(it.column, "fix")}
                    className="px-3 py-1.5 rounded-md bg-accent text-base font-mono text-xs font-semibold"
                  >
                    Divide by 1000
                  </button>
                  <button
                    onClick={() => onChoose(it.column, "keep")}
                    className="px-3 py-1.5 rounded-md border border-border text-text font-mono text-xs"
                  >
                    Keep as is
                  </button>
                  {choice === "failed" && <span className="font-mono text-xs text-red-400">could not apply, try again</span>}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}

export default function CleaningAgentReport({
  report,
  choices,
  onChoose,
  scaleChoices,
  onScaleChoose,
}: {
  report: CleanAgentResponse;
  choices?: Record<string, FillChoice>;
  onChoose?: (column: string, choice: "unknown" | "empty") => void;
  scaleChoices?: Record<string, ScaleChoice>;
  onScaleChoose?: (column: string, choice: "fix" | "keep") => void;
}) {
  if (!report.success) {
    return (
      <>
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
      <AgentRepairSection validators={report.validators} />
      <AgentFillSection fill={report.fill} choices={choices} onChoose={onChoose} />
      </>
    );
  }

  const defects = report.defects ?? [];

  if (!report.agentUsed) {
    return (
      <>
        <div className="border border-border rounded-lg bg-panel p-4 mb-6">
          {report.message && <p className="font-mono text-xs text-text mb-2">{report.message}</p>}
          <p className="font-mono text-xs text-accent">
            ✓ No data quality defects found — rule-based cleaning and the agent&apos;s own audit
            both came back clean.
          </p>
        </div>
        <AgentRepairSection validators={report.validators} />
        <AgentFillSection fill={report.fill} choices={choices} onChoose={onChoose} />
      </>
    );
  }

  const { before, after, reasoning } = report;
  const reverted = report.reverted ?? [];
  // Green: safe changes were applied and nothing was reverted. Amber: nothing was changed, or something was reverted.
  const filledSomething = (report.fill?.totalFilled ?? 0) > 0 || (report.validators?.entries?.length ?? 0) > 0;
  const bannerIsGreen = (report.applied === true || filledSomething) && reverted.length === 0;

  return (
    <div className="border border-accent/40 rounded-lg bg-panel p-5 mb-6">
      <p className="font-mono text-xs text-accent tracking-widest mb-4">
        // DATA QUALITY AGENT REPORT — {report.filename}
      </p>

      {report.message && (
        <div
          className={`rounded-lg border px-4 py-3 mb-5 ${
            bannerIsGreen
              ? "border-accent/40 bg-accent/10 text-accent"
              : "border-[#FBBF24]/40 bg-[#FBBF24]/10 text-[#FBBF24]"
          }`}
        >
          <p className="text-sm">{report.message}</p>
          {reverted.length > 0 && (
            <ul className="mt-2 list-disc pl-5 font-mono text-xs space-y-0.5">
              {reverted.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {(report.newColumns ?? []).length > 0 && (
        <p className="font-mono text-xs text-muted -mt-3 mb-5">
          {(report.newColumns ?? []).length} flag column{(report.newColumns ?? []).length === 1 ? "" : "s"} added
          {": "}
          {(report.newColumns ?? []).join(", ")}
        </p>
      )}

      {before && after && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-5">
          <DeltaStat label="NULL VALUES (before agent)" before={before.nullValues} after={after.nullValues} />
          <div className="border border-border rounded-lg bg-panel px-5 py-4">
            <p className="font-mono text-[11px] text-muted tracking-widest mb-1">DEFECTS DIAGNOSED</p>
            <p className="font-mono text-xl font-bold text-accent">{defects.length}</p>
          </div>
        </div>
      )}

      <AgentRepairSection validators={report.validators} />
      <AgentFillSection fill={report.fill} choices={choices} onChoose={onChoose} />
      <ScaleReviewSection items={report.scaleReview} choices={scaleChoices} onChoose={onScaleChoose} />

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
        {(reasoning ?? []).map((line: string, i: number) => {
          const text = line.replace(/^-\s*/, "");
          // "proposal only" lines are display text: something a person should decide. They are never executed.
          const isProposal = /proposal only\s*:/i.test(text);
          return (
            <p
              key={i}
              className={`border-b border-border/50 pb-2 ${isProposal ? "text-muted italic" : "text-text"}`}
            >
              {isProposal && (
                <span className="not-italic text-[10px] tracking-widest mr-2 px-1.5 py-0.5 rounded bg-border/40">
                  PROPOSAL ONLY
                </span>
              )}
              {text}
            </p>
          );
        })}
      </div>
    </div>
  );
}
