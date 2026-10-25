"use client";

import { useState } from "react";

/**
 * "How this was computed" — the product's core promise is that answers
 * come from real computation, so the computation has to be inspectable.
 * Collapsed by default so it doesn't crowd the answer, one click away
 * for anyone who wants to check it.
 */
export default function ShowWork({
  code,
  language = "python",
  evidence,
  question,
  label = "How this was computed",
}: {
  code?: string | null;
  language?: "python" | "sql";
  evidence?: string | null;
  question?: string | null;
  label?: string;
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  if (!code && !evidence) return null;

  async function copy() {
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — ignore */
    }
  }

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="font-mono text-[11px] text-muted hover:text-accent tracking-wide"
      >
        {open ? "▾" : "▸"} {label}
      </button>
      {open && (
        <div className="mt-2 border border-border rounded-md bg-base p-3 flex flex-col gap-3">
          {question && (
            <p className="font-mono text-[11px] text-muted">
              QUESTION ASKED OF THE DATA: <span className="text-text">{question}</span>
            </p>
          )}
          {code && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <p className="font-mono text-[10px] text-muted tracking-widest">
                  {language === "sql" ? "SQL RUN IN SANDBOX" : "CODE RUN IN SANDBOX"}
                </p>
                <button onClick={copy} className="font-mono text-[10px] text-muted hover:text-accent">
                  {copied ? "copied" : "copy"}
                </button>
              </div>
              <pre className="text-xs font-mono text-text overflow-x-auto whitespace-pre">{code}</pre>
            </div>
          )}
          {evidence && (
            <div>
              <p className="font-mono text-[10px] text-muted tracking-widest mb-1">COMPUTED RESULT</p>
              <pre className="text-xs font-mono text-text overflow-x-auto whitespace-pre">{evidence}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
