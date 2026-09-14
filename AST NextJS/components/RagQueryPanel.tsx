"use client";

import { useState } from "react";

const EXAMPLE_QUERIES = [
  "What is this document about?",
  "Summarize the key points",
  "What risks or concerns are mentioned?",
  "What are the main recommendations?",
];

export default function RagQueryPanel({
  disabled,
  onSubmit,
  onGetInsights,
}: {
  disabled: boolean;
  onSubmit: (query: string) => void;
  onGetInsights: () => void;
}) {
  const [query, setQuery] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || disabled) return;
    onSubmit(query.trim());
  }

  return (
    <div className="border border-border rounded-lg p-6 bg-panel">
      <p className="font-mono text-sm text-muted mb-3">// ASK THE DOCUMENT</p>
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled}
          placeholder={
            disabled ? "Upload a document first..." : "Ask a question about the document..."
          }
          className="flex-1 bg-base border border-border rounded-md px-4 py-2
                     text-text placeholder:text-muted focus:outline-none
                     focus:border-accent disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={disabled || !query.trim()}
          className="px-5 py-2 rounded-md bg-accent text-base font-mono font-semibold
                     disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Ask
        </button>
      </form>

      <div className="mt-3">
        <button
          type="button"
          disabled={disabled}
          onClick={onGetInsights}
          className="text-sm font-mono text-accent/80 hover:text-accent
                     disabled:opacity-40 disabled:cursor-not-allowed underline underline-offset-4"
        >
          Or get auto-extracted key insights →
        </button>
      </div>

      <div className="mt-4">
        <p className="font-mono text-xs text-muted mb-2">// EXAMPLE QUESTIONS</p>
        <div className="flex flex-col gap-1">
          {EXAMPLE_QUERIES.map((ex) => (
            <button
              key={ex}
              type="button"
              disabled={disabled}
              onClick={() => setQuery(ex)}
              className="text-left text-sm font-mono text-accent/80 hover:text-accent
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              &gt; {ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
