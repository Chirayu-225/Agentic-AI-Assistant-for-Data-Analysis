"use client";

import { useState } from "react";

const EXAMPLE_QUERIES = [
  "What are the top 5 rows by sales?",
  "Show correlation between price and volume",
  "Group by category and sum profit",
  "What columns have null values?",
];

export default function QueryPanel({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (query: string) => void;
}) {
  const [query, setQuery] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || disabled) return;
    onSubmit(query.trim());
  }

  return (
    <div className="border border-border rounded-lg p-6 bg-panel">
      <p className="font-mono text-sm text-muted mb-3">// QUERY</p>
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled}
          placeholder={
            disabled ? "Upload a CSV first..." : "Ask a question about your data..."
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
          Run
        </button>
      </form>

      <div className="mt-4">
        <p className="font-mono text-xs text-muted mb-2">// QUERY EXAMPLES</p>
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
