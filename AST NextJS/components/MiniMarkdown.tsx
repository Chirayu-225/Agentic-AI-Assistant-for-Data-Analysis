"use client";

import { parseMiniMarkdown, type Inline } from "@/lib/miniMarkdown";

function Runs({ inline }: { inline: Inline[] }) {
  return (
    <>
      {inline.map((r, i) =>
        r.code ? (
          <code key={i} className="font-mono text-[0.9em] bg-base px-1 rounded">{r.text}</code>
        ) : r.bold ? (
          <strong key={i} className="text-text font-semibold">{r.text}</strong>
        ) : (
          <span key={i}>{r.text}</span>
        )
      )}
    </>
  );
}

/** Renders a model answer's bold, bullets, numbered lists and short
 * headings as real elements (see lib/miniMarkdown.ts). Text only — no raw
 * HTML is ever injected, so answer text can't add markup to the page. */
export default function MiniMarkdown({ text, className = "" }: { text: string; className?: string }) {
  const blocks = parseMiniMarkdown(text);
  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {blocks.map((b, i) => {
        if (b.kind === "heading")
          return <p key={i} className="text-text font-semibold mt-1"><Runs inline={b.inline} /></p>;
        if (b.kind === "paragraph")
          return <p key={i}><Runs inline={b.inline} /></p>;
        const List = b.kind === "bullets" ? "ul" : "ol";
        return (
          <List key={i} className={`${b.kind === "bullets" ? "list-disc" : "list-decimal"} pl-5 flex flex-col gap-1`}>
            {b.items.map((it, j) => <li key={j}><Runs inline={it} /></li>)}
          </List>
        );
      })}
    </div>
  );
}
