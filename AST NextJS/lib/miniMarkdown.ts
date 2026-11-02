/**
 * lib/miniMarkdown.ts — the small subset of Markdown that model answers use
 * (bold, bullet and numbered lists, short headings, inline code), parsed
 * into blocks. Rendered by components/MiniMarkdown.tsx as real React
 * elements — never as raw HTML — and flattened to plain text for the PDF.
 *
 * Why: follow-up and document answers arrive as Markdown. Printed as one
 * <p>, they showed literal "*" and "**" in a single dense paragraph.
 */

export type Inline = { text: string; bold?: boolean; code?: boolean };
export type Block =
  | { kind: "heading"; inline: Inline[] }
  | { kind: "paragraph"; inline: Inline[] }
  | { kind: "bullets"; items: Inline[][] }
  | { kind: "numbered"; items: Inline[][] };

/** "**bold** and `code`" -> inline runs. */
export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ text: text.slice(last, at) });
    const tok = m[0];
    if (tok.startsWith("**")) out.push({ text: tok.slice(2, -2), bold: true });
    else out.push({ text: tok.slice(1, -1), code: true });
    last = at + tok.length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  // stray single asterisks used for emphasis (*why*) -> plain text
  return out.map((r) => (r.code ? r : { ...r, text: r.text.replace(/(^|\s)\*(\S[^*]*?)\*(?=\s|[.,;:!?)]|$)/g, "$1$2") }));
}

export function parseMiniMarkdown(md: string): Block[] {
  const blocks: Block[] = [];
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  let para: string[] = [];
  const flushPara = () => {
    if (para.length) blocks.push({ kind: "paragraph", inline: parseInline(para.join(" ")) });
    para = [];
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushPara(); continue; }
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    const heading = line.match(/^#{1,4}\s+(.*)$/) || line.match(/^\*\*([^*]+)\*\*:?$/);
    if (bullet || numbered) {
      flushPara();
      const kind = bullet ? "bullets" : "numbered";
      const item = parseInline((bullet ?? numbered)![1]);
      const prev = blocks.at(-1);
      if (prev && prev.kind === kind) prev.items.push(item);
      else blocks.push({ kind, items: [item] } as Block);
    } else if (heading) {
      flushPara();
      blocks.push({ kind: "heading", inline: parseInline(heading[1]) });
    } else {
      para.push(line);
    }
  }
  flushPara();
  return blocks;
}

/** Plain text for places that can't render formatting (the PDF report). */
export function miniMarkdownToPlain(md: string): string {
  const flat = (inline: Inline[]) => inline.map((r) => r.text).join("");
  return parseMiniMarkdown(md)
    .map((b) => {
      if (b.kind === "bullets") return b.items.map((it) => `• ${flat(it)}`).join("\n");
      if (b.kind === "numbered") return b.items.map((it, i) => `${i + 1}. ${flat(it)}`).join("\n");
      return flat(b.inline);
    })
    .join("\n");
}
