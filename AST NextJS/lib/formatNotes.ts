/**
 * lib/formatNotes.ts — turns /data-checks cards into the short notes shown
 * under a result. A column's note appears only if that column is actually used
 * in the displayed code: in quotes ("col" / 'col') or as df.col. Matching is
 * case-sensitive and the column name is regex-escaped. One line per flagged
 * column (its format and encoding messages joined), at most MAX_NOTES lines.
 */

import type { DataCheckCard } from "@/lib/pythonClient";

export const MAX_NOTES = 2;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function columnUsedInCode(column: string, code: string): boolean {
  if (!column || !code) return false;
  const c = escapeRegExp(column);
  const quoted = new RegExp(`(["'])${c}\\1`);
  const attr = new RegExp(`\\bdf\\.${c}(?![A-Za-z0-9_])`);
  return quoted.test(code) || attr.test(code);
}

export const MAX_FORMAT_ROWS = 5;

export interface FormatRow {
  signature: string;
  count: number;
  example: string;
}

export interface FormatNote {
  text: string;
  formats: FormatRow[]; // up to MAX_FORMAT_ROWS; empty for an encoding-only column
  moreFormats: number; // signatures beyond the rows shown
}

function isFormatCard(card: DataCheckCard): boolean {
  return card.reason === "format_drift" || card.reason === "format_variance";
}

// Rows come from the column record data_checks.py returns: top_signatures[{signature, count, share, example}]
// and n_signatures. Cards carry column_index, which is the record's position in columns[].
function formatRows(record: unknown): { rows: FormatRow[]; more: number } {
  const rec = record as { top_signatures?: unknown; n_signatures?: unknown } | null | undefined;
  const top = Array.isArray(rec?.top_signatures) ? (rec!.top_signatures as Record<string, unknown>[]) : [];
  const rows: FormatRow[] = [];
  for (const t of top) {
    if (!t || typeof t.signature !== "string" || typeof t.count !== "number") continue;
    rows.push({ signature: t.signature, count: t.count, example: String(t.example ?? "") });
    if (rows.length === MAX_FORMAT_ROWS) break;
  }
  const total = typeof rec?.n_signatures === "number" ? rec.n_signatures : rows.length;
  return { rows, more: Math.max(0, total - rows.length) };
}

export function formatNoteItems(
  cards: DataCheckCard[] | undefined | null,
  code: string | undefined | null,
  columns?: Record<string, unknown>[] | null,
): FormatNote[] {
  if (!Array.isArray(cards) || !code) return [];
  const byColumn = new Map<string, { messages: string[]; formatIndex: number | null }>(); // insertion order = card order (already ranked by the server)
  for (const card of cards) {
    if (!card || typeof card.message !== "string" || card.message === "") continue;
    const column = String(card.column);
    if (!columnUsedInCode(column, code)) continue;
    const entry = byColumn.get(column) ?? { messages: [], formatIndex: null };
    if (!entry.messages.includes(card.message)) entry.messages.push(card.message);
    if (isFormatCard(card) && entry.formatIndex === null) entry.formatIndex = card.column_index;
    byColumn.set(column, entry);
  }
  return Array.from(byColumn.values())
    .map(({ messages, formatIndex }) => {
      const rec = formatIndex !== null && Array.isArray(columns) ? columns[formatIndex] : undefined;
      const { rows, more } = rec ? formatRows(rec) : { rows: [], more: 0 };
      return { text: messages.join(" "), formats: rows, moreFormats: rows.length ? more : 0 };
    })
    .slice(0, MAX_NOTES);
}

export function formatNotes(cards: DataCheckCard[] | undefined | null, code: string | undefined | null): string[] {
  return formatNoteItems(cards, code).map((n) => n.text);
}
