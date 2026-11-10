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

export function formatNotes(cards: DataCheckCard[] | undefined | null, code: string | undefined | null): string[] {
  if (!Array.isArray(cards) || !code) return [];
  const byColumn = new Map<string, string[]>(); // insertion order = card order (already ranked by the server)
  for (const card of cards) {
    if (!card || typeof card.message !== "string" || card.message === "") continue;
    const column = String(card.column);
    if (!columnUsedInCode(column, code)) continue;
    const messages = byColumn.get(column) ?? [];
    if (!messages.includes(card.message)) messages.push(card.message);
    byColumn.set(column, messages);
  }
  return Array.from(byColumn.values())
    .map((messages) => messages.join(" "))
    .slice(0, MAX_NOTES);
}
