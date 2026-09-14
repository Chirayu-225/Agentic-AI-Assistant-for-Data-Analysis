/**
 * lib/pdfReport.ts — builds the downloadable Auto-Analyze PDF report,
 * entirely client-side.
 *
 * WHY CLIENT-SIDE, NOT A BACKEND ENDPOINT
 *   The obvious-looking alternative — render each Plotly chart to PNG on
 *   the FastAPI backend (via Plotly's kaleido export) and assemble the
 *   PDF there with reportlab — was tried first and dropped for a
 *   concrete, tested reason: kaleido v1+ requires a real headless
 *   Chrome binary to be installed on the server. That's a heavy,
 *   environment-fragile dependency for a Python backend on a
 *   constrained host (Render free tier and similar), and downgrading to
 *   kaleido's old bundled-Chromium version (0.2.x) isn't viable either
 *   — the plotly.js Python package version already in use has dropped
 *   support for it.
 *
 *   The charts are already rendered live in the browser (react-plotly.js
 *   in ResultDisplay.tsx / AutoAnalyzeReport.tsx) — Plotly.js's own
 *   `Plotly.toImage()` can export any already-rendered chart straight to
 *   a PNG data URL with zero server round-trip and zero new backend
 *   dependency. jsPDF then assembles those images with the insight/
 *   action text into the final PDF, all in the browser.
 *
 * TWO REAL BUGS FIXED HERE (found from an actual generated PDF, not
 * hypothetically):
 *
 *   1. GARBLED TEXT ("16 /% higher" instead of "16% higher"). Root
 *      cause, confirmed by reproducing it with a real render: LLM
 *      output routinely contains a NARROW NO-BREAK SPACE (U+202F)
 *      before a "%" sign (a common typographic convention in some
 *      training data), or an em dash, curly quote, or similar. jsPDF's
 *      built-in core fonts only support WinAnsiEncoding (code points
 *      0-255) — anything outside that range doesn't just fail to
 *      render as itself, it corrupts jsPDF's encoding for the ENTIRE
 *      string it's part of (confirmed: extracting the broken PDF's
 *      text showed a `(cid:0)` marker between every character in the
 *      line, not just at the bad character). sanitizeForPdf() below
 *      normalizes the known common cases and strips anything else
 *      outside the safe range, so one stray character from the model
 *      can't corrupt a whole line ever again.
 *
 *   2. TRUNCATED TEXT (a word cut off mid-way, e.g. "invento" instead
 *      of "inventory"). Root cause: the previous version called
 *      ensureSpace() with a fixed, guessed height ("20mm should be
 *      enough for one insight") BEFORE knowing how many lines the
 *      actual finding/action text would wrap to. A long insight wraps
 *      to 3+ lines and blows past that guess, running off the bottom
 *      of the page with no new page triggered. Fixed by measuring the
 *      real wrapped line count with doc.splitTextToSize() FIRST, then
 *      reserving the page space that measurement says is actually
 *      needed — a card is now guaranteed to be measured correctly
 *      before a single pixel of it is drawn, and moves to a fresh page
 *      whole rather than splitting mid-card.
 */

import jsPDF from "jspdf";

export interface ReportInsight {
  finding: string;
  action: string;
}

export interface ReportChart {
  title: string;
  why: string;
  imageDataUrl: string | null; // null if this chart failed to capture
}

export interface ReportQaPair {
  question: string;
  answer: string;
}

// A printed BI report reads as a light, clean document, not a dark
// terminal theme — this keeps the app's actual brand colors (teal
// accent, indigo finding, amber action) as accents against white,
// which is also the only sane choice for anything anyone might print.
const COLORS = {
  accent: [13, 148, 136] as [number, number, number], // darkened #2DD4BF for print contrast
  accentLight: [204, 251, 241] as [number, number, number], // pale teal tint for header band
  finding: [79, 70, 200] as [number, number, number], // darkened #818CF8
  action: [161, 98, 7] as [number, number, number], // darkened #FBBF24
  muted: [110, 110, 122] as [number, number, number],
  text: [24, 24, 32] as [number, number, number],
  cardBorder: [225, 225, 232] as [number, number, number],
};

const PAGE_WIDTH = 210; // A4 mm
const PAGE_HEIGHT = 297;
const MARGIN = 18;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2;
const CARD_PAD = 5;

/**
 * Normalizes text before it ever reaches jsPDF. See this file's
 * top-of-file docstring, bug #1, for why this exists — one
 * unsupported character corrupts a whole line's encoding, not just
 * itself, so this replaces the known common cases and strips anything
 * else outside jsPDF's core-font-safe range as a last resort.
 */
function sanitizeForPdf(text: string): string {
  return text
    .replace(/[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]/g, " ") // all Unicode space variants -> plain space
    .replace(/[\u2013\u2014]/g, " - ") // en/em dash
    .replace(/[\u2018\u2019]/g, "'") // curly single quotes
    .replace(/[\u201C\u201D]/g, '"') // curly double quotes
    .replace(/[\u00D7]/g, "x") // multiplication sign
    .replace(/\u2192/g, "->") // arrow
    .replace(/\u2022/g, "-") // bullet
    .replace(/[^\x00-\xFF]/g, ""); // anything else outside WinAnsi range — better silently dropped than corrupting the line
}

function measureWrapped(doc: jsPDF, text: string, maxWidth: number): string[] {
  return doc.splitTextToSize(sanitizeForPdf(text), maxWidth);
}

function ensureSpace(doc: jsPDF, y: number, needed: number): number {
  if (y + needed > PAGE_HEIGHT - MARGIN) {
    doc.addPage();
    return MARGIN;
  }
  return y;
}

function drawSectionHeader(doc: jsPDF, title: string, y: number): number {
  y = ensureSpace(doc, y, 16);
  doc.setFillColor(...COLORS.accent);
  doc.rect(MARGIN, y - 3.2, 2.2, 5.2, "F"); // small accent tab, not just a line — reads more like a report section marker
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12.5);
  doc.setTextColor(...COLORS.text);
  doc.text(title, MARGIN + 5, y);
  y += 3;
  doc.setDrawColor(...COLORS.cardBorder);
  doc.setLineWidth(0.3);
  doc.line(MARGIN, y, PAGE_WIDTH - MARGIN, y);
  return y + 7;
}

/** Pre-measures the full card height BEFORE reserving page space or
 * drawing anything — this is the fix for bug #2 above. */
function drawInsightCard(doc: jsPDF, index: number, insight: ReportInsight, y: number): number {
  const innerWidth = CONTENT_WIDTH - CARD_PAD * 2 - 3; // 3mm for the left accent bar
  const findingLines = measureWrapped(doc, `${index + 1}.  ${insight.finding}`, innerWidth);
  const actionLines = measureWrapped(doc, insight.action, innerWidth - 4);

  const findingHeight = findingLines.length * 5;
  const actionHeight = actionLines.length * 4.3;
  const cardHeight = CARD_PAD * 2 + findingHeight + 2 + 4 + actionHeight; // +4 for the "RECOMMENDED ACTION" label line

  y = ensureSpace(doc, y, cardHeight + 4);

  // Card background + left accent bar (mirrors the app UI's indigo
  // left-border treatment on insight cards)
  doc.setFillColor(250, 250, 252);
  doc.roundedRect(MARGIN, y, CONTENT_WIDTH, cardHeight, 1.5, 1.5, "F");
  doc.setFillColor(...COLORS.finding);
  doc.rect(MARGIN, y, 1.6, cardHeight, "F");

  let cy = y + CARD_PAD + 3;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(...COLORS.finding);
  doc.text(findingLines, MARGIN + CARD_PAD + 3, cy);
  cy += findingHeight + 2;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(7);
  doc.setTextColor(...COLORS.action);
  doc.text("RECOMMENDED ACTION", MARGIN + CARD_PAD + 3, cy);
  cy += 4;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.text);
  doc.text(actionLines, MARGIN + CARD_PAD + 3, cy);

  return y + cardHeight + 5;
}

function drawQaCard(doc: jsPDF, qa: ReportQaPair, y: number): number {
  const innerWidth = CONTENT_WIDTH - CARD_PAD * 2;
  const qLines = measureWrapped(doc, `Q: ${qa.question}`, innerWidth);
  const aLines = measureWrapped(doc, qa.answer, innerWidth);
  const qHeight = qLines.length * 4.6;
  const aHeight = aLines.length * 4.4;
  const cardHeight = CARD_PAD * 2 + qHeight + 2 + aHeight;

  y = ensureSpace(doc, y, cardHeight + 4);

  doc.setFillColor(250, 250, 252);
  doc.setDrawColor(...COLORS.cardBorder);
  doc.setLineWidth(0.2);
  doc.roundedRect(MARGIN, y, CONTENT_WIDTH, cardHeight, 1.5, 1.5, "FD");

  let cy = y + CARD_PAD + 2;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9.5);
  doc.setTextColor(...COLORS.text);
  doc.text(qLines, MARGIN + CARD_PAD, cy);
  cy += qHeight + 2;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(...COLORS.muted);
  doc.text(aLines, MARGIN + CARD_PAD, cy);

  return y + cardHeight + 5;
}

export function generateAutoAnalyzeReport(params: {
  filename: string;
  insights: ReportInsight[];
  charts: ReportChart[];
  followUpQa: ReportQaPair[];
}) {
  const { filename, insights, charts, followUpQa } = params;
  const doc = new jsPDF({ unit: "mm", format: "a4" });

  // ── Header / masthead ──
  doc.setFillColor(...COLORS.accentLight);
  doc.rect(0, 0, PAGE_WIDTH, 32, "F");
  doc.setFillColor(...COLORS.accent);
  doc.rect(0, 0, PAGE_WIDTH, 1.6, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(...COLORS.text);
  doc.text("Auto-Analyze Report", MARGIN, 16);

  doc.setFont("courier", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(...COLORS.muted);
  doc.text(sanitizeForPdf(filename), MARGIN, 23);
  doc.text(new Date().toLocaleString(), MARGIN, 28);

  let y = 42;

  // ── Business Insights ──
  y = drawSectionHeader(doc, "Business Insights", y);
  if (insights.length === 0) {
    doc.setFont("helvetica", "italic");
    doc.setFontSize(9.5);
    doc.setTextColor(...COLORS.muted);
    doc.text("No insights were generated for this dataset.", MARGIN, y);
    y += 8;
  } else {
    insights.forEach((ins, i) => {
      y = drawInsightCard(doc, i, ins, y);
    });
  }

  // ── Visualizations ──
  const capturedCharts = charts.filter((c) => c.imageDataUrl);
  if (capturedCharts.length > 0) {
    y += 3;
    y = drawSectionHeader(doc, "Visualizations", y);

    for (const chart of capturedCharts) {
      const imgWidth = CONTENT_WIDTH - CARD_PAD * 2;
      const imgHeight = imgWidth * (400 / 700); // matches the capture aspect ratio in captureChartImage
      const titleLines = measureWrapped(doc, chart.title, imgWidth);
      const cardHeight = CARD_PAD * 2 + titleLines.length * 5 + (chart.why ? 4 : 0) + imgHeight + 3;

      y = ensureSpace(doc, y, cardHeight + 4);

      doc.setFillColor(255, 255, 255);
      doc.setDrawColor(...COLORS.cardBorder);
      doc.setLineWidth(0.2);
      doc.roundedRect(MARGIN, y, CONTENT_WIDTH, cardHeight, 1.5, 1.5, "FD");

      let cy = y + CARD_PAD + 3;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10.5);
      doc.setTextColor(...COLORS.text);
      doc.text(titleLines, MARGIN + CARD_PAD, cy);
      cy += titleLines.length * 5;

      if (chart.why) {
        doc.setFont("courier", "normal");
        doc.setFontSize(7.5);
        doc.setTextColor(...COLORS.muted);
        doc.text(sanitizeForPdf(`supports insight ${chart.why}`), MARGIN + CARD_PAD, cy);
        cy += 4;
      }

      try {
        doc.addImage(chart.imageDataUrl as string, "PNG", MARGIN + CARD_PAD, cy, imgWidth, imgHeight);
      } catch {
        // A single chart image failing to embed shouldn't break the
        // rest of the report — note it and move on.
        doc.setFont("helvetica", "italic");
        doc.setFontSize(9);
        doc.setTextColor(...COLORS.muted);
        doc.text("(chart image could not be embedded)", MARGIN + CARD_PAD, cy + 6);
      }

      y = y + cardHeight + 5;
    }
  }

  // ── Follow-up Q&A ──
  if (followUpQa.length > 0) {
    y += 3;
    y = drawSectionHeader(doc, "Follow-up Questions", y);
    followUpQa.forEach((qa) => {
      y = drawQaCard(doc, qa, y);
    });
  }

  // ── Footer on every page ──
  const pageCount = doc.getNumberOfPages();
  for (let p = 1; p <= pageCount; p++) {
    doc.setPage(p);
    doc.setDrawColor(...COLORS.cardBorder);
    doc.setLineWidth(0.2);
    doc.line(MARGIN, PAGE_HEIGHT - 12, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 12);
    doc.setFont("courier", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(...COLORS.muted);
    doc.text("Analyst Assistant - Auto-Analyze Report", MARGIN, PAGE_HEIGHT - 7);
    doc.text(`Page ${p} of ${pageCount}`, PAGE_WIDTH - MARGIN - 22, PAGE_HEIGHT - 7);
  }

  const safeName = filename.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]/gi, "_");
  doc.save(`${safeName}_auto_analyze_report.pdf`);
}

/**
 * Captures a Plotly graphDiv as a PNG data URL for embedding in the
 * report. Dynamically imports plotly.js-dist-min (rather than importing
 * it at module scope) so this stays client-only, consistent with how
 * ResultDisplay.tsx's <Plot> component is itself loaded with ssr:false.
 */
export async function captureChartImage(graphDiv: HTMLElement): Promise<string | null> {
  try {
    const Plotly = (await import("plotly.js-dist-min")).default;
    const dataUrl = await Plotly.toImage(graphDiv, {
      format: "png",
      width: 700,
      height: 400,
    });
    return dataUrl;
  } catch (err) {
    console.error("Chart capture failed:", err);
    return null;
  }
}
