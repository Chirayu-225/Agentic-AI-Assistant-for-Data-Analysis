/**
 * app/api/rag/index/route.ts — indexes an already-uploaded document
 * (Blob URL in hand) for RAG Q&A. Mirrors app/api/analyze/route.ts.
 *
 * ADDITIONALLY: if the Python service found real tables inside a PDF
 * (see table_extraction.py), each comes back as base64 CSV bytes. This
 * route uploads each one to Vercel Blob server-side (using `put()`
 * directly — not the client-upload token-exchange flow FileUpload.tsx
 * uses, since these bytes originate on the server, not from a browser
 * file input) so each extracted table becomes an ordinary Blob URL the
 * existing CSV Analysis pipeline can consume completely unmodified.
 */

import { NextRequest, NextResponse } from "next/server";
import { put } from "@vercel/blob";
import { indexDocument, type ExtractedTable } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { fileUrl, filename, sessionId } = body as Record<string, unknown>;

  if (
    typeof fileUrl !== "string" ||
    typeof filename !== "string" ||
    typeof sessionId !== "string"
  ) {
    return NextResponse.json(
      { error: "fileUrl, filename, and sessionId are all required strings" },
      { status: 400 }
    );
  }

  try {
    const result = await indexDocument({ fileUrl, filename, sessionId });

    if (result.success && result.tables && result.tables.length > 0) {
      result.tables = await Promise.all(
        result.tables.map(async (table: ExtractedTable) => {
          if (!table.csv_base64) return table;
          try {
            const csvBuffer = Buffer.from(table.csv_base64, "base64");
            const safeTitle = table.title.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
            const blob = await put(
              `extracted-tables/${sessionId}/${result.doc_id}/${safeTitle}.csv`,
              csvBuffer,
              { access: "public", contentType: "text/csv", addRandomSuffix: true }
            );
            // csv_base64 dropped once uploaded — the browser doesn't
            // need both a URL and the raw bytes for the same table.
            return { ...table, url: blob.url, csv_base64: undefined };
          } catch (err) {
            // One table failing to upload shouldn't sink the others,
            // or the document's RAG indexing, which already succeeded.
            console.error("Failed to upload extracted table to Blob:", err);
            return table;
          }
        })
      );
    }

    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Indexing service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
