/**
 * app/api/rag/index/route.ts — indexes an already-uploaded document
 * (file URL in hand) for RAG Q&A. Mirrors app/api/analyze/route.ts.
 *
 * ADDITIONALLY: if the Python service found real tables inside a PDF
 * (see table_extraction.py), each comes back as base64 CSV bytes. This
 * route stores each one through the same upload path as a browser file
 * (uploadToPython -> local://<id>) so each extracted table becomes an
 * ordinary file URL the existing CSV Analysis pipeline can consume
 * completely unmodified.
 */

import { NextRequest, NextResponse } from "next/server";
import { indexDocument, uploadToPython, type ExtractedTable, upstreamErrorResponse } from "@/lib/pythonClient";

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
            const { url } = await uploadToPython(new Uint8Array(csvBuffer), `${table.title}.csv`);
            // csv_base64 dropped once uploaded — the browser doesn't
            // need both a URL and the raw bytes for the same table.
            return { ...table, url, csv_base64: undefined };
          } catch (err) {
            // One table failing to upload shouldn't sink the others,
            // or the document's RAG indexing, which already succeeded.
            console.error("Failed to store extracted table:", err);
            return table;
          }
        })
      );
    }

    return NextResponse.json(result);
  } catch (err) {
    return upstreamErrorResponse(err, "Indexing service unavailable");
  }
}
