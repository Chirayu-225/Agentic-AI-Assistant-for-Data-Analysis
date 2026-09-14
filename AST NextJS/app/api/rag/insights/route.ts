/**
 * app/api/rag/insights/route.ts — auto-extracted ranked insights for an
 * already-indexed document, no query needed. Mirrors the other RAG routes.
 */

import { NextRequest, NextResponse } from "next/server";
import { getDocumentInsights } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { docId, filename } = body as Record<string, unknown>;

  if (typeof docId !== "string" || typeof filename !== "string") {
    return NextResponse.json(
      { error: "docId and filename are both required strings" },
      { status: 400 }
    );
  }

  try {
    const result = await getDocumentInsights({ docId, filename });
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Insights service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
