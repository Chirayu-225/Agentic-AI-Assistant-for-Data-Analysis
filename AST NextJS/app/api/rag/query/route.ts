/**
 * app/api/rag/query/route.ts — asks a question about an already-indexed
 * document. Mirrors app/api/analyze/route.ts.
 */

import { NextRequest, NextResponse } from "next/server";
import { queryDocument } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { docId, filename, query } = body as Record<string, unknown>;

  if (
    typeof docId !== "string" ||
    typeof filename !== "string" ||
    typeof query !== "string"
  ) {
    return NextResponse.json(
      { error: "docId, filename, and query are all required strings" },
      { status: 400 }
    );
  }

  if (!query.trim()) {
    return NextResponse.json({ error: "query cannot be empty" }, { status: 400 });
  }

  try {
    const result = await queryDocument({ docId, filename, query });
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Query service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
