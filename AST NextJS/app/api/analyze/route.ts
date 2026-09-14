/**
 * app/api/analyze/route.ts — the only route the frontend calls to run a
 * natural-language query against an already-uploaded CSV.
 *
 * This route does NOT do any pandas/LLM work itself — it validates the
 * request and forwards it to the Python service (see lib/pythonClient.ts
 * and the PYTHON_API_CONTRACT.md note at the bottom of this repo's README
 * for the exact contract the Python side needs to implement).
 */

import { NextRequest, NextResponse } from "next/server";
import { analyzeCsv } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { fileUrl, filename, query, sessionId } = body as Record<string, unknown>;

  if (
    typeof fileUrl !== "string" ||
    typeof filename !== "string" ||
    typeof query !== "string" ||
    typeof sessionId !== "string"
  ) {
    return NextResponse.json(
      { error: "fileUrl, filename, query, and sessionId are all required strings" },
      { status: 400 }
    );
  }

  if (!query.trim()) {
    return NextResponse.json({ error: "query cannot be empty" }, { status: 400 });
  }

  try {
    const result = await analyzeCsv({ fileUrl, filename, query, sessionId });
    return NextResponse.json(result);
  } catch (err) {
    // A failure here is almost always the Python service being
    // unreachable, timed out, or rejecting the shared-secret header —
    // not a user error, so don't imply the query itself was wrong.
    return NextResponse.json(
      { error: `Analysis service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
