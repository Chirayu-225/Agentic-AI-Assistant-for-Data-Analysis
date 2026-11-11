/**
 * app/api/clean/fill/route.ts — fills the remaining blank cells from evidence inside the file itself: a value that
 * another column proves (every row of one customer has the same postal code), or the column median for numbers.
 * Text with no evidence stays blank unless `unknown` is true (a separate click that labels it "Unknown"). No model is
 * involved, so nothing is invented. Explicitly opt-in, because it changes the values the user queries.
 */

import { NextRequest, NextResponse } from "next/server";
import { fillMissing, upstreamErrorResponse } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { fileUrl, filename, sessionId, unknown, unknownColumns } = body as Record<string, unknown>;

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
    const columns = Array.isArray(unknownColumns)
      ? unknownColumns.filter((c): c is string => typeof c === "string")
      : [];
    const result = await fillMissing({ fileUrl, filename, sessionId, unknown: unknown === true, unknownColumns: columns });
    return NextResponse.json(result);
  } catch (err) {
    return upstreamErrorResponse(err, "Fill service unavailable");
  }
}
