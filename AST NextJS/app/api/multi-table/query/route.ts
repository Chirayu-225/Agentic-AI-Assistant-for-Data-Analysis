/**
 * app/api/multi-table/query/route.ts — answers a natural-language
 * question that requires joining 2+ already-uploaded datasets, via
 * DuckDB-backed SQL running inside the existing sandboxed pipeline.
 * See multi_table_service.py's docstring for the full safety design.
 */

import { NextRequest, NextResponse } from "next/server";
import { runMultiTableQuery } from "@/lib/pythonClient";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { datasets, query, sessionId } = body as Record<string, unknown>;

  if (
    !Array.isArray(datasets) ||
    datasets.length < 2 ||
    typeof query !== "string" ||
    typeof sessionId !== "string"
  ) {
    return NextResponse.json(
      { error: "datasets (array of 2+), query, and sessionId are all required" },
      { status: 400 }
    );
  }

  try {
    const result = await runMultiTableQuery({
      datasets: datasets as { alias: string; fileUrl: string; filename: string }[],
      query,
      sessionId,
    });
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: `Multi-table query service unavailable: ${(err as Error).message}` },
      { status: 502 }
    );
  }
}
