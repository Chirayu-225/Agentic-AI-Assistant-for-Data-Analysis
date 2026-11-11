/**
 * app/api/connect/mysql/route.ts — imports ONE table from a MySQL server into the backend's local
 * storage and returns {url: "local://<id>", filename, rows}, exactly like an upload.
 *
 * The password travels browser -> this route -> Python service for this one request. It is never
 * stored or logged here: the request body is not logged and errors never include it.
 */

import { NextRequest, NextResponse } from "next/server";
import { importFromMySql, upstreamErrorResponse } from "@/lib/pythonClient";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function text(v: unknown, max = 256): string | null {
  return typeof v === "string" && v.length <= max ? v : null;
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const b = (body ?? {}) as Record<string, unknown>;

  const host = text(b.host)?.trim();
  const user = text(b.user);
  const password = text(b.password, 512) ?? "";
  const database = text(b.database, 64)?.trim();
  const table = text(b.table, 64)?.trim();
  const sessionId = text(b.sessionId);
  const port = b.port === undefined || b.port === "" ? 3306 : Number(b.port);
  const limit = b.limit === undefined || b.limit === "" ? undefined : Number(b.limit);

  if (!host || !user || !database || !table || !sessionId) {
    return NextResponse.json(
      { success: false, error: "Host, username, database and table name are all required." },
      { status: 400 }
    );
  }
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    return NextResponse.json({ success: false, error: "The port must be a number between 1 and 65535." }, { status: 400 });
  }
  if (limit !== undefined && (!Number.isInteger(limit) || limit < 1)) {
    return NextResponse.json({ success: false, error: "The row limit must be a whole number." }, { status: 400 });
  }

  try {
    return NextResponse.json(
      await importFromMySql({ host, port, user, password, database, table, limit, ssl: b.ssl === true, sessionId })
    );
  } catch (err) {
    return upstreamErrorResponse(err, "MySQL import unavailable");
  }
}
