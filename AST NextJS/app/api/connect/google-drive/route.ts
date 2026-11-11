/**
 * app/api/connect/google-drive/route.ts — imports a link-shared Google Drive file or Google Sheet
 * into the backend's local storage and returns {url: "local://<id>", filename}, exactly like an
 * upload. The backend builds the download request itself from the file ID in the link; the link
 * string is never fetched as a URL. See AST 2.0/data_connectors.py.
 */

import { NextRequest, NextResponse } from "next/server";
import { importFromGoogleDrive, upstreamErrorResponse } from "@/lib/pythonClient";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const { link, sessionId } = (body ?? {}) as Record<string, unknown>;

  if (typeof link !== "string" || !link.trim() || link.length > 8192 || typeof sessionId !== "string") {
    return NextResponse.json({ success: false, error: "Paste the Google Drive link to a file or folder." }, { status: 400 });
  }

  try {
    return NextResponse.json(await importFromGoogleDrive({ link: link.trim(), sessionId }));
  } catch (err) {
    return upstreamErrorResponse(err, "Google Drive import unavailable");
  }
}
