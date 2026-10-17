/**
 * app/api/upload/route.ts — receives the file from the browser and streams it
 * straight to the Python service, which stores it on local disk.
 *
 * No third-party storage and no token: running the project on another computer
 * needs only PYTHON_SERVICE_URL and PYTHON_SERVICE_SECRET. The request body is
 * passed through as a stream, never buffered here, so large files (up to the
 * backend's MAX_UPLOAD_MB, default 200) cost almost no memory.
 *
 * (This used to authorize direct browser-to-Vercel-Blob uploads to get around
 * Vercel's 4.5MB serverless body limit. That limit does not exist when running
 * locally. If you ever deploy to Vercel, bring a storage service back.)
 *
 * Request:  POST /api/upload   body = raw file bytes, header X-Filename (URL-encoded)
 * Response: { url: "local://<id>", bytes: <n> }
 */

import { NextResponse } from "next/server";
import { uploadToPython, UploadRejectedError, upstreamErrorResponse } from "@/lib/pythonClient";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request): Promise<NextResponse> {
  if (!request.body) {
    return NextResponse.json({ error: "No file received" }, { status: 400 });
  }

  let filename = "upload";
  try {
    filename = decodeURIComponent(request.headers.get("x-filename") ?? "upload");
  } catch {
    /* malformed encoding — keep the default name */
  }

  try {
    const result = await uploadToPython(request.body, filename);
    return NextResponse.json(result);
  } catch (err) {
    if (err instanceof UploadRejectedError) {
      // Backend said no (too large, empty file, bad secret...) — pass its reason through.
      console.error("[/api/upload] rejected by Python service:", err.status, err.message);
      return NextResponse.json({ error: err.message }, { status: err.status === 401 ? 502 : err.status });
    }
    console.error("[/api/upload] failed:", err);
    return upstreamErrorResponse(err, "Upload service unavailable");
  }
}
