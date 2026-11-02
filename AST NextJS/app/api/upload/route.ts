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

import { NextResponse, type NextRequest } from "next/server";
import { uploadToPython, UploadRejectedError, upstreamErrorResponse } from "@/lib/pythonClient";
import { ACCESS_COOKIE, checkAccess } from "@/lib/accessGate";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest): Promise<NextResponse> {
  // This route is excluded from middleware.ts so large uploads aren't
  // truncated at Next.js's 10 MB middleware body limit — which means the
  // access-code check has to happen here instead.
  const denied = await checkAccess(request.cookies.get(ACCESS_COOKIE)?.value);
  if (denied) return NextResponse.json({ error: denied.error }, { status: denied.status });

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
    const declaredSize = request.headers.get("x-file-size") ?? undefined;
    const result = await uploadToPython(request.body, filename, declaredSize);
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
