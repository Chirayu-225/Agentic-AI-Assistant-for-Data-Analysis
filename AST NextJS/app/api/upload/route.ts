/**
 * app/api/upload/route.ts — authorizes direct browser-to-Blob-storage
 * uploads. This route never receives the file itself, only a small
 * token-exchange request — which is exactly how it sidesteps Vercel's
 * 4.5MB serverless function body limit for the actual file bytes.
 *
 * Flow:
 *   1. Browser calls upload() (see components/FileUpload.tsx), which
 *      hits this route first to get a signed token.
 *   2. Browser uploads the actual file bytes directly to Vercel Blob,
 *      using that token — this app's server is not in that data path.
 *   3. Vercel Blob calls onUploadCompleted (below) once the upload
 *      finishes, so the file's identity (URL) plus the session/user
 *      namespace get logged server-side, in a call your Next.js server
 *      DID make, not just data the browser claims.
 *
 * TODO once auth exists: replace the placeholder session lookup in
 * onBeforeGenerateToken with your real session check, and reject
 * unauthenticated requests before generating a token at all.
 */

import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { NextResponse } from "next/server";

const MAX_UPLOAD_BYTES = 200 * 1024 * 1024; // matches the original app's 200MB CSV limit

export async function POST(request: Request): Promise<NextResponse> {
  const body = (await request.json()) as HandleUploadBody;

  try {
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        // TODO: replace with real auth once it exists —
        //   const session = await getSession(request);
        //   if (!session) throw new Error("Unauthorized");
        // Without a real check here, this route allows anonymous
        // uploads to your Blob store — fine for local dev, not for
        // anything public.

        return {
          allowedContentTypes: [
            "text/csv",
            "application/vnd.ms-excel", // legacy .xls
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // .xlsx
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown",
          ],
          maximumSizeInBytes: MAX_UPLOAD_BYTES,
          // Without this, uploading a file with the same name twice (very
          // common during testing, and inevitable once two different
          // users upload same-named files) fails with a "blob already
          // exists" error — Vercel Blob doesn't overwrite by default.
          // A random suffix means every upload gets its own unique key,
          // no collisions possible.
          addRandomSuffix: true,
          // Carries the session_id through to onUploadCompleted below —
          // this is what namespaces the file to a user (same session_id
          // concept as sandbox_proxy.py / rag_engine.py use on the
          // Python side).
          tokenPayload: clientPayload ?? "",
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        // This runs server-side, after the upload genuinely finished —
        // a good place to log/audit "user X uploaded file Y" once you
        // have a database, without ever having touched the file bytes.
        //
        // LOCAL DEV NOTE: on @vercel/blob v2+, this callback only fires
        // automatically when the app is actually deployed on Vercel.
        // Running `next dev` locally, Blob storage has no public URL to
        // call back to, so this simply won't run — that's expected, not
        // a bug. If you need to test it locally, set VERCEL_BLOB_CALLBACK_URL
        // to a tunnel URL (e.g. ngrok) per Vercel's docs. Once deployed to
        // Vercel for real, it works with no extra config.
        console.log("Upload completed:", blob.url, "namespace:", tokenPayload);
      },
    });

    return NextResponse.json(jsonResponse);
  } catch (err) {
    // handleUpload throws for oversized files, disallowed content types,
    // a rejected token request, or — most commonly during setup — a
    // missing/invalid BLOB_READ_WRITE_TOKEN. The browser only ever shows
    // "Failed to retrieve the client token" regardless of which of these
    // it is, so log the real reason here where it's actually visible.
    console.error("[/api/upload] handleUpload failed:", err);
    return NextResponse.json(
      { error: (err as Error).message },
      { status: 400 }
    );
  }
}
