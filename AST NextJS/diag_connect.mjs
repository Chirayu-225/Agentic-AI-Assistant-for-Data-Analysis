// diag_connect.mjs - makes the SAME call the Next.js server makes (POST with the secret header, 60 s timeout), reading
// .env.local exactly like the app, and prints the REAL underlying error. Never prints the secret. Temporary.
// Run from the "AST NextJS" folder:   node diag_connect.mjs
import fs from "node:fs";
const env = {};
for (const line of fs.readFileSync(".env.local", "utf8").split(/\r?\n/)) {
  const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
  if (m && !line.trim().startsWith("#")) env[m[1]] = m[2].replace(/^["']|["']$/g, "");
}
const base = (env.PYTHON_SERVICE_URL || "").replace(/\/+$/, "");
console.log("PYTHON_SERVICE_URL :", JSON.stringify(base), "| secret set:", Boolean(env.PYTHON_SERVICE_SECRET), "| node", process.version);
async function call(label, link) {
  try {
    const res = await fetch(`${base}/connect/google-drive`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Service-Secret": env.PYTHON_SERVICE_SECRET ?? "" },
      body: JSON.stringify({ link, sessionId: "diag" }),
      signal: AbortSignal.timeout(60000),
    });
    console.log(label, "->", res.status, (await res.text()).slice(0, 220));
  } catch (e) {
    console.log(label, "-> FAILED:", e.message, "| cause:", e.cause && (e.cause.code || e.cause.message), "| name:", e.name);
  }
}
await call("folder link", "https://drive.google.com/drive/folders/1GKJLXG95j0cyvRCpgCVYsUJawU_gl6CC?usp=sharing");
