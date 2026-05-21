/**
 * Next.js API Route: POST /api/dataset/upload-zip
 * Proxies multipart ZIP uploads to the FastAPI backend for extraction.
 *
 * Without this static handler, Next.js matches the dynamic [name] route which
 * only implements GET and DELETE, causing a 405 Method Not Allowed on ZIP uploads.
 */
import { NextRequest, NextResponse } from 'next/server';

// Large ZIP uploads can take several minutes — prevent the default timeout from
// killing the request mid-stream.
export const maxDuration = 300;

/**
 * Determine the base URL for the backend service.
 *
 * Uses the BACKEND_URL environment variable if present; otherwise returns
 * http://127.0.0.1:<port> where the port is 8000 in development or BACKEND_PORT (default 18000) in production.
 *
 * @returns The base URL to use for backend requests.
 */
function backendBase(): string {
  const dev = process.env.NODE_ENV !== 'production';
  const defaultPort = dev ? '8000' : (process.env.BACKEND_PORT || '18000');
  return process.env.BACKEND_URL || `http://127.0.0.1:${defaultPort}`;
}

/**
 * Proxies multipart ZIP upload requests to the backend's /api/dataset/upload-zip endpoint and returns the backend response.
 *
 * Streams the request body directly to FastAPI without buffering it in Node.js
 * memory — avoids ECONNRESET on large files caused by buffering the entire
 * multipart body before forwarding.
 *
 * @param request - Incoming Next.js request containing multipart form data (the uploaded ZIP)
 * @returns A NextResponse containing the backend's JSON response (status mirrors the backend). If the backend returns non-OK, returns a JSON error with the backend status and truncated detail; if a proxy error occurs, returns a JSON error with status 500.
 */
export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type');
    if (!contentType) {
      return NextResponse.json(
        { error: 'Missing content-type header — multipart boundary required' },
        { status: 400 }
      );
    }

    const forwardHeaders: Record<string, string> = { 'content-type': contentType };
    const contentLength = request.headers.get('content-length');
    if (contentLength) forwardHeaders['content-length'] = contentLength;

    const res = await fetch(`${backendBase()}/api/dataset/upload-zip`, {
      method: 'POST',
      // Stream directly — do NOT await request.formData() which buffers the
      // entire body into memory before forwarding, causing timeouts on large ZIPs.
      body: request.body,
      headers: forwardHeaders,
      // Required by Node.js fetch to allow a streaming request body.
      // @ts-expect-error duplex is a valid Node.js fetch option not yet in the TypeScript types
      duplex: 'half',
    });

    const resContentType = res.headers.get('content-type') || '';

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { error: `Backend ${res.status}`, detail: errText.slice(0, 1000) },
        { status: res.status }
      );
    }

    if (resContentType.includes('application/json')) {
      return NextResponse.json(await res.json(), { status: res.status });
    }

    const text = await res.text();
    try {
      return NextResponse.json(JSON.parse(text), { status: res.status });
    } catch {
      return NextResponse.json(
        { success: true, detail: text.slice(0, 1000) },
        { status: res.status }
      );
    }
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: 'Proxy error', detail: msg }, { status: 500 });
  }
}
