/**
 * Next.js API Route: POST /api/dataset/upload-zip
 * Proxies multipart ZIP uploads to the FastAPI backend for extraction.
 *
 * Without this static handler, Next.js matches the dynamic [name] route which
 * only implements GET and DELETE, causing a 405 Method Not Allowed on ZIP uploads.
 */
import { NextRequest, NextResponse } from 'next/server';

function backendBase(): string {
  const dev = process.env.NODE_ENV !== 'production';
  const defaultPort = dev ? '8000' : (process.env.BACKEND_PORT || '18000');
  return process.env.BACKEND_URL || `http://127.0.0.1:${defaultPort}`;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const res = await fetch(`${backendBase()}/api/dataset/upload-zip`, {
      method: 'POST',
      body: formData,
    });

    const contentType = res.headers.get('content-type') || '';

    if (!res.ok) {
      const errText = await res.text();
      return NextResponse.json(
        { error: `Backend ${res.status}`, detail: errText.slice(0, 1000) },
        { status: res.status }
      );
    }

    if (contentType.includes('application/json')) {
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

export const config = { api: { bodyParser: false } };
