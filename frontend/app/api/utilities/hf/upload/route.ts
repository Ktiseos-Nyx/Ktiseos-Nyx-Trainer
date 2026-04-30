/**
 * POST /api/utilities/hf/upload
 * Proxy to FastAPI's HuggingFace upload endpoint.
 * If hf_token is omitted, the stored token is injected server-side so the
 * raw credential is never sent to the browser.
 */

import { NextRequest, NextResponse } from 'next/server';
import { settingsService } from '@/lib/node-services/settings-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    let hfToken: string = body.hf_token || '';
    if (!hfToken) {
      const keys = await settingsService.getApiKeys();
      hfToken = keys.huggingface_token || '';
    }

    if (!hfToken) {
      return NextResponse.json(
        { error: 'No HuggingFace token provided and none saved in settings.' },
        { status: 400 }
      );
    }

    const backendPort = process.env.BACKEND_PORT || '8000';
    const backendUrl = process.env.BACKEND_URL || `http://127.0.0.1:${backendPort}`;

    const response = await fetch(`${backendUrl}/api/utilities/hf/upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, hf_token: hfToken }),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
