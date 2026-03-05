/**
 * Next.js API Routes: /api/jobs/tagging
 * Proxies tagging requests to the FastAPI backend which handles
 * WD14 tagging with GPU-accelerated onnxruntime.
 *
 * POST - Start a new WD14 tagging job
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

/**
 * POST /api/jobs/tagging
 * Proxy to FastAPI backend's /api/dataset/tag endpoint
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${BACKEND_URL}/api/dataset/tag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.detail || `Backend error: ${response.status}` },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
