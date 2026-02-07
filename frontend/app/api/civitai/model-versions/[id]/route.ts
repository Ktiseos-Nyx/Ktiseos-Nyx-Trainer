/**
 * Next.js API Route: GET /api/civitai/model-versions/[id]
 * Get information about a specific model version
 *
 * Migrated from Python FastAPI: api/routes/civitai.py -> get_model_version
 */

import { NextRequest, NextResponse } from 'next/server';
import { civitaiFetch, CivitaiModelVersion } from '@/lib/node-services/civitai-service';

interface RouteParams {
  params: Promise<{ id: string }>;
}

export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { id } = await params;
    const versionId = parseInt(id, 10);

    if (isNaN(versionId)) {
      return NextResponse.json(
        { success: false, error: 'Invalid version ID' },
        { status: 400 }
      );
    }

    // Fetch from Civitai
    const result = await civitaiFetch<CivitaiModelVersion>(`/model-versions/${versionId}`);

    if (result.success && result.data) {
      return NextResponse.json({
        success: true,
        data: result.data,
      });
    } else if (result.status === 404) {
      return NextResponse.json(
        { success: false, error: 'Model version not found' },
        { status: 404 }
      );
    } else {
      return NextResponse.json(
        { success: false, error: result.error },
        { status: result.status || 500 }
      );
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { success: false, error: errorMessage },
      { status: 500 }
    );
  }
}
