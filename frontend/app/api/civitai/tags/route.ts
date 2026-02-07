/**
 * Next.js API Route: GET /api/civitai/tags
 * Get available Civitai tags for filtering
 *
 * Migrated from Python FastAPI: api/routes/civitai.py -> get_tags
 */

import { NextRequest, NextResponse } from 'next/server';
import { civitaiFetch, CivitaiTagsResponse } from '@/lib/node-services/civitai-service';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;

    // Extract query parameters
    const limit = parseInt(searchParams.get('limit') || '20', 10);
    const query = searchParams.get('query');

    // Build params
    const params: Record<string, string | number | boolean | undefined> = {
      limit: Math.min(Math.max(limit, 1), 200),
    };

    if (query) {
      params.query = query;
    }

    // Fetch from Civitai
    const result = await civitaiFetch<CivitaiTagsResponse>('/tags', params);

    if (result.success && result.data) {
      return NextResponse.json({
        success: true,
        data: result.data,
      });
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
