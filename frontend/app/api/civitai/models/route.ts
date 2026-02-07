/**
 * Next.js API Route: GET /api/civitai/models
 * Browse Civitai models with filters
 *
 * Migrated from Python FastAPI: api/routes/civitai.py -> browse_models
 */

import { NextRequest, NextResponse } from 'next/server';
import { civitaiFetch, CivitaiModelsResponse } from '@/lib/node-services/civitai-service';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;

    // Extract query parameters
    const limit = parseInt(searchParams.get('limit') || '20', 10);
    const page = searchParams.get('page');
    const cursor = searchParams.get('cursor');
    const query = searchParams.get('query');
    const tag = searchParams.get('tag');
    const username = searchParams.get('username');
    const types = searchParams.get('types');
    const baseModel = searchParams.get('baseModel');
    const sort = searchParams.get('sort') || 'Highest Rated';
    const period = searchParams.get('period') || 'AllTime';
    const nsfw = searchParams.get('nsfw') === 'true';

    // Build params object
    const params: Record<string, string | number | boolean | undefined> = {
      limit: Math.min(Math.max(limit, 1), 100),
      sort,
      period,
    };

    // Handle pagination - cursor-based for search, page-based otherwise
    const hasSearch = query || tag || username;

    if (hasSearch) {
      if (query) params.query = query;
      if (tag) params.tag = tag;
      if (username) params.username = username;
      if (cursor) params.cursor = cursor;
    } else {
      params.page = page ? parseInt(page, 10) : 1;
    }

    if (types) params.types = types;
    if (baseModel) params.baseModel = baseModel;
    if (!nsfw) params.nsfw = 'false';

    // Fetch from Civitai
    const result = await civitaiFetch<CivitaiModelsResponse>('/models', params);

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
