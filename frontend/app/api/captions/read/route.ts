/**
 * Next.js API Route: GET /api/captions/read
 * Read a single caption file
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> read_caption
 */

import { NextRequest, NextResponse } from 'next/server';
import { captionService } from '@/lib/node-services/caption-service';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const image_path = searchParams.get('image_path');

    if (!image_path) {
      return NextResponse.json(
        { error: 'Missing image_path parameter' },
        { status: 400 }
      );
    }

    const result = await captionService.readCaption({
      image_path,
      caption_extension: searchParams.get('caption_extension') || '.txt',
    });

    return NextResponse.json(result);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
