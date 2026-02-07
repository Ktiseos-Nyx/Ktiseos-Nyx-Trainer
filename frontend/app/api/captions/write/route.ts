/**
 * Next.js API Route: POST /api/captions/write
 * Write a single caption file
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> write_caption
 */

import { NextRequest, NextResponse } from 'next/server';
import { captionService } from '@/lib/node-services/caption-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.image_path || body.caption_text === undefined) {
      return NextResponse.json(
        { error: 'Missing required fields: image_path, caption_text' },
        { status: 400 }
      );
    }

    const result = await captionService.writeCaption({
      image_path: body.image_path,
      caption_text: body.caption_text,
      caption_extension: body.caption_extension || '.txt',
    });

    if (!result.success) {
      return NextResponse.json(
        { error: result.message, detail: result.errors },
        { status: 400 }
      );
    }

    return NextResponse.json(result);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);

    return NextResponse.json(
      { error: errorMessage, detail: errorMessage },
      { status: 500 }
    );
  }
}
