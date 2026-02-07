/**
 * Next.js API Route: POST /api/captions/replace
 * Replace text in captions
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> replace_text
 */

import { NextRequest, NextResponse } from 'next/server';
import { captionService } from '@/lib/node-services/caption-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.dataset_dir || body.find_text === undefined || body.replace_text === undefined) {
      return NextResponse.json(
        { error: 'Missing required fields: dataset_dir, find_text, replace_text' },
        { status: 400 }
      );
    }

    const result = await captionService.replaceText({
      dataset_dir: body.dataset_dir,
      find_text: body.find_text,
      replace_text: body.replace_text,
      use_regex: body.use_regex || false,
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
