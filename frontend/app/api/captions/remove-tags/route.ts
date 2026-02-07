/**
 * Next.js API Route: POST /api/captions/remove-tags
 * Remove specific tags from captions
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> remove_tags
 */

import { NextRequest, NextResponse } from 'next/server';
import { captionService } from '@/lib/node-services/caption-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.dataset_dir || !body.tags_to_remove) {
      return NextResponse.json(
        { error: 'Missing required fields: dataset_dir, tags_to_remove' },
        { status: 400 }
      );
    }

    // Ensure tags_to_remove is an array
    if (!Array.isArray(body.tags_to_remove)) {
      return NextResponse.json(
        { error: 'tags_to_remove must be an array' },
        { status: 400 }
      );
    }

    const result = await captionService.removeTags({
      dataset_dir: body.dataset_dir,
      tags_to_remove: body.tags_to_remove,
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
