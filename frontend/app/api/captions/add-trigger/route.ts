/**
 * Next.js API Route: POST /api/captions/add-trigger
 * Add trigger word to captions
 *
 * Migrated from Python FastAPI: api/routes/dataset.py -> add_trigger_word
 */

import { NextRequest, NextResponse } from 'next/server';
import { captionService } from '@/lib/node-services/caption-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.dataset_dir || !body.trigger_word || !body.position) {
      return NextResponse.json(
        { error: 'Missing required fields: dataset_dir, trigger_word, position' },
        { status: 400 }
      );
    }

    // Validate position
    if (body.position !== 'start' && body.position !== 'end') {
      return NextResponse.json(
        { error: 'Position must be "start" or "end"' },
        { status: 400 }
      );
    }

    const result = await captionService.addTriggerWord({
      dataset_dir: body.dataset_dir,
      trigger_word: body.trigger_word,
      position: body.position,
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
