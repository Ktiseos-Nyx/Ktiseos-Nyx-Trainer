/**
 * Next.js API Routes: /api/jobs/captioning/blip
 * Create BLIP captioning jobs
 *
 * POST - Start a new BLIP captioning job
 */

import { NextRequest, NextResponse } from 'next/server';
import { createBlipCaptioningJob } from '@/lib/node-services/job-manager';

/**
 * POST /api/jobs/captioning/blip
 * Start a new BLIP captioning job
 *
 * Body: BLIPConfig (dataset_dir required; caption_extension, batch_size,
 * max_workers, beam_search, num_beams, top_p, max_length, min_length,
 * recursive, debug optional)
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    if (!body.dataset_dir) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required field: dataset_dir',
        },
        { status: 400 }
      );
    }

    const jobId = await createBlipCaptioningJob({
      inputDir: body.dataset_dir,
      captionExtension: body.caption_extension ?? '.txt',
      batchSize: body.batch_size,
      maxWorkers: body.max_workers,
      beamSearch: body.beam_search ?? false,
      numBeams: body.num_beams,
      topP: body.top_p,
      maxLength: body.max_length,
      minLength: body.min_length,
      recursive: body.recursive ?? false,
      debug: body.debug ?? false,
    });

    return NextResponse.json({
      success: true,
      job_id: jobId,
      message: 'BLIP captioning job started',
    });
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
