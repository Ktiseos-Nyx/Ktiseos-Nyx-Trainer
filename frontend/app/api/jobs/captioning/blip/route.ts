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
 * Body: {
 *   input_dir: string,
 *   caption_extension?: string (default: '.txt')
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { input_dir, caption_extension } = body;

    if (!input_dir) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required field: input_dir',
        },
        { status: 400 }
      );
    }

    const jobId = await createBlipCaptioningJob(input_dir, caption_extension);

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
