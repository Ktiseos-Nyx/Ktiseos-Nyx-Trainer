/**
 * Next.js API Routes: /api/jobs/captioning/git
 * Create GIT captioning jobs
 *
 * POST - Start a new GIT captioning job
 */

import { NextRequest, NextResponse } from 'next/server';
import { createGitCaptioningJob } from '@/lib/node-services/job-manager';

/**
 * POST /api/jobs/captioning/git
 * Start a new GIT captioning job
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

    const jobId = await createGitCaptioningJob(input_dir, caption_extension);

    return NextResponse.json({
      success: true,
      job_id: jobId,
      message: 'GIT captioning job started',
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
