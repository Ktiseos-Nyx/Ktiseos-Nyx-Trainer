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
 * Body: GITConfig (dataset_dir required; caption_extension, model_id,
 * batch_size, max_workers, max_length, remove_words, recursive, debug optional)
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

    const jobId = await createGitCaptioningJob({
      inputDir: body.dataset_dir,
      captionExtension: body.caption_extension ?? '.txt',
      modelId: body.model_id,
      batchSize: body.batch_size,
      maxWorkers: body.max_workers,
      maxLength: body.max_length,
      removeWords: body.remove_words ?? false,
      recursive: body.recursive ?? false,
      debug: body.debug ?? false,
    });

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
