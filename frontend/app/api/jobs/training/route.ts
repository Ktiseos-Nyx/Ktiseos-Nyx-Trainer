/**
 * Next.js API Routes: /api/jobs/training
 * Create training jobs
 *
 * POST - Start a new training job
 */

import { NextRequest, NextResponse } from 'next/server';
import { createTrainingJob } from '@/lib/node-services/job-manager';

/**
 * POST /api/jobs/training
 * Start a new training job
 *
 * Body: {
 *   config_path: string,
 *   dataset_path: string
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { config_path, dataset_path } = body;

    if (!config_path || !dataset_path) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: config_path, dataset_path',
        },
        { status: 400 }
      );
    }

    const jobId = await createTrainingJob(config_path, dataset_path);

    return NextResponse.json({
      success: true,
      job_id: jobId,
      message: 'Training job started',
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
