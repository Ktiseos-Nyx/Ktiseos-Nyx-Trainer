/**
 * Next.js API Routes: /api/jobs/training
 * Create training jobs
 *
 * POST - Start a new training job
 */

import { NextRequest, NextResponse } from 'next/server';
import { createTrainingJob, type ModelArchitecture } from '@/lib/node-services/job-manager';

const VALID_ARCHITECTURES: ModelArchitecture[] = ['sd15', 'sdxl', 'sd3', 'flux', 'lumina'];

/**
 * POST /api/jobs/training
 * Start a new training job
 *
 * Body: {
 *   config_path: string,
 *   dataset_path: string,
 *   architecture?: 'sd15' | 'sdxl' | 'sd3' | 'flux' | 'lumina' (default: 'sdxl')
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { config_path, dataset_path, architecture } = body;

    if (!config_path || !dataset_path) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: config_path, dataset_path',
        },
        { status: 400 }
      );
    }

    // Validate architecture if provided
    const arch: ModelArchitecture = architecture || 'sdxl';
    if (!VALID_ARCHITECTURES.includes(arch)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid architecture: ${architecture}. Valid: ${VALID_ARCHITECTURES.join(', ')}`,
        },
        { status: 400 }
      );
    }

    const jobId = await createTrainingJob(config_path, dataset_path, arch);

    return NextResponse.json({
      success: true,
      job_id: jobId,
      architecture: arch,
      message: `Training job started (${arch})`,
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
