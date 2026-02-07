/**
 * Next.js API Routes: /api/jobs
 * Job management endpoints
 *
 * GET - List all jobs
 * POST - Create and start a new job
 */

import { NextRequest, NextResponse } from 'next/server';
import { jobManager } from '@/lib/node-services/job-manager';

/**
 * GET /api/jobs
 * List all jobs
 */
export async function GET() {
  try {
    const jobs = jobManager.getAllJobs();

    return NextResponse.json({
      success: true,
      jobs,
      count: jobs.length,
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

/**
 * POST /api/jobs
 * Create and start a new job
 *
 * Body: {
 *   type: 'training' | 'tagging' | 'captioning_blip' | 'captioning_git',
 *   command: string,
 *   args: string[],
 *   cwd?: string
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, command, args, cwd } = body;

    if (!type || !command || !args) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: type, command, args',
        },
        { status: 400 }
      );
    }

    const jobId = jobManager.createJob(type, command, args, cwd);
    const started = jobManager.startJob(jobId);

    if (!started) {
      return NextResponse.json(
        {
          success: false,
          error: 'Failed to start job',
        },
        { status: 500 }
      );
    }

    const job = jobManager.getJob(jobId);

    return NextResponse.json({
      success: true,
      job_id: jobId,
      job,
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
