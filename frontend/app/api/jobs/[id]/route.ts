/**
 * Next.js API Routes: /api/jobs/[id]
 * Single job management
 *
 * GET - Get job status
 * DELETE - Delete job (if not running)
 */

import { NextRequest, NextResponse } from 'next/server';
import { jobManager } from '@/lib/node-services/job-manager';

/**
 * GET /api/jobs/[id]
 * Get job status and details
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const job = jobManager.getJob(id);

    if (!job) {
      return NextResponse.json(
        {
          success: false,
          error: `Job ${id} not found`,
        },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
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

/**
 * DELETE /api/jobs/[id]
 * Delete job (only if not running)
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const deleted = jobManager.deleteJob(id);

    if (!deleted) {
      return NextResponse.json(
        {
          success: false,
          error: `Failed to delete job ${id} (may be running or not found)`,
        },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      message: `Job ${id} deleted`,
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
