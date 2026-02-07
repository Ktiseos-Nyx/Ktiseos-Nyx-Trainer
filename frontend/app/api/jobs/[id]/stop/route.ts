/**
 * Next.js API Routes: /api/jobs/[id]/stop
 * Stop a running job
 *
 * POST - Stop/kill job
 */

import { NextRequest, NextResponse } from 'next/server';
import { jobManager } from '@/lib/node-services/job-manager';

/**
 * POST /api/jobs/[id]/stop
 * Stop a running job
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const stopped = jobManager.stopJob(id);

    if (!stopped) {
      return NextResponse.json(
        {
          success: false,
          error: `Failed to stop job ${id} (may not be running or not found)`,
        },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      message: `Job ${id} stopped`,
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
