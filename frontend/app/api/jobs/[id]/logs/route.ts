/**
 * Next.js API Routes: /api/jobs/[id]/logs
 * Get job logs
 *
 * GET - Get all logs for a job
 *
 * Note: For real-time streaming, use WebSocket connection to /ws/jobs/[id]/logs
 * This HTTP endpoint returns the current log snapshot
 */

import { NextRequest, NextResponse } from 'next/server';
import { jobManager } from '@/lib/node-services/job-manager';

/**
 * GET /api/jobs/[id]/logs
 * Get all logs for a job
 *
 * Query params:
 * - since: timestamp (optional) - only return logs after this timestamp
 * - limit: number (optional) - limit number of logs returned
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

    // Parse query params
    const searchParams = request.nextUrl.searchParams;
    const sinceParam = searchParams.get('since');
    const limitParam = searchParams.get('limit');

    let logs = job.logs;

    // Filter by timestamp if 'since' is provided
    if (sinceParam) {
      const since = parseInt(sinceParam, 10);
      logs = logs.filter((log) => log.timestamp > since);
    }

    // Limit number of logs if 'limit' is provided
    if (limitParam) {
      const limit = parseInt(limitParam, 10);
      logs = logs.slice(-limit); // Get last N logs
    }

    return NextResponse.json({
      success: true,
      job_id: id,
      logs,
      total_logs: job.logs.length,
      status: job.status,
      progress: job.progress,
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
