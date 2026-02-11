/**
 * Next.js API Routes: /api/jobs/tagging
 * Create tagging jobs using Node.js ONNX Runtime (NO Python subprocess!)
 *
 * POST - Start a new WD14 tagging job
 */

import { NextRequest, NextResponse } from 'next/server';
import type { TaggingConfig } from '@/lib/node-services/tagging-service';
import { jobManager } from '@/lib/node-services/job-manager';

/**
 * POST /api/jobs/tagging
 * Start a new WD14 tagging job using Node.js ONNX Runtime
 *
 * Body: TaggingConfig (full configuration)
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.dataset_dir) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required field: dataset_dir',
        },
        { status: 400 }
      );
    }

    // Build config with defaults
    const config: TaggingConfig = {
      dataset_dir: body.dataset_dir,
      model: body.model || 'SmilingWolf/wd-vit-large-tagger-v3',
      threshold: body.threshold ?? 0.35,
      general_threshold: body.general_threshold,
      character_threshold: body.character_threshold,
      caption_extension: body.caption_extension || '.txt',
      caption_separator: body.caption_separator || ', ',
      undesired_tags: body.undesired_tags || '',
      remove_underscore: body.remove_underscore ?? true,
      character_tags_first: body.character_tags_first ?? false,
      use_rating_tags: body.use_rating_tags ?? false,
      use_rating_tags_as_last_tag: body.use_rating_tags_as_last_tag ?? false,
      append_tags: body.append_tags ?? false,
    };

    // Create job ID
    const jobId = `tagging_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    // Create job entry in job manager
    const job = {
      id: jobId,
      type: 'tagging' as const,
      status: 'running' as const,
      command: 'node',
      args: ['onnx-tagging'],
      created_at: Date.now(),
      started_at: Date.now(),
      logs: [],
    };

    // Register job (we're not using child_process, but we still track it)
    jobManager['jobs'].set(jobId, job);

    // Run tagging in background (async, don't await)
    (async () => {
      try {
        console.log(`[Job ${jobId}] Starting WD14 tagging (Node.js ONNX Runtime)`);

        // Log to job
        jobManager.events.emit('log', jobId, {
          timestamp: Date.now(),
          level: 'info',
          message: `Starting tagging with model: ${config.model}`,
          raw: `[INFO] Starting tagging with model: ${config.model}`,
        });

        // Lazy import: onnxruntime-node and sharp have native binaries that
        // break Next.js build if imported at module level
        const { taggingService } = await import('@/lib/node-services/tagging-service');

        const result = await taggingService.tagDataset(
          config,
          (current, total, filename) => {
            // Emit progress
            const progress = Math.floor((current / total) * 100);
            job.progress = progress;
            jobManager.events.emit('progress', jobId, progress);

            // Emit log
            jobManager.events.emit('log', jobId, {
              timestamp: Date.now(),
              level: 'info',
              message: `[${current}/${total}] Processing: ${filename}`,
              raw: `[INFO] [${current}/${total}] Processing: ${filename}`,
            });
          }
        );

        // Mark job as completed
        job.status = 'completed';
        job.completed_at = Date.now();
        job.exit_code = 0;
        job.progress = 100;

        jobManager.events.emit('status', jobId, 'completed');
        jobManager.events.emit('complete', jobId, 0);

        if (result.errors.length > 0) {
          jobManager.events.emit('log', jobId, {
            timestamp: Date.now(),
            level: 'error',
            message: `Completed with ${result.errors.length} errors`,
            raw: `[ERROR] Completed with ${result.errors.length} errors`,
          });
        }

        console.log(`[Job ${jobId}] Completed: ${result.processed}/${result.total} images`);
      } catch (error) {
        // Mark job as failed
        job.status = 'failed';
        job.completed_at = Date.now();
        job.error = error instanceof Error ? error.message : String(error);

        jobManager.events.emit('status', jobId, 'failed');
        jobManager.events.emit('error', jobId, job.error);

        console.error(`[Job ${jobId}] Failed:`, error);
      }
    })();

    return NextResponse.json({
      success: true,
      job_id: jobId,
      message: 'Tagging job started (Node.js ONNX Runtime - CPU)',
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
