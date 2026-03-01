/**
 * Next.js API Routes: /api/jobs/tagging
 * Create tagging jobs via the tagging orchestrator.
 *
 * POST - Start a new WD14 tagging job
 *
 * The orchestrator automatically selects the best backend:
 * - Python (GPU-accelerated via onnxruntime-gpu) if available
 * - Node.js ONNX Runtime (CPU fallback) otherwise
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  startTagging,
  type TaggingRequest,
} from '@/lib/node-services/tagging-orchestrator';

/**
 * POST /api/jobs/tagging
 * Start a new WD14 tagging job
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.dataset_dir) {
      return NextResponse.json(
        { success: false, error: 'Missing required field: dataset_dir' },
        { status: 400 }
      );
    }

    // Build request with defaults
    const taggingRequest: TaggingRequest = {
      dataset_dir: body.dataset_dir,
      model: body.model || 'SmilingWolf/wd-vit-large-tagger-v3',
      threshold: body.threshold ?? 0.35,
      general_threshold: body.general_threshold,
      character_threshold: body.character_threshold,
      caption_extension: body.caption_extension || '.txt',
      batch_size: body.batch_size,
      caption_separator: body.caption_separator || ', ',
      undesired_tags: body.undesired_tags || '',
      remove_underscore: body.remove_underscore ?? true,
      character_tags_first: body.character_tags_first ?? false,
      always_first_tags: body.always_first_tags,
      append_tags: body.append_tags ?? false,
      recursive: body.recursive ?? false,
      use_rating_tags: body.use_rating_tags ?? false,
      use_rating_tags_as_last_tag: body.use_rating_tags_as_last_tag ?? false,
      tag_replacement: body.tag_replacement,
      character_tag_expand: body.character_tag_expand ?? false,
      force_prepend_trigger: body.force_prepend_trigger ?? true,
    };

    const { jobId, backend } = await startTagging(taggingRequest);

    const backendLabel =
      backend === 'python'
        ? 'Python onnxruntime (GPU-accelerated)'
        : 'Node.js ONNX Runtime (CPU)';

    return NextResponse.json({
      success: true,
      job_id: jobId,
      backend,
      message: `Tagging job started (${backendLabel})`,
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
