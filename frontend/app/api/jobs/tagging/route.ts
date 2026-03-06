/**
 * Next.js API Routes: /api/jobs/tagging
 * Create tagging jobs via the Node.js job manager.
 * Spawns the Python WD14 tagger as a child process with real-time log capture.
 *
 * POST - Start a new WD14 tagging job
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  createTaggingJob,
  type TaggingJobConfig,
} from '@/lib/node-services/job-manager';

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

    // Map request body to TaggingJobConfig
    const config: TaggingJobConfig = {
      inputDir: body.dataset_dir,
      model: body.model || 'SmilingWolf/wd-vit-large-tagger-v3',
      threshold: body.threshold ?? 0.35,
      generalThreshold: body.general_threshold,
      characterThreshold: body.character_threshold,
      captionExtension: body.caption_extension || '.txt',
      batchSize: body.batch_size,
      captionSeparator: body.caption_separator || ', ',
      undesiredTags: body.undesired_tags || '',
      alwaysFirstTags: body.always_first_tags,
      tagReplacement: body.tag_replacement,
      removeUnderscore: body.remove_underscore ?? true,
      characterTagsFirst: body.character_tags_first ?? false,
      appendTags: body.append_tags ?? false,
      recursive: body.recursive ?? false,
      useRatingTags: body.use_rating_tags ?? false,
      useRatingTagsAsLastTag: body.use_rating_tags_as_last_tag ?? false,
      characterTagExpand: body.character_tag_expand ?? false,
    };

    const jobId = await createTaggingJob(config);

    return NextResponse.json({
      success: true,
      job_id: jobId,
      backend: 'python',
      message: 'Tagging job started (Python onnxruntime GPU-accelerated)',
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
