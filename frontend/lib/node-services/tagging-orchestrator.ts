/**
 * Tagging Orchestrator - Single entry point for all tagging operations
 *
 * Handles backend detection (Python vs Node.js), job dispatch,
 * and post-processing (trigger tag prepending).
 *
 * Python path: spawns custom/tag_images_by_wd14_tagger.py via job-manager subprocess
 * Node.js path: runs tagging-service.ts in-process via ONNX Runtime (CPU fallback)
 */

import fs from 'fs/promises';
import path from 'path';
import { execSync } from 'child_process';
import {
  jobManager,
  createTaggingJob,
  getProjectRoot,
  type TaggingJobConfig,
} from './job-manager';
import type { TaggingConfig } from './tagging-service';

// ========== Types ==========

/**
 * API request body for tagging operations.
 * All fields are snake_case to match the REST API convention.
 */
export interface TaggingRequest {
  /** Directory containing images to tag (required) */
  dataset_dir: string;
  /** HuggingFace repo ID for the tagger model */
  model?: string;
  /** Global confidence threshold */
  threshold?: number;
  /** Threshold for general category tags */
  general_threshold?: number;
  /** Threshold for character category tags */
  character_threshold?: number;
  /** Extension for output caption files (e.g. ".txt") */
  caption_extension?: string;
  /** Batch size for inference */
  batch_size?: number;
  /** Separator between tags in caption files */
  caption_separator?: string;
  /** Comma-separated list of tags to exclude */
  undesired_tags?: string;
  /** Replace underscores with spaces */
  remove_underscore?: boolean;
  /** Place character tags before general tags */
  character_tags_first?: boolean;
  /** Comma-separated list of tags to always place first */
  always_first_tags?: string;
  /** Append to existing caption files instead of overwriting */
  append_tags?: boolean;
  /** Search subdirectories recursively */
  recursive?: boolean;
  /** Add rating tags as first tag */
  use_rating_tags?: boolean;
  /** Add rating tags as last tag */
  use_rating_tags_as_last_tag?: boolean;
  /** Tag replacement rules: "source1,target1;source2,target2" */
  tag_replacement?: string;
  /** Expand character tag parentheses to separate tags */
  character_tag_expand?: boolean;
  /** Force prepend trigger tags to all caption files after tagging completes */
  force_prepend_trigger?: boolean;
}

export type TaggingBackend = 'python' | 'node';

// ========== Backend Detection ==========

/** Cached backend detection result (null = not yet detected) */
let cachedBackend: TaggingBackend | null = null;

/**
 * Detect whether to use the Python tagger (GPU-accelerated) or Node.js fallback (CPU).
 *
 * Checks:
 * 1. Python tagger script exists at {projectRoot}/custom/tag_images_by_wd14_tagger.py
 * 2. `python --version` succeeds (Python is installed and on PATH)
 *
 * Result is cached for the lifetime of the process.
 */
export async function detectBackend(): Promise<TaggingBackend> {
  if (cachedBackend !== null) {
    return cachedBackend;
  }

  const projectRoot = getProjectRoot();
  const taggerScript = path.join(projectRoot, 'custom', 'tag_images_by_wd14_tagger.py');

  try {
    // Check if tagger script exists
    await fs.access(taggerScript);

    // Check if python is available
    execSync('python --version', { stdio: 'ignore', timeout: 5000 });

    cachedBackend = 'python';
    console.log('[Tagging Orchestrator] Backend: Python (GPU-accelerated)');
  } catch {
    cachedBackend = 'node';
    console.log('[Tagging Orchestrator] Backend: Node.js ONNX Runtime (CPU fallback)');
  }

  return cachedBackend;
}

/**
 * Reset the cached backend detection (useful for testing).
 */
export function resetBackendCache(): void {
  cachedBackend = null;
}

// ========== Main Entry Point ==========

/**
 * Start a tagging job using the best available backend.
 *
 * Returns the job ID and which backend was selected.
 */
export async function startTagging(
  request: TaggingRequest
): Promise<{ jobId: string; backend: TaggingBackend }> {
  const backend = await detectBackend();

  if (backend === 'python') {
    return startPythonTagging(request);
  } else {
    return startNodeTagging(request);
  }
}

// ========== Python Backend ==========

/**
 * Start tagging via the Python WD14 tagger subprocess.
 * Maps snake_case TaggingRequest to camelCase TaggingJobConfig for job-manager.
 */
async function startPythonTagging(
  request: TaggingRequest
): Promise<{ jobId: string; backend: TaggingBackend }> {
  // Map snake_case request to camelCase TaggingJobConfig
  const config: TaggingJobConfig = {
    inputDir: request.dataset_dir,
    model: request.model,
    threshold: request.threshold,
    generalThreshold: request.general_threshold,
    characterThreshold: request.character_threshold,
    captionExtension: request.caption_extension,
    batchSize: request.batch_size,
    captionSeparator: request.caption_separator,
    undesiredTags: request.undesired_tags,
    alwaysFirstTags: request.always_first_tags,
    tagReplacement: request.tag_replacement,
    removeUnderscore: request.remove_underscore,
    characterTagsFirst: request.character_tags_first,
    appendTags: request.append_tags,
    recursive: request.recursive,
    useRatingTags: request.use_rating_tags,
    useRatingTagsAsLastTag: request.use_rating_tags_as_last_tag,
    characterTagExpand: request.character_tag_expand,
  };

  const jobId = await createTaggingJob(config);

  // If trigger tag post-processing is requested, listen for job completion
  if (request.force_prepend_trigger && request.always_first_tags) {
    const triggerTags = request.always_first_tags;
    const separator = request.caption_separator || ', ';
    const captionExt = request.caption_extension || '.txt';
    const datasetDir = request.dataset_dir;

    const onComplete = async (completedJobId: string, exitCode: number) => {
      if (completedJobId !== jobId) return; // Not our job
      jobManager.events.removeListener('complete', onComplete);

      if (exitCode === 0) {
        try {
          await postProcessTriggerTags(datasetDir, triggerTags, separator, captionExt);
          console.log(`[Tagging Orchestrator] Post-processed trigger tags for job ${jobId}`);
        } catch (err) {
          console.error(`[Tagging Orchestrator] Trigger tag post-processing failed for job ${jobId}:`, err);
        }
      }
    };

    jobManager.events.on('complete', onComplete);
  }

  return { jobId, backend: 'python' };
}

// ========== Node.js Backend ==========

/**
 * Start tagging via the Node.js ONNX Runtime (in-process, CPU).
 * Uses the public jobManager.createJob() API to register the job properly.
 */
async function startNodeTagging(
  request: TaggingRequest
): Promise<{ jobId: string; backend: TaggingBackend }> {
  // Create job through the proper public API
  const jobId = jobManager.createJob('tagging', 'node', ['onnx-tagging']);

  // Get the job and manually transition to running state
  const job = jobManager.getJob(jobId);
  if (!job) {
    throw new Error(`Failed to create job: ${jobId} not found after creation`);
  }

  job.status = 'running';
  job.started_at = Date.now();
  jobManager.events.emit('status', jobId, 'running');

  // Build TaggingConfig for the Node.js service with defaults
  const nodeConfig: TaggingConfig = {
    dataset_dir: request.dataset_dir,
    model: request.model || 'SmilingWolf/wd-vit-large-tagger-v3',
    threshold: request.threshold ?? 0.35,
    general_threshold: request.general_threshold,
    character_threshold: request.character_threshold,
    caption_extension: request.caption_extension || '.txt',
    caption_separator: request.caption_separator || ', ',
    undesired_tags: request.undesired_tags || '',
    remove_underscore: request.remove_underscore ?? true,
    character_tags_first: request.character_tags_first ?? false,
    use_rating_tags: request.use_rating_tags ?? false,
    use_rating_tags_as_last_tag: request.use_rating_tags_as_last_tag ?? false,
    append_tags: request.append_tags ?? false,
  };

  // Run tagging asynchronously (don't block the HTTP response)
  (async () => {
    try {
      console.log(`[Job ${jobId}] Starting WD14 tagging (Node.js ONNX Runtime)`);

      jobManager.events.emit('log', jobId, {
        timestamp: Date.now(),
        level: 'info',
        message: `Starting tagging with model: ${nodeConfig.model}`,
        raw: `[INFO] Starting tagging with model: ${nodeConfig.model}`,
      });

      // Lazy import to avoid native binary issues at build time
      const { taggingService } = await import('./tagging-service');

      const result = await taggingService.tagDataset(
        nodeConfig,
        (current, total, filename) => {
          const progress = Math.floor((current / total) * 100);
          job.progress = progress;
          jobManager.events.emit('progress', jobId, progress);

          jobManager.events.emit('log', jobId, {
            timestamp: Date.now(),
            level: 'info',
            message: `[${current}/${total}] Processing: ${filename}`,
            raw: `[INFO] [${current}/${total}] Processing: ${filename}`,
          });
        }
      );

      // Post-process trigger tags if requested
      if (request.force_prepend_trigger && request.always_first_tags) {
        try {
          await postProcessTriggerTags(
            request.dataset_dir,
            request.always_first_tags,
            request.caption_separator || ', ',
            request.caption_extension || '.txt'
          );
          console.log(`[Job ${jobId}] Post-processed trigger tags`);
        } catch (err) {
          console.error(`[Job ${jobId}] Trigger tag post-processing failed:`, err);
        }
      }

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

  return { jobId, backend: 'node' };
}

// ========== Post-Processing ==========

/**
 * Prepend trigger tags to all caption files in a dataset directory.
 *
 * For each caption file:
 * 1. Read existing content
 * 2. Split into individual tags
 * 3. Check which trigger tags are missing (case-insensitive)
 * 4. Prepend any missing trigger tags
 * 5. Write back
 *
 * This is idempotent - running it multiple times won't duplicate trigger tags.
 */
async function postProcessTriggerTags(
  datasetDir: string,
  triggerTagsStr: string,
  separator: string,
  captionExtension: string
): Promise<void> {
  const projectRoot = getProjectRoot();
  const resolvedDir = path.isAbsolute(datasetDir)
    ? datasetDir
    : path.resolve(projectRoot, datasetDir);

  // Parse trigger tags
  const triggerTags = triggerTagsStr
    .split(',')
    .map((t) => t.trim())
    .filter((t) => t.length > 0);

  if (triggerTags.length === 0) {
    return;
  }

  // Ensure caption extension starts with a dot
  const ext = captionExtension.startsWith('.') ? captionExtension : `.${captionExtension}`;

  // Find all caption files
  let entries: string[];
  try {
    entries = await fs.readdir(resolvedDir);
  } catch (err) {
    console.error(`[Tagging Orchestrator] Cannot read directory ${resolvedDir}:`, err);
    return;
  }

  const captionFiles = entries.filter((f) => f.endsWith(ext));

  if (captionFiles.length === 0) {
    console.log(`[Tagging Orchestrator] No caption files (${ext}) found in ${resolvedDir}`);
    return;
  }

  console.log(`[Tagging Orchestrator] Post-processing ${captionFiles.length} caption files with trigger tags: ${triggerTags.join(', ')}`);

  for (const captionFile of captionFiles) {
    const filePath = path.join(resolvedDir, captionFile);

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const existingTags = content
        .split(separator)
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      // Build lowercase set of existing tags for case-insensitive comparison
      const existingLower = new Set(existingTags.map((t) => t.toLowerCase()));

      // Find trigger tags that are missing
      const missingTriggers = triggerTags.filter(
        (t) => !existingLower.has(t.toLowerCase())
      );

      if (missingTriggers.length === 0) {
        continue; // All trigger tags already present
      }

      // Prepend missing trigger tags
      const newTags = [...missingTriggers, ...existingTags];
      const newContent = newTags.join(separator);

      await fs.writeFile(filePath, newContent, 'utf-8');
    } catch (err) {
      console.error(`[Tagging Orchestrator] Error processing ${captionFile}:`, err);
    }
  }
}
