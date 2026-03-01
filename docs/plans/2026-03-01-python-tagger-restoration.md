# Python Tagger Restoration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the Python WD14 tagger as primary backend via a tagging orchestrator service, with Node.js ONNX as automatic fallback.

**Architecture:** New `tagging-orchestrator.ts` service handles backend detection (Python vs Node.js), job dispatch through the standard job manager, and post-processing (trigger word injection). The tagging API route becomes a thin handler like the captioning routes.

**Tech Stack:** Node.js, TypeScript, child_process spawn, ONNX Runtime (fallback), Python wd14 tagger

---

### Task 1: Expand `createTaggingJob()` in job-manager.ts

**Files:**
- Modify: `frontend/lib/node-services/job-manager.ts:406-434`

**Step 1: Replace the existing `createTaggingJob()` function**

Replace lines 406-434 of `job-manager.ts` with an expanded version that accepts a full config object and builds proper CLI args for the Python tagger.

```typescript
/**
 * Configuration for tagging jobs (passed from orchestrator)
 */
export interface TaggingJobConfig {
  inputDir: string;
  model?: string;
  threshold?: number;
  generalThreshold?: number;
  characterThreshold?: number;
  captionExtension?: string;
  batchSize?: number;
  removeUnderscore?: boolean;
  undesiredTags?: string;
  characterTagsFirst?: boolean;
  alwaysFirstTags?: string;
  appendTags?: boolean;
  recursive?: boolean;
  useRatingTags?: boolean;
  useRatingTagsAsLastTag?: boolean;
  captionSeparator?: string;
  tagReplacement?: string;
  characterTagExpand?: boolean;
}

/**
 * Create and start a tagging job (Python subprocess)
 */
export async function createTaggingJob(
  config: TaggingJobConfig
): Promise<string> {
  const projectRoot = getProjectRoot();
  const pythonPath = getPythonPath();

  const scriptPath = path.join(
    projectRoot,
    'custom',
    'tag_images_by_wd14_tagger.py'
  );

  // Build CLI args from config
  const args: string[] = [
    scriptPath,
    config.inputDir,
    '--onnx', // Always use ONNX for GPU support
  ];

  if (config.model) {
    args.push('--repo_id', config.model);
  }
  if (config.threshold !== undefined) {
    args.push('--thresh', String(config.threshold));
  }
  if (config.generalThreshold !== undefined) {
    args.push('--general_threshold', String(config.generalThreshold));
  }
  if (config.characterThreshold !== undefined) {
    args.push('--character_threshold', String(config.characterThreshold));
  }
  if (config.captionExtension) {
    args.push('--caption_extension', config.captionExtension);
  }
  if (config.batchSize !== undefined) {
    args.push('--batch_size', String(config.batchSize));
  }
  if (config.captionSeparator) {
    args.push('--caption_separator', config.captionSeparator);
  }
  if (config.undesiredTags) {
    args.push('--undesired_tags', config.undesiredTags);
  }
  if (config.alwaysFirstTags) {
    args.push('--always_first_tags', config.alwaysFirstTags);
  }
  if (config.tagReplacement) {
    args.push('--tag_replacement', config.tagReplacement);
  }
  if (config.removeUnderscore) {
    args.push('--remove_underscore');
  }
  if (config.characterTagsFirst) {
    args.push('--character_tags_first');
  }
  if (config.appendTags) {
    args.push('--append_tags');
  }
  if (config.recursive) {
    args.push('--recursive');
  }
  if (config.useRatingTags) {
    args.push('--use_rating_tags');
  }
  if (config.useRatingTagsAsLastTag) {
    args.push('--use_rating_tags_as_last_tag');
  }
  if (config.characterTagExpand) {
    args.push('--character_tag_expand');
  }

  const jobId = jobManager.createJob('tagging', pythonPath, args, projectRoot);
  jobManager.startJob(jobId);

  return jobId;
}
```

**Step 2: Verify the file compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No errors related to job-manager.ts

**Step 3: Commit**

```bash
git add frontend/lib/node-services/job-manager.ts
git commit -m "feat: expand createTaggingJob() to accept full config object

Replaces the minimal (inputDir, model) signature with a TaggingJobConfig
interface that maps all Python tagger CLI arguments. Always passes --onnx
for GPU support via onnxruntime-gpu.

Closes part of #150"
```

---

### Task 2: Create the tagging orchestrator

**Files:**
- Create: `frontend/lib/node-services/tagging-orchestrator.ts`

**Step 1: Create the orchestrator file**

```typescript
/**
 * Tagging Orchestrator - manages backend selection and job dispatch
 *
 * Detects whether Python tagger is available (GPU-accelerated) and falls
 * back to Node.js ONNX Runtime (CPU-only) if not. Handles post-processing
 * for trigger word injection after either backend completes.
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

export interface TaggingRequest {
  dataset_dir: string;
  model?: string;
  threshold?: number;
  general_threshold?: number;
  character_threshold?: number;
  caption_extension?: string;
  batch_size?: number;
  caption_separator?: string;
  undesired_tags?: string;
  remove_underscore?: boolean;
  character_tags_first?: boolean;
  always_first_tags?: string;
  append_tags?: boolean;
  recursive?: boolean;
  use_rating_tags?: boolean;
  use_rating_tags_as_last_tag?: boolean;
  tag_replacement?: string;
  character_tag_expand?: boolean;
  force_prepend_trigger?: boolean;
}

export type TaggingBackend = 'python' | 'node';

// ========== Backend Detection ==========

let cachedBackend: TaggingBackend | null = null;

/**
 * Detect which tagging backend is available.
 * Checks for Python executable and tagger script.
 * Result is cached for the process lifetime.
 */
export async function detectBackend(): Promise<TaggingBackend> {
  if (cachedBackend) return cachedBackend;

  const projectRoot = getProjectRoot();
  const scriptPath = path.join(projectRoot, 'custom', 'tag_images_by_wd14_tagger.py');

  try {
    // Check script exists
    await fs.access(scriptPath);

    // Check python is available
    execSync('python --version', { stdio: 'ignore', timeout: 5000 });

    cachedBackend = 'python';
    console.log('[TaggingOrchestrator] Backend: Python (GPU-accelerated)');
  } catch {
    cachedBackend = 'node';
    console.log('[TaggingOrchestrator] Backend: Node.js ONNX (CPU fallback)');
  }

  return cachedBackend;
}

/**
 * Reset cached backend detection (for testing or after environment changes)
 */
export function resetBackendCache(): void {
  cachedBackend = null;
}

// ========== Job Dispatch ==========

/**
 * Start a tagging job using the best available backend.
 * Returns the job ID for tracking via WebSocket/polling.
 */
export async function startTagging(request: TaggingRequest): Promise<{
  jobId: string;
  backend: TaggingBackend;
}> {
  const backend = await detectBackend();

  if (backend === 'python') {
    return startPythonTagging(request);
  } else {
    return startNodeTagging(request);
  }
}

/**
 * Start tagging via Python subprocess (GPU-accelerated)
 */
async function startPythonTagging(request: TaggingRequest): Promise<{
  jobId: string;
  backend: TaggingBackend;
}> {
  const config: TaggingJobConfig = {
    inputDir: request.dataset_dir,
    model: request.model,
    threshold: request.threshold,
    generalThreshold: request.general_threshold,
    characterThreshold: request.character_threshold,
    captionExtension: request.caption_extension,
    batchSize: request.batch_size,
    removeUnderscore: request.remove_underscore,
    undesiredTags: request.undesired_tags,
    characterTagsFirst: request.character_tags_first,
    alwaysFirstTags: request.always_first_tags,
    appendTags: request.append_tags,
    recursive: request.recursive,
    useRatingTags: request.use_rating_tags,
    useRatingTagsAsLastTag: request.use_rating_tags_as_last_tag,
    captionSeparator: request.caption_separator,
    tagReplacement: request.tag_replacement,
    characterTagExpand: request.character_tag_expand,
  };

  const jobId = await createTaggingJob(config);

  // Listen for job completion to run post-processing
  if (request.force_prepend_trigger && request.always_first_tags) {
    const triggerTags = request.always_first_tags;
    const separator = request.caption_separator || ', ';
    const ext = request.caption_extension || '.txt';
    const datasetDir = request.dataset_dir;

    const onComplete = async (completedJobId: string, exitCode: number) => {
      if (completedJobId !== jobId) return;
      jobManager.events.removeListener('complete', onComplete);

      if (exitCode === 0) {
        try {
          await postProcessTriggerTags(datasetDir, triggerTags, separator, ext);
          jobManager.events.emit('log', jobId, {
            timestamp: Date.now(),
            level: 'info',
            message: `Post-processing: prepended trigger tags [${triggerTags}]`,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          jobManager.events.emit('log', jobId, {
            timestamp: Date.now(),
            level: 'error',
            message: `Post-processing failed: ${msg}`,
          });
        }
      }
    };
    jobManager.events.on('complete', onComplete);
  }

  return { jobId, backend: 'python' };
}

/**
 * Start tagging via Node.js ONNX Runtime (CPU fallback)
 * Uses the proper jobManager.createJob() API (not the private map hack).
 */
async function startNodeTagging(request: TaggingRequest): Promise<{
  jobId: string;
  backend: TaggingBackend;
}> {
  // Create job through proper API
  const jobId = jobManager.createJob('tagging', 'node', ['onnx-tagging']);

  // Manually transition to running since we're not spawning a process
  const job = jobManager.getJob(jobId);
  if (!job) throw new Error('Failed to create tagging job');
  job.status = 'running';
  job.started_at = Date.now();
  jobManager.events.emit('status', jobId, 'running');

  // Build config for Node.js tagging service
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

  // Run tagging async (don't block the response)
  (async () => {
    try {
      jobManager.events.emit('log', jobId, {
        timestamp: Date.now(),
        level: 'info',
        message: `Starting tagging with model: ${nodeConfig.model} (Node.js CPU fallback)`,
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
          });
        }
      );

      // Post-process trigger tags if needed
      if (request.force_prepend_trigger && request.always_first_tags) {
        await postProcessTriggerTags(
          request.dataset_dir,
          request.always_first_tags,
          request.caption_separator || ', ',
          request.caption_extension || '.txt'
        );
        jobManager.events.emit('log', jobId, {
          timestamp: Date.now(),
          level: 'info',
          message: `Post-processing: prepended trigger tags [${request.always_first_tags}]`,
        });
      }

      // Mark completed
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
        });
      }
    } catch (error) {
      job.status = 'failed';
      job.completed_at = Date.now();
      job.error = error instanceof Error ? error.message : String(error);
      jobManager.events.emit('status', jobId, 'failed');
      jobManager.events.emit('error', jobId, job.error);
    }
  })();

  return { jobId, backend: 'node' };
}

// ========== Post-Processing ==========

/**
 * Prepend trigger tags to caption files if not already present.
 * Runs after either backend completes successfully.
 */
async function postProcessTriggerTags(
  datasetDir: string,
  triggerTagsStr: string,
  separator: string,
  captionExtension: string
): Promise<void> {
  const projectRoot = getProjectRoot();
  const datasetPath = path.resolve(projectRoot, datasetDir);

  const triggerTags = triggerTagsStr.split(',').map((t) => t.trim()).filter(Boolean);
  if (triggerTags.length === 0) return;

  // Find all caption files
  const files = await fs.readdir(datasetPath);
  const captionFiles = files.filter((f) => f.endsWith(captionExtension));

  for (const captionFile of captionFiles) {
    const filePath = path.join(datasetPath, captionFile);
    const content = await fs.readFile(filePath, 'utf-8');
    const existingTags = content.split(separator).map((t) => t.trim()).filter(Boolean);

    // Check which trigger tags are missing (case-insensitive)
    const existingLower = new Set(existingTags.map((t) => t.toLowerCase()));
    const missingTriggers = triggerTags.filter(
      (t) => !existingLower.has(t.toLowerCase())
    );

    if (missingTriggers.length > 0) {
      const newTags = [...missingTriggers, ...existingTags];
      await fs.writeFile(filePath, newTags.join(separator), 'utf-8');
    }
  }
}
```

**Step 2: Verify the file compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No errors related to tagging-orchestrator.ts

**Step 3: Commit**

```bash
git add frontend/lib/node-services/tagging-orchestrator.ts
git commit -m "feat: add tagging orchestrator for Python/Node.js backend selection

New service layer that detects Python tagger availability and dispatches
tagging jobs through the proper job manager flow. Falls back to Node.js
ONNX CPU if Python unavailable. Includes post-processing for trigger
word injection.

Part of #150"
```

---

### Task 3: Rewrite the tagging route to use the orchestrator

**Files:**
- Modify: `frontend/app/api/jobs/tagging/route.ts` (full rewrite, lines 1-150)

**Step 1: Replace the entire route file**

```typescript
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
import { startTagging, type TaggingRequest } from '@/lib/node-services/tagging-orchestrator';

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

    const backendLabel = backend === 'python'
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
```

**Step 2: Verify the file compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No errors

**Step 3: Commit**

```bash
git add frontend/app/api/jobs/tagging/route.ts
git commit -m "feat: rewrite tagging route to use orchestrator

Thin handler that delegates to tagging-orchestrator.ts. Validates input,
builds TaggingRequest, calls startTagging(), returns job_id + backend.
Follows the same pattern as the captioning routes.

Fixes #150: GPU support, job 404s, and WebSocket log streaming all
resolved by routing through the standard job manager flow."
```

---

### Task 4: Verify end-to-end integration

**Files:**
- Read-only verification of all modified files

**Step 1: Run TypeScript compilation check**

Run: `cd frontend && npx tsc --noEmit --pretty`
Expected: No errors

**Step 2: Run the linter**

Run: `cd frontend && npm run lint 2>&1 | tail -20`
Expected: No new errors in modified files

**Step 3: Verify the build succeeds**

Run: `cd frontend && npm run build 2>&1 | tail -30`
Expected: Build completes without errors. The orchestrator's lazy import of tagging-service.ts means native ONNX binaries won't break the build.

**Step 4: Commit build fix if needed**

If any compilation/lint issues arise, fix them and commit:
```bash
git commit -m "fix: resolve lint/type issues from tagger restoration"
```

---

### Task 5: Final commit and cleanup

**Step 1: Review all changes**

Run: `git diff main --stat` to see all files changed.
Verify exactly these files were modified/created:
- `frontend/lib/node-services/job-manager.ts` (modified)
- `frontend/lib/node-services/tagging-orchestrator.ts` (new)
- `frontend/app/api/jobs/tagging/route.ts` (modified)
- `docs/plans/2026-03-01-python-tagger-restoration-design.md` (new)
- `docs/plans/2026-03-01-python-tagger-restoration.md` (new)

**Step 2: Verify no regressions in other job types**

Confirm that `createBlipCaptioningJob`, `createGitCaptioningJob`, and `createTrainingJob` in job-manager.ts are unchanged.

Run: `grep -n "export async function create" frontend/lib/node-services/job-manager.ts`
Expected: All four factory functions present (training, tagging, blip, git)
