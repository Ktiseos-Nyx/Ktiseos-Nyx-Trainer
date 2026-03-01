# Design: Restore Python WD14 Tagger via Tagging Orchestrator

**Date**: 2026-03-01
**Issue**: #150 - WD14 Tagger: Node.js ONNX is CPU-only, need Python tagger for GPU
**Branch**: `fix/vastai-frontend-binding-112`

## Problem

The WD14 tagging route (`/api/jobs/tagging`) was migrated to use a Node.js ONNX service (`tagging-service.ts`), but `onnxruntime-node` is CPU-only. This makes tagging painfully slow on GPU instances (VastAI). The route also bypasses the job manager's public API, causing job 404s and broken WebSocket log streaming.

## Decision

Restore the Python tagger (`custom/tag_images_by_wd14_tagger.py`) as the primary backend, routed through the standard job manager subprocess flow. Keep the Node.js ONNX tagger as an automatic fallback for environments where Python isn't available. Introduce a tagging orchestrator service layer to manage backend selection, job dispatch, and post-processing.

## Architecture

### New file: `frontend/lib/node-services/tagging-orchestrator.ts`

Single entry point for all tagging operations. Responsibilities:

1. **Backend detection** - Check if Python + tagger script exist, cache result
2. **Job dispatch** - Python path: `createTaggingJob()` (child process via job manager). Node.js path: `jobManager.createJob()` + `taggingService.tagDataset()` with proper job lifecycle
3. **Post-processing** - Trigger word injection after tagging completes (flag-controlled via `forcePrependTrigger`)
4. **Unified config** - One `TaggingRequest` interface consumed by both backends

### Data flow

```
Browser POST /api/jobs/tagging
        |
        v
  route.ts (validate input, call orchestrator, return job_id)
        |
        v
  tagging-orchestrator.ts
        |
        +-- detectBackend() -> 'python' | 'node'
        |
        +-- if 'python':
        |     createTaggingJob(config) via job-manager.ts
        |       -> spawn: python custom/tag_images_by_wd14_tagger.py [args]
        |       -> stdout/stderr parsed by job manager -> events -> WebSocket
        |       -> on exit(0): postProcess() if forcePrependTrigger
        |
        +-- if 'node':
        |     jobManager.createJob() (proper public API)
        |       -> taggingService.tagDataset() with progress callbacks -> events
        |       -> on complete: postProcess() if forcePrependTrigger
        |
        +-- postProcess(config):
              read .txt caption files, prepend trigger tags if missing, write back
```

### Config interface

```typescript
interface TaggingJobConfig {
  inputDir: string;
  model: string;
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
  forcePrependTrigger?: boolean;
}
```

### Files modified

| File | Change |
|------|--------|
| `frontend/lib/node-services/tagging-orchestrator.ts` | **New** - orchestrator service |
| `frontend/lib/node-services/job-manager.ts` | Expand `createTaggingJob()` to accept full config, build CLI args from config |
| `frontend/app/api/jobs/tagging/route.ts` | Rewrite to thin handler calling orchestrator |
| `frontend/lib/node-services/tagging-service.ts` | No changes (used as-is by orchestrator for Node.js fallback) |

### Post-processing: trigger word injection

After either backend completes successfully, if `forcePrependTrigger === true` and `alwaysFirstTags` is non-empty:

1. Glob all caption files (`.txt` by default) in the dataset directory
2. For each file, read current tags
3. Check if trigger tags are already present (case-insensitive)
4. If missing, prepend to tag list
5. Write back with the configured separator

This runs in Node.js regardless of which backend did the tagging.

### Python detection

```typescript
async function detectBackend(): Promise<'python' | 'node'> {
  // 1. Check python executable exists (try 'python', 'python3')
  // 2. Check custom/tag_images_by_wd14_tagger.py exists
  // 3. Cache result for process lifetime
  // Returns 'python' if both pass, 'node' otherwise
}
```

### Progress parsing for Python backend

The Python tagger uses `tqdm` which outputs to stderr. The job manager's `parseLogLine()` already handles `%` pattern matching for progress extraction. tqdm lines like `45%|████ | 5/11 [00:03<00:04]` will be parsed for the percentage.

### What this fixes

1. **GPU acceleration** - Python `onnxruntime-gpu` uses CUDA automatically
2. **Job 404s** - Jobs created through `jobManager.createJob()` are in the shared Map
3. **WebSocket logs** - Child process stdout/stderr flows through the standard event pipeline
4. **Trigger word injection** - Post-processing handles tags the model can't detect

## Rejected alternatives

- **Node.js ONNX GPU**: `onnxruntime-node` is CPU-only. An experimental `onnxruntime-gpu` npm package exists but is poorly documented and has platform-specific build issues. Not worth the hassle when Python works.
- **Python-only, no fallback**: Simpler but loses resilience for edge cases where Python isn't available.
- **Minimal rewire (no orchestrator)**: Stuffs all logic into the route. Works but mixes concerns. The orchestrator gives a clean home for backend detection, dispatch, and post-processing.
