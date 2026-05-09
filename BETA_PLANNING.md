# Beta Planning Document

**Current Version:** Alpha (v0.1.0-dev)
**Target:** Beta Release
**Last Updated:** 2026-04-30

---

## Overview

This document tracks planned features, known issues, and upgrade priorities for the Alpha-to-Beta transition. Beta focus is **UI improvements + feature upgrades**.

---

## 1. Tag/Caption System Upgrades

### 1.1 Tag Viewer with Frequency Counts (NEW FEATURE)
**Priority:** High
**Inspiration:** Civitai `TrainingImagesTagViewer` (Apache 2.0)
**Status:** Not started

Add a Tag Viewer panel to the dataset page that:
- Aggregates all tags across all images in a dataset
- Displays each unique tag with its occurrence count, sorted by frequency
- Supports search/filter within the tag list
- Allows multi-select of tags (for bulk operations)
- Shows a "None" sentinel for untagged images
- Shows image count and selected tag count badges

**Implementation notes:**
- Core logic: `flatMap` all caption .txt files -> count occurrences with reduce -> sort by frequency
- Frontend component with selectable chips/badges
- Requires a new API endpoint: `GET /api/dataset/{name}/tag-summary` that reads all .txt files and returns `{ tag: string, count: number }[]`

**Files to create/modify:**
- `frontend/app/dataset/[name]/` - new tag viewer component
- `api/routes/dataset.py` - new endpoint
- `services/tagging_service.py` - aggregation logic

### 1.2 Bulk Tag Operations (NEW FEATURE)
**Priority:** High
**Inspiration:** Civitai `TrainingEditTagsModal` + `TrainingImagesTagViewer` actions menu (Apache 2.0)
**Status:** Not started

Once the Tag Viewer exists, add bulk actions on selected tags:
- **Remove tags** - delete selected tags from ALL images in dataset (with confirmation)
- **Replace tags** - find-and-replace: show each selected tag with input for replacement, apply across all images

**Implementation notes:**
- Civitai's replace modal is ~85 lines, very clean pattern
- Needs API endpoints: `POST /api/dataset/{name}/tags/remove` and `POST /api/dataset/{name}/tags/replace`
- Should operate on .txt files on disk, not just in-memory

### 1.3 Upgrade Overwrite Mode from Bool to 3-Way
**Priority:** Medium
**Status:** Not started

Current: `append_tags: bool` (append or overwrite)
Target: `overwrite_mode: "ignore" | "append" | "overwrite"`

- **ignore** = skip files that already have labels
- **append** = add new tags to end of existing labels
- **overwrite** = replace existing labels entirely

**Files to modify:**
- `services/models/tagging.py` - change `append_tags: bool` to `overwrite_mode: Literal["ignore", "append", "overwrite"]`
- `frontend/app/dataset/[name]/auto-tag/page.tsx` - replace checkbox with segmented control
- `services/tagging_service.py` - handle new mode in tag processing
- `custom/tag_images_by_wd14_tagger.py` - update CLI args

### 1.4 Per-Image Visual Tag Editor
**Priority:** Low (Beta+)
**Status:** Not started

Visual tag editing per image: show tags as badge chips with X to remove, textarea to add new tags. Would require the frontend to read/write individual .txt caption files via API.

---

## 2. Checkpoint Training - Audit Results

### 2.1 Current State
Checkpoint training (full fine-tune) IS implemented in the backend:
- `TrainingMode.CHECKPOINT` exists in `services/models/training.py:142`
- Script selection works in `services/trainers/kohya.py:289-303`:
  - SD1.5 -> `fine_tune.py`
  - SDXL -> `sdxl_train.py`
  - Flux -> `flux_train.py`
  - SD3 -> `sd3_train.py`
  - Lumina -> `lumina_train.py`
  - Chroma -> `flux_train.py` (with --model_type chroma)
  - Anima -> `anima_train.py`
  - HunyuanImage -> LoRA only (correctly blocked)
- TOML generation correctly skips network args for checkpoint mode (`kohya_toml.py:183-184`)
- Validation catches high network_dim for non-checkpoint mode (`training.py:81`)

### 2.2 Known Issues - Checkpoint Training

**Issue CT-1: No checkpoint-specific validation**
- **Severity:** Medium
- **Location:** `api/routes/training.py:33-253`
- **Problem:** No validation specific to checkpoint mode (e.g., VRAM warnings, learning rate ranges differ from LoRA)
- **Fix:** Add checkpoint-specific validation in `validate_training_config_extended()`

**Issue CT-2: No UI indication of checkpoint mode limitations**
- **Severity:** Low
- **Location:** Frontend training config page
- **Problem:** Users can select checkpoint mode but may not understand VRAM requirements or that LoRA-specific fields (network_dim, network_alpha, etc.) are irrelevant in this mode
- **Fix:** Conditionally hide/disable LoRA fields when checkpoint mode selected, show VRAM warnings

**Issue CT-3: Lumina checkpoint training script may not exist**
- **Severity:** Needs verification
- **Location:** `services/trainers/kohya.py:297`
- **Problem:** Maps to `lumina_train.py` but needs verification that this script exists in sd_scripts
- **Fix:** Verify script existence, add runtime check

**Issue CT-4: Anima missing from checkpoint script map**
- **Severity:** High (bug)
- **Location:** `services/trainers/kohya.py:299`
- **Problem:** `script_map` for CHECKPOINT mode doesn't include Anima. It falls back to `fine_tune.py` (SD1.5 script) which is the wrong script. `anima_train.py` exists but isn't mapped.
- **Fix:** Add `ModelType.ANIMA: "anima_train.py"` to the checkpoint script_map

**Issue CT-5: CheckpointTrainingConfig frontend type incomplete**
- **Severity:** Medium
- **Location:** `frontend/components/checkpoint/CheckpointTrainingConfig.tsx:19`
- **Problem:** Type only includes `SD15 | SDXL | FLUX | SD3 | LUMINA` - missing SD3.5, Chroma, Anima
- **Fix:** Update TypeScript type to match backend ModelType enum

**Issue CT-6: Redundant TOML generation**
- **Severity:** Low (tech debt)
- **Location:** `services/trainers/kohya.py:51-78`
- **Problem:** TOML files generated in `trainer/runtime_store/` then copied to `config/`. Should generate once in final location.
- **Fix:** Generate directly to final destination

**Issue CT-7: WebSocket routes exist but are unused**
- **Severity:** Low (dead code)
- **Location:** `services/websocket.py:19-100`
- **Problem:** WebSocket endpoints defined but frontend uses HTTP polling due to VastAI Caddy proxy incompatibility. Code is never called.
- **Fix:** Document the reason or conditionally disable

**Issue CT-8: TODO comment references incomplete Node.js migration**
- **Severity:** Low (architectural clarity)
- **Location:** `frontend/lib/api.ts:794`
- **Problem:** Comment says "Node.js route exists at /jobs/training" and "TODO: Unify this in a future update" - indicates incomplete migration
- **Fix:** Decide on architecture direction and clean up or complete migration

---

## 3. Merging Tool - Audit Results

### 3.1 Current State
Both LoRA and Checkpoint merging are implemented:
- **LoRA merge:** `POST /api/utilities/lora/merge` - supports SD, SDXL, Flux, SVD
- **Checkpoint merge:** `POST /api/utilities/checkpoint/merge` - weighted merge with UNet-only option
- **LoRA resize:** `POST /api/utilities/lora/resize` - SVD-based rank reduction
- Frontend has a 3-tab UI (Merge LoRAs, Merge Checkpoints, Resize LoRA)

### 3.2 Known Issues - Merging Tool

**Issue MG-1: Alpha parameter silently ignored in LoRA resize**
- **Severity:** High (UX deception)
- **Location:** `services/lora_service.py:98-101`
- **Problem:** Frontend allows setting `newAlpha` but `resize_lora.py` doesn't support `--new_alpha`. Alpha is auto-calculated from SVD rank. User thinks they're controlling alpha but it's ignored.
- **Fix:** Either remove alpha input from frontend with explanation, or show it as read-only "will be auto-calculated"

**Issue MG-2: Parameter naming inconsistency**
- **Severity:** Medium (confusion)
- **Location:** `services/lora_service.py:298` vs `services/lora_service.py:400`
- **Problem:** LoRA merge uses `--save_precision` but checkpoint merge uses `--saving_precision` (different Kohya scripts expect different arg names). This is correct behavior but confusing in the codebase.
- **Fix:** Add comments explaining the naming difference is intentional per-script

**Issue MG-3: No subprocess timeout**
- **Severity:** Medium (reliability)
- **Location:** `services/lora_service.py:320-327, 428-435`
- **Problem:** `await process.communicate()` has no timeout. A hung merge process blocks the worker indefinitely.
- **Fix:** Use `asyncio.wait_for(process.communicate(), timeout=3600)` with proper cleanup

**Issue MG-4: No CUDA availability check**
- **Severity:** Medium
- **Location:** `services/lora_service.py:312, 420`
- **Problem:** If user requests `device: "cuda"` but no GPU is available, the merge script will fail with an unhelpful error.
- **Fix:** Check `torch.cuda.is_available()` before passing cuda device, return clear error

**Issue MG-5: SD3 merge not exposed**
- **Severity:** Low (missing feature)
- **Location:** `trainer/derrian_backend/sd_scripts/tools/merge_sd3_safetensors.py` exists but isn't wired up
- **Problem:** SD3 users can't merge models via the web UI
- **Fix:** Add SD3 merge option to the API and frontend

**Issue MG-6: Stdout not logged during merges**
- **Severity:** Low (observability)
- **Location:** `services/lora_service.py:320-327`
- **Problem:** Subprocess stdout is captured but never logged or surfaced
- **Fix:** Log at debug level, optionally stream to frontend for progress

**Issue MG-7: No merge progress reporting**
- **Severity:** Low (UX)
- **Problem:** Large checkpoint merges (2-7GB each) can take minutes with no progress indicator beyond a spinner
- **Fix:** Parse stdout for progress info, send via job status polling

**Issue MG-8: Missing output file existence check**
- **Severity:** Low
- **Location:** `services/lora_service.py:334`
- **Problem:** Reads file size without verifying output file actually exists
- **Fix:** Add `output_path.exists()` check before stat()

---

## 4. LoRA Training - Audit Results

### 4.1 Current State
LoRA training pipeline is solid (~99% functional). Audit performed on:
- API routes (`api/routes/training.py`)
- Training service (`services/training_service.py`)
- Kohya trainer (`services/trainers/kohya.py`)
- TOML generator (`services/trainers/kohya_toml.py`)
- Training config model (`services/models/training.py`)
- Job manager (`services/jobs/job_manager.py`)
- Frontend API client (`frontend/lib/api.ts`)

**Note on enum comparisons:** `ModelType` and `LoRAType` both inherit from `(str, Enum)`, so string comparisons like `config.model_type == "Flux"` work correctly. The `normalize_model_type` validator also handles frontend variants like `'SD15'` -> `'SD1.5'` cleanly.

### 4.2 Known Issues - LoRA Training

**Issue LT-1: wandb_key field exists but is never used**
- **Severity:** Medium (broken feature)
- **Location:** `services/models/training.py:168` (defined), nowhere referenced
- **Problem:** `TrainingConfig.wandb_key` is defined as a config field but never read by `kohya_toml.py` or set as an environment variable in `kohya.py`. W&B integration silently fails for users who set this.
- **Fix:** In `kohya.py` env setup, add `if self.config.wandb_key: env["WANDB_API_KEY"] = self.config.wandb_key` before launching the subprocess.

**Issue LT-2: enable_bucket force-enabled, user preference ignored**
- **Severity:** Low (functionality lock)
- **Location:** `services/trainers/kohya_toml.py:110`
- **Problem:** `dataset["enable_bucket"] = True # Force bucketing enabled` - hard-coded, ignores `self.config.enable_bucket`. Users can't disable bucketing even if they want to (e.g., for fixed-resolution datasets).
- **Fix:** Replace with `dataset["enable_bucket"] = self.config.enable_bucket`

**Issue LT-3: network_train_unet_only added in checkpoint mode**
- **Severity:** Medium
- **Location:** `services/trainers/kohya_toml.py:346`
- **Problem:** `network_train_unet_only` is added to args unconditionally in `_get_training_arguments()`, but it's a LoRA-only parameter. For checkpoint/fine-tune training (`fine_tune.py`, `sdxl_train.py`, etc.) this argument is invalid and may cause Kohya to error out or warn.
- **Fix:** Wrap in `if self.config.training_mode != TrainingMode.CHECKPOINT:` like the network args section already does at line 184.

**Issue LT-4: "Full" LoRA type semantic confusion**
- **Severity:** Low (UX/conceptual)
- **Location:** `services/models/training.py:36`, `services/trainers/kohya_toml.py:270-272`
- **Problem:** `LoRAType.FULL = "Full"` is exposed as a LoRA algorithm option (mapped to `lycoris.kohya` with `algo=full`), but "Full" means native fine-tuning (DreamBooth) which conceptually belongs in checkpoint training mode. Users selecting `training_mode=lora` + `lora_type=Full` get an unusual config that may conflict with `network_train_unet_only=True`.
- **Fix:** Either (a) auto-switch to checkpoint mode when Full is selected, (b) add validation warning, or (c) document clearly that Full is a LyCORIS algorithm distinct from `training_mode=checkpoint`.

**Issue LT-5: No LyCORIS algorithm-specific validation**
- **Severity:** Low (UX)
- **Location:** `api/routes/training.py:33-253`
- **Problem:** No validation warns users about algorithm-specific learning rate ranges (e.g., IA3 needs 5e-3 to 1e-2, much higher than standard LoRA), or about unused parameters (e.g., `conv_dim`/`conv_alpha` only apply to LoCon/LoHa, `factor` only to LoKR).
- **Fix:** Add per-algorithm warnings in `validate_training_config_extended()`.

**Issue LT-6: network_module field is misleading**
- **Severity:** Low (API clarity)
- **Location:** `services/models/training.py:197-199`
- **Problem:** `network_module` is a writable field with default `"networks.lora"` but its docstring says "derived from lora_type". The TOML generator always overrides it via `_get_network_config()`. API users who set this field will see it silently ignored.
- **Fix:** Either remove the field, make it computed/read-only, or actually respect user overrides.

---

## 5. Miscellaneous Bug Fixes for Beta

### 5.0 WandB / Logging UI Missing Entirely

**Issue UI-1: WandB and logging fields have no UI in the training form**
- **Severity:** Medium (feature exists in backend but is invisible to users)
- **Location:** `frontend/components/training/cards/*.tsx` (none render these fields)
- **Affected fields (all defined in schema, validation, defaults, presets, but no input UI):**
  - `wandb_key` (`hooks/useTrainingForm.ts:167`, `lib/validation.ts:138`)
  - `wandb_run_name` (`hooks/useTrainingForm.ts:175`, `lib/validation.ts:371`)
  - `log_with` (tensorboard/wandb backend selector)
  - `log_tracker_name`
  - `log_tracker_config`
  - `log_prefix`
  - `logging_dir`
- **Problem:** The fields exist throughout the entire data layer (Zod schema, defaults, types, preset save/load, frontend API client, backend Pydantic model) but no training tab component actually renders an input for them. Users have no way to configure W&B or even basic tensorboard logging from the UI - they'd have to manually edit a saved preset JSON to set these values. This is also why issue LT-1 (`wandb_key` not being read by backend) hasn't been noticed - nobody can set it in the first place.
- **Fix:** Add a "Logging" section to one of the existing cards (probably `SavingCard.tsx` or a new `LoggingCard.tsx`):
  - Dropdown: `log_with` (None / TensorBoard / WandB)
  - Text input: `logging_dir`
  - Text input: `log_prefix`
  - When `log_with === "wandb"`, conditionally show:
    - Password input: `wandb_key` (with link to https://wandb.ai/authorize)
    - Text input: `wandb_run_name`
    - Text input: `log_tracker_name` (project name)
- **Related:** Must be paired with LT-1 fix (wire wandb_key into the trainer env vars), otherwise the UI will exist but still won't actually log to W&B.

---

### 5.0.5 Dashboard Redesign - Missing Pages and Generic Look

**Issue UI-2: Dashboard incomplete and looks AI-generated**
- **Severity:** Medium (discoverability + first impression)
- **Location:** `frontend/app/dashboard/page.tsx`
- **Problem:** The dashboard only links to 9 pages but the project has 14+ user-facing routes. The visual design is also a generic "9 cards in a grid" layout that feels boilerplate. The navbar at `frontend/components/blocks/navigation/navbar.tsx` already groups routes into proper categories - the dashboard should mirror that organization.

**Pages currently on dashboard (9):**
`/models`, `/files`, `/dataset`, `/training`, `/calculator`, `/utilities`, `/settings`, `/docs`, `/about`

**Pages MISSING from dashboard:**
- `/checkpoint-training` - distinct from LoRA training
- `/dataset/auto-tag` - WD14/BLIP/GIT auto-tagging interface
- `/dataset/tags` - tag editor with gallery
- `/dataset-uppy` - alternate Uppy-based uploader
- `/huggingface-upload` - HF upload page
- `/changelog`
- `/models/browse` - Civitai downloader

**Visual design issues:**
- Flat 3-column grid with no hierarchy or grouping
- All cards look identical (same size, same layout) - no visual distinction between primary workflow vs settings
- Random rainbow icon colors (`text-cyan-400`, `text-pink-400`, etc.) without semantic meaning
- "Quick Start" section at the bottom is just a numbered list - could be a visual workflow diagram
- No indication of what's currently in progress (active jobs, recent datasets, etc.)

**Proposed redesign directions** (pick one or combine):

1. **Workflow-grouped layout (matches navbar structure):**
   - **Dataset Prep** section: Upload, Auto-Tag, Tag Editor, File Manager
   - **Training** section: LoRA Training, Checkpoint Training, Calculator
   - **Models** section: Download Models, Civitai Browser, HuggingFace Upload
   - **Tools** section: Merge/Resize Utilities, Settings
   - **Help** section: Docs, About, Changelog
   - Each section has a header, then its cards. More scannable than 9 random tiles.

2. **Workflow-driven hero section:**
   - Top of page: large "Start Training Workflow" with numbered visual steps (Dataset → Tag → Configure → Train → Upload), each clickable
   - Below: secondary cards for tools, settings, help
   - Treats the dashboard as a launcher for the primary workflow rather than an undifferentiated link grid

3. **Live status dashboard:**
   - Top: "Active Jobs" widget (any running training/tagging jobs with progress)
   - "Recent Datasets" widget (last 5 datasets, click to jump back in)
   - "Recent Models" widget (last 5 trained LoRAs, with quick HF upload action)
   - Below: traditional navigation cards
   - Makes the dashboard feel "alive" rather than static

4. **Sidebar nav + main content (most app-like):**
   - Dashboard becomes a real "home page" with actionable widgets
   - Persistent left sidebar replaces the navbar dropdown menus
   - Less reliant on the dashboard for discoverability since the sidebar always shows everything

**Recommended:** Combine #1 (workflow grouping) with a small version of #3 (active jobs widget at top). This is the lowest effort high-impact change - keeps the existing card pattern but groups them semantically and adds one "alive" element.

**Components to use** (per `frontend/CLAUDE.md`):
- `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent` from shadcn
- `Separator` between sections
- `Badge` for status indicators on the active jobs widget
- Existing semantic colors (`text-primary`, `text-muted-foreground`) instead of arbitrary rainbow

---

### 5.0.7 Listener/Request Cancellation Console Noise

**Issue UI-3: "Listener cancelled" / AbortError messages on page navigation**
- **Severity:** Low (mostly cosmetic, but masks real issues)
- **Locations:**
  - `frontend/lib/api.ts:121-188` - `pollJobLogs()`
  - `frontend/components/training/TrainingMonitor.tsx:115-133` - polling interval + visibilitychange listener
  - `frontend/components/DatasetUploader.tsx:231-232, 308-309` - upload AbortControllers (10min timeout)
  - `frontend/app/models/browse/page.tsx:93-148` - in-flight request cancellation
  - `frontend/hooks/useSettings.ts:80-81` - storage event listener
- **Problem:** When users navigate between pages, several legitimate cleanup paths fire:
  1. Polling intervals (`setInterval`/`setTimeout` recursion) get cleared
  2. AbortControllers cancel in-flight fetch requests, throwing `AbortError`
  3. `window`/`document` event listeners get removed
  4. The `pollJobLogs()` function uses a `stopped` boolean flag instead of AbortController, so the fetch continues to completion before its result is discarded - wasteful but not buggy
- **Why this is mostly fine:** All of these are CORRECT React cleanup behavior on unmount. The errors don't propagate to the UI. They're just console noise.
- **Why it's worth fixing:**
  1. Console noise hides REAL errors. Users can't tell signal vs noise.
  2. **Subtle potential bug in `pollJobLogs`:** If a fetch errors out (network blip) right as the user navigates away, the `catch` block checks `stopped` BEFORE deciding to call `onError`, but there's a tiny race window. If `onError` fires on an unmounted component, React logs "Can't perform a state update on an unmounted component" - harmless but ugly.
  3. The 10-minute upload timeout in `DatasetUploader.tsx` won't actually trigger normally, but if the page unmounts mid-upload the AbortError gets logged.

**Fixes:**
1. **Use AbortController in `pollJobLogs()`** instead of (or in addition to) the `stopped` flag:
   ```typescript
   const controller = new AbortController();
   // In poll(): fetch(url, { signal: controller.signal })
   // In stop(): controller.abort(); stopped = true;
   ```
   Then in the catch block, ignore `AbortError` explicitly:
   ```typescript
   catch (err) {
     if (err.name === 'AbortError') return; // Expected, no-op
     if (!stopped && onError) onError(err);
   }
   ```

2. **Same pattern for `DatasetUploader.tsx`** - cancel uploads on unmount via existing AbortController

3. **`TrainingMonitor.tsx` polling** - already correct, just verify the cleanup runs before any pending poll resolves

4. **Optional: Add a global fetch wrapper** that silently swallows `AbortError` to prevent any future occurrences from leaking to the console

**User-reported behavior:** "Listener things that cancel" appearing in console on page navigation. Not extension-based, page-based. Doesn't seem to cause UI issues but is concerning noise.

---

### 5.0.8 Training Log Polling - Updates Feel Inconsistent

**Issue UI-4: Training logs only update "when they feel like it"**
- **Severity:** Medium (core monitoring UX)
- **Locations:**
  - `frontend/lib/api.ts:840-907` - `trainingAPI.pollLogs()`
  - `frontend/components/training/TrainingMonitor.tsx:136-186` - log polling effect
  - `frontend/components/training/TrainingMonitor.tsx:99-133` - status polling effect (for comparison)
- **User-reported behavior:** "The training log section works in the training tabs page -- just that it doesn't tend to update as often it as it should as more of a 'I only poll when i feel like it'"
- **Problem:** Several issues conspire to make log updates feel sporadic:
  1. **Hardcoded 1000ms interval** at `api.ts:895` (`setTimeout(poll, 1000)`) - not configurable
  2. **Sequential, not interval-based:** the next poll only schedules AFTER the previous fetch completes. If the backend is slow (e.g. 800ms response), effective polling rate becomes ~1.8s per cycle. If the backend stalls, polling stalls with it.
  3. **No tab visibility check** in log polling - unlike the status polling (`TrainingMonitor.tsx:102` correctly checks `document.hidden`), the log poller in `api.ts:pollLogs` keeps polling when the tab is hidden. Browsers will then aggressively throttle background `setTimeout` calls (up to once per minute in Chrome), so when the user comes back to the tab, logs appear "frozen" until the next throttled poll fires.
  4. **Backend log buffering:** Python subprocess stdout is line-buffered by default, but Kohya/accelerate may buffer further. Even if the frontend polls perfectly, logs may not appear in the backend's log file until a buffer flushes.
  5. **Line-index pagination:** uses `?since=nextSince` - works fine, but if a poll fails partway through, lines can be missed (no retry of failed range)
- **Why it "feels inconsistent":** combination of #2 (no fixed cadence) + #3 (background throttling) + #4 (backend buffer flushes) means logs arrive in unpredictable bursts rather than a steady stream.

**Fixes (in order of impact):**

1. **Add visibility-aware polling to `pollLogs`:**
   ```typescript
   const poll = async () => {
     if (stopped) return;
     if (typeof document !== 'undefined' && document.hidden) {
       // Skip this poll, but reschedule so we resume when tab is focused
       timeoutId = setTimeout(poll, 1000);
       return;
     }
     // ... existing fetch logic
   };
   // Also wire up a visibilitychange listener to immediately poll on focus:
   document.addEventListener('visibilitychange', () => {
     if (!document.hidden && !stopped) poll();
   });
   ```

2. **Use fixed-cadence polling instead of sequential:** Replace recursive `setTimeout` with `setInterval`, OR keep `setTimeout` but record poll start time and schedule next from that:
   ```typescript
   const start = Date.now();
   await fetch(...);
   const elapsed = Date.now() - start;
   timeoutId = setTimeout(poll, Math.max(0, 1000 - elapsed));
   ```
   This guarantees ~1s cadence even if a poll takes 800ms.

3. **Make poll interval configurable** - accept an optional `intervalMs` parameter so the training monitor can poll faster (e.g. 500ms) for active jobs and slow down (5s) for queued ones.

4. **Backend log flushing:** Ensure the Kohya subprocess wrapper in `services/trainers/kohya.py` runs Python with `-u` (unbuffered) and/or sets `PYTHONUNBUFFERED=1` in the env. This is the single biggest "why don't I see logs?" cause.

5. **Long-term: switch to Server-Sent Events (SSE):** A `GET /api/training/logs/{jobId}/stream` endpoint that streams new lines as they arrive would eliminate polling entirely. Works through Caddy proxies (unlike WebSockets per CT-7) since SSE is just chunked HTTP. The client side would use `EventSource` with automatic reconnect.

6. **Combine UI-4 fix with UI-3 fix:** The AbortController refactor in UI-3 should land in the same PR, since both touch `pollLogs`.

**Acceptance criteria:**
- New log lines appear within 1-1.5s of being written (when tab is focused)
- Polling pauses when tab hidden, resumes immediately on focus
- No "frozen" log feeling after returning from another tab
- Console shows no AbortError noise on page navigation (combined with UI-3)

---

### 5.0.9 Training Monitor — tqdm ETA and Step Progress Parsing

**Issue UI-5: No ETA, step count, or percentage in the training monitor**
- **Severity:** Medium (long-term imperative — fine for short runs, painful for multi-hour ones)
- **Location:** `frontend/components/training/TrainingMonitor.tsx` + log polling
- **User-reported:** Epoch numbers show but ETA/step progress don't. Fine for a 32-image 4090 run, a problem for anything longer.

**Why it's missing:** Kohya outputs timing via tqdm progress bars in stdout. A typical line looks like:

```
steps: 100%|██████████| 320/320 [08:23<00:00,  1.57s/it]
```

The `[elapsed<remaining, it/s]` is all the data we need but we currently pass log lines through as raw text without parsing them.

**Fix:** Add a tqdm line parser to the log polling pipeline:
- Regex to detect tqdm progress lines: `/(\d+)%\|.*\|\s*(\d+)\/(\d+)\s*\[(\d+:\d+)<(\d+:\d+),\s*([\d.]+)it\/s\]/`
- Extract: current step, total steps, elapsed, remaining (ETA), it/s
- Surface in TrainingMonitor as: progress bar, ETA countdown, steps/s throughput
- Epoch lines (`Epoch X/Y`) already parse fine — keep those, add step-level detail alongside

**Implementation notes:**
- Parser lives in a utility function, tested independently
- TrainingMonitor gets a new "progress" state slot separate from raw log lines
- tqdm output format is consistent across Kohya versions — safe to rely on
- Only activate the parser when a training job is active (don't waste cycles on idle log polling)

---

### 5.1 Preset Optimizer Args Contamination

**Issue PR-1: optimizer_args field picks up general training args from community presets**
- **Severity:** Medium (causes cryptic training failures)
- **User-reported:** Citron's Adafactor preset stuffed precision/device args (`state_storage_dtype=bfloat16 state_storage_device=cuda` style) into `optimizer_args`. Adafactor then threw `ValueError: not enough values to unpack` because those aren't valid optimizer args.
- **Root cause:** Community presets (and possibly our own) miscategorise general training args as optimizer_args. These are actually top-level training fields (`mixed_precision`, `fp8_base`, etc.) that got bundled into the freetext `optimizer_args` blob.
- **Workaround:** Manually clear `optimizer_args` before training if switching optimizers or loading community presets.

**Fixes:**
1. **Preset cleanup** — audit all bundled presets, move any precision/device args out of `optimizer_args` into proper top-level fields
2. **UX warning** — when `optimizer_args` is non-empty and optimizer type changes, warn the user: "These args may be specific to a different optimizer — clear them?"
3. **Validation** — before training starts, validate that `optimizer_args` entries look like actual optimizer args (key=value pairs that the selected optimizer recognises), not training-level settings
4. **Community preset naming** — if a preset bundles optimizer-specific args, the name should make that clear (e.g. "Citron Adafactor - SDXL" not just "Citron")

---

### 5.1 HuggingFace Upload - Form State Doesn't Persist

**Issue HF-1: HF upload form loses all data on page navigation**
- **Severity:** Medium (significant UX pain point)
- **Location:** `frontend/app/huggingface-upload/page.tsx:11-25`
- **Problem:** The HF upload page uses `useState` for all form fields (token, owner, repo name, repo type, commit message, remote folder, etc.). When the user navigates away from the page and back, all data is lost. Users uploading multiple LoRAs/models in a session have to re-enter the same information repeatedly (token, owner, etc.) - reportedly 20+ times.
- **Fix:** Migrate form state to a Zustand store with `persist` middleware (or use localStorage directly), at minimum for:
  - `hfToken` (consider security: maybe sessionStorage instead of localStorage)
  - `owner`
  - `repoType`
  - `commitMessage` (default)
  - `createPR` preference
- **Note:** `selectedFiles` and `uploading`/`uploadResult` should NOT persist (they're per-upload state)
- **Security consideration:** HF tokens are sensitive. Consider:
  - Using sessionStorage (clears on browser close) instead of localStorage
  - Adding a "Remember token" opt-in checkbox
  - Or storing the token via the existing settings management system that already handles HF tokens (per STATUS.md, settings management is "Working")
- **Better approach:** The existing settings page already manages HF tokens. The upload page should READ the token from settings rather than asking for it again. Only fields like owner/repo would need their own persistence.

---

## 6. Feature Priority Matrix (Beta)

### Must Have (Alpha -> Beta gate)
| Feature | Category | Effort | Status |
|---------|----------|--------|--------|
| Fix Anima checkpoint script mapping (CT-4) | Bug Fix | Tiny | ✅ Done (pre-existing) |
| Fix network_train_unet_only in checkpoint mode (LT-3) | Bug Fix | Tiny | ✅ Done 2026-04-30 |
| Wire up wandb_key environment variable (LT-1) | Bug Fix | Tiny | ⏳ Next session |
| Add WandB/Logging UI section (UI-1) | New Feature | Small | ⏳ Next session |
| Dashboard redesign with all routes (UI-2) | UX | Medium | 🔵 Design work, deferred |
| HF upload form persistence (HF-1) | UX/Bug Fix | Small | ✅ Done 2026-04-30 |
| Tag Viewer with frequency counts | New Feature | Medium | ⏳ Not started |
| Bulk tag remove/replace | New Feature | Medium | ⏳ Not started |
| Fix alpha parameter UX in LoRA resize (MG-1) | Bug Fix | Small | ✅ Done 2026-04-30 |
| Add subprocess timeouts to merge operations (MG-3) | Reliability | Small | ✅ Done 2026-04-30 |
| CUDA availability check for merges (MG-4) | Error Handling | Small | ✅ Done 2026-04-30 |
| Update CheckpointTrainingConfig types (CT-5) | Bug Fix | Tiny | ✅ Done 2026-04-30 |

### Should Have (Beta quality)
| Feature | Category | Effort | Status |
|---------|----------|--------|--------|
| 3-way overwrite mode for tagging | Enhancement | Small | ⏳ Not started |
| Checkpoint-specific validation (CT-1) | Enhancement | Small | ⏳ Not started |
| Hide LoRA fields in checkpoint mode (CT-2) | UX | Medium | 🚫 N/A — separate pages, no unified form |
| Merge progress reporting (MG-7) | UX | Medium | ⏳ Not started |
| SD3 merge support (MG-5) | Feature | Small | ⏳ Deferred (part of #342) |
| Clean up redundant TOML generation (CT-6) | Tech Debt | Small | ⏳ Not started |
| Respect enable_bucket user setting (LT-2) | Bug Fix | Tiny | ✅ Done 2026-04-30 |
| LyCORIS algorithm-specific validation (LT-5) | Enhancement | Medium | ⏳ Not started |
| Clarify "Full" LoRA type semantics (LT-4) | UX | Small | ⏳ Not started |
| Silence AbortError console noise (UI-3) | Polish | Small | ⏳ Not started |
| Fix training log polling cadence + visibility (UI-4) | UX/Bug Fix | Small | ✅ Done 2026-04-30 |
| Add PYTHONUNBUFFERED to Kohya subprocess (UI-4 part) | Bug Fix | Tiny | ⏳ Next session |
| MG-2: Document save_precision naming difference | Code Clarity | Tiny | ✅ Done 2026-04-30 |

### Nice to Have (Beta+)
| Feature | Category | Effort |
|---------|----------|--------|
| Per-image visual tag editor | New Feature | Large |
| Caption editor with search highlighting | Enhancement | Medium |
| Merge presets/templates | UX | Medium |
| Merge dry-run/preview mode | Feature | Medium |
| EQ VAE support - SDXL (VAE-EQ-1) | Advanced Feature | Small |
| Qwen-Image VAE reflection padding - Anima (VAE-EQ-2) | Advanced Feature | Small (needs research) |
| Batch Downloader (BD-1) | New Feature | Medium |
| Training monitor tqdm ETA parsing (UI-5) | Enhancement | Small |

---

## 6.5 Batch Downloader

**Issue BD-1: Model sourcing is fragile and region-dependent**
- **Priority:** Beta+ (nice to have, but solves a real accessibility problem)
- **Status:** Not started
- **Motivation:** Civitai's API geoblocks certain regions (UK datacenter IPs get 451'd due to UK Online Safety Act compliance). Users shouldn't need a working Civitai API to download models — they should be able to paste any link and have it work.

### Concept

A dedicated **Batch Downloader** page where users paste a list of URLs or magnet links (one per line) and the app routes each to the right downloader automatically:

| Link type | Downloader |
|-----------|-----------|
| `https://huggingface.co/...` | `huggingface-cli` / `hf_hub_download` |
| `magnet:?xt=...` | `aria2c` |
| `https://civitai.com/...` | Existing Civitai API (with token) |
| Any other HTTP/HTTPS | `aria2c` (better than wget for resumable downloads) |
| Google Drive | `gdown` |

### UI Design

- Large textarea: paste URLs one per line
- Global destination dropdown: Models / LoRAs / VAEs / Dataset / Output
- Per-line destination override (optional, can skip for v1)
- Progress list showing each download's status as it runs
- aria2c is already installed on VastAI/RunPod instances (required by existing workflow)

### Inspiration

Inspired by the A1111 `BatchLinks` extension which used `#destination` hashtag syntax to route downloads. Our version replaces the hashtag hack with a proper destination dropdown — cleaner UX, same flexibility.

### Implementation notes
- Backend: new `POST /api/utilities/batch-download` endpoint that accepts a list of `{url, destination}` objects, spawns aria2c/hf-cli/gdown as appropriate per URL, streams progress back
- Frontend: new page at `frontend/app/batch-download/page.tsx`
- aria2c already present on instances — no new provisioning needed
- gdown may need `pip install gdown` added to requirements
- No torrent tracker/indexer integration — users provide their own links. Completely neutral technology.

---

## 7. EQ VAE / Reflection Padding Support

### 7.1 Background

EQ VAEs (e.g. `KBlueLeaf/EQ-SDXL-VAE`, `Anzhc/MS-LC-EQ-D-VR_VAE`) require reflection padding on their Conv2d layers instead of the default zero padding. Without it, they produce edge artifacts. The fix is applied post-load by mutating `module.padding_mode = "reflect"` on every Conv2d with non-zero padding.

Reference: https://github.com/kohya-ss/sd-scripts/issues/2189

### VAE-EQ-1: SDXL EQ VAE Support

**Priority:** Nice to Have (Beta+)  
**Status:** Not started

Jelosus2's fork of sd-scripts has a clean 15-line implementation (`library/sdxl_train_util.py`):

```python
def vae_with_reflection(vae):
    for module in vae.modules():
        if isinstance(module, torch.nn.Conv2d):
            pad_h, pad_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            if pad_h > 0 or pad_w > 0:
                module.padding_mode = "reflect"
    return vae
```

And a `--vae_reflection` CLI arg in `train_network.py`.

**Our vendored backend** (`trainer/derrian_backend/sd_scripts/`) does NOT have this patch. It only needs to be ported to `library/sdxl_train_util.py` + add `--vae_reflection` arg.

**Files to change:**
- `trainer/derrian_backend/sd_scripts/library/sdxl_train_util.py` — add `vae_with_reflection()` + call in `load_target_model()`
- `trainer/derrian_backend/sd_scripts/train_network.py` — add `--vae_reflection` arg
- `services/models/training.py` — add `vae_reflection: bool = False`
- `services/trainers/kohya_toml.py` — write `vae_reflection = true` for SDXL when set
- Frontend: add checkbox in training form (SDXL-only, shown conditionally)

### VAE-EQ-2: Anima Qwen-Image VAE Reflection Padding

**Priority:** Nice to Have (Beta+)  
**Status:** Needs research

The Qwen-Image VAE used by Anima is a different architecture (16-channel, 8x spatial downscale) loaded via `library/qwen_image_autoencoder_kl.py`, not `sdxl_train_util.py`. Whether reflection padding applies and what effect it has on Anima training quality needs verification.

**Research needed:** Check if Circlestone Labs' Anima documentation mentions EQ VAE or reflection padding. Check `qwen_image_autoencoder_kl.py` Conv2d layer padding values to see if the patch would even touch anything meaningful.

### VAE-EQ-3: HakuLatent — Long-Horizon Research Item

**Priority:** Future / Research only  
**Status:** Track, do not implement yet  
**Reference:** https://github.com/KohakuBlueleaf/HakuLatent (Apache-2.0)

HakuLatent is KohakuBlueleaf's Python framework for *training* VAEs with EQ (equivariance) regularization — it is the upstream source of the EQ VAEs that VAE-EQ-1 and VAE-EQ-2 are about consuming. It applies rotation/scale/crop/affine transforms during VAE training to produce more geometry-consistent latent spaces.

**What it is not:** A Kohya SS plugin, a LoRA tool, or anything with a web UI integration surface. It produces better VAEs; our job is using those VAEs correctly (reflection padding).

**Plausible future connection:** If the project ever adds VAE fine-tuning (training or adapting a VAE on a custom dataset — e.g. improving a domain-specific VAE for a character artist's style), HakuLatent would be the correct library to wrap. This is a research-grade, long-tail feature.

**Current status of the library:** Active WIP, no stable releases, explicit TODO list with unfinished trainers. Not ready to integrate even if we wanted to.

**When to revisit:** After VAE-EQ-1 and VAE-EQ-2 land and users start asking "can I train my own EQ VAE here?"

---

### 7.2 Session Notes (2026-04-15) — Anima Audit

During an Anima support audit, the following bug was found and **fixed**:

**FIXED: `networks.lora` → `networks.lora_anima` bug (`services/trainers/kohya_toml.py`)**
- `_get_network_config()` was returning `networks.lora` for all standard LoRA, including Anima
- Anima requires `networks.lora_anima` (confirmed by official docs + real training metadata)
- Using `networks.lora` would train wrong layer sets entirely — silently broken output
- Fix: added `ModelType.ANIMA` check in both the LoRA case and the default fallback

**Anima support status post-fix:** Complete for basic training. All required args (qwen3 path, AE path, per-layer LRs, timestep/flow args, blocks_to_swap) are wired. Default tokenizer configs (`configs/t5_old/`, `configs/qwen3_06b/`) are bundled. Training scripts exist. One minor gap: `optimizer_args` UI description says "JSON" but Kohya expects space-separated `key=value` pairs — relevant for CAME users needing `state_storage_dtype=bfloat16 state_storage_device=cuda` for 4070-class GPUs.

---

## 7.9 Hardcoded Presets Not Wired Into Training Submit Flow

**Priority:** High — blocks actual training  
**Status:** Confirmed broken 2026-04-30  

### What we know

Presets hardcoded in `frontend/hooks/useTrainingForm.ts` (Citron's Illustrious, Citron's PDXL, Citron's Anima etc.) populate form fields visually but do NOT properly hook into the training submit flow. When you load one and hit Train:

- No request reaches FastAPI
- No logs appear (backend is completely silent)
- No 422, no 500, nothing — complete silence
- The button appears to work but nothing happens

This is NOT a backend config field mismatch issue — it's the frontend never actually sending the training request at all.

### Why it happened

Presets were hardcoded into `useTrainingForm.ts` as a quick solution ("we needed it to work RIGHT NOW") rather than going through the proper preset save/load system. The values populate the form display but don't correctly feed into whatever the submit handler reads to build the training payload.

### Fix needed

- Audit how `useTrainingForm.ts` applies hardcoded preset values to form state vs how the submit handler reads form state
- Either: fix the wiring so hardcoded presets properly update the form state the submit handler uses
- Or better: migrate all hardcoded presets out of `useTrainingForm.ts` into proper JSON files in `presets/` and use the existing preset load system
- The JSON migration is the RIGHT fix — hardcoding presets in a hook is the root cause of this whole class of bugs

### Related

- PR-1 (optimizer_args contamination) — same root cause, presets in wrong place
- Preset UX architecture filtering (in MEMORY.md backlog) — filter by model_type, only possible once presets are proper JSON

---

## 8. Anima Deep Dive — Research Session Needed

**Priority:** Beta (before Anima is considered properly supported)  
**Status:** Attempted training, hit size mismatch error — config fields not properly understood  

### What we know

- `networks.lora_anima` is correctly wired (fixed previously)
- Training script exists in sd-scripts
- Attempted a real training run — failed with: `size mismatch for layers.27.mlp.gate_proj.weight: copying a param with shape torch.Size([3072, 1024]) from checkpoint, the shape in current model is torch.Size([22016, 4096])`
- This suggests something is being loaded as an LLM/text encoder that is the wrong size or wrong architecture entirely

### What we don't know

- Exactly which files Anima's training script expects and from where
- Whether the LLM component is embedded in the base model or must be provided separately
- Whether our UI fields map correctly to what `anima_train.py` actually expects
- Whether `clip_skip`, `gemma2`, and other fields are being passed/ignored correctly for Anima

### Research plan for next dedicated session

1. Read `trainer/derrian_backend/sd_scripts/anima_train.py` argument list in full
2. Find a real working Anima training config from the community (Circlestone Labs discord, ArcEnCiel community)
3. Map actual required args → our UI fields → fix any mismatches
4. Document the correct file structure for an Anima training setup (base model, VAE, text encoder, tokenizer configs)
5. Add Anima-specific validation to catch wrong file types before training starts

### Note on `ARCHITECTURE.md`

Future Claude needs context it currently has to re-derive every session. A dedicated session to document the backend config fields per model type would save significant time. Proposed: `docs/ARCHITECTURE.md` covering:
- What each model type (SD15, SDXL, Flux, SD3, Lumina, Anima) actually requires
- Which UI fields map to which CLI args in Kohya
- Which fields are model-type-specific vs universal
- Known gotchas per model type

This file gets `grep`ped at session start instead of guessing from training data.

---

## 8. Session Notes (2026-04-30) — Priority for Next Week

### URGENT: Full Training Logs (next 1-2 sessions)
Training logs are essentially broken in production — the monitor dies early, stdout doesn't flush, and users have no idea what's happening mid-run. Confirmed today that a training can run for HOURS with zero visible progress (Adafactor + 305 images + 10 epochs on a 4090 took well over an hour with zero UI feedback). This is the single most important UX fix for beta.

Fixes needed in priority order:
1. **PYTHONUNBUFFERED=1** on Kohya subprocess (UI-4 backend part) — single biggest impact
2. **tqdm line parser** (UI-5) — surface step count, ETA, it/s in the monitor
3. **Training monitor reconnect** — if the monitor component dies, user should be able to re-attach to a running job by job ID without refreshing the whole page

### Calculator Enhancement — Optimizer-Aware Time Estimation
The step calculator currently uses basic Kohya math. Proposal: make it optimizer-aware so it can give rough time estimates based on:
- Optimizer choice (Adafactor is slower per step than AdamW8bit, Prodigy is variable)
- Dataset size × repeats × epochs → total steps
- Batch size effect on step count
- Resolution effect on VRAM and speed

Not exact science but "rough estimate with caveats" is infinitely more useful than nothing. Would have saved a lot of "is this an all-nighter?" uncertainty today.

### Hardcoded Presets Migration (section 7.9)
High priority — confirmed blocks training silently. Migrate all presets from `useTrainingForm.ts` into proper JSON files in `presets/`. This fixes the silent training failure AND the PR-1 optimizer_args contamination in one shot.

---

## 9. Attribution Requirements

When implementing features inspired by Civitai's codebase, add to `ATTRIBUTIONS.md`:

```markdown
## Civitai
**Repository:** [civitai/civitai](https://github.com/civitai/civitai)
**License:** Apache License 2.0
**Usage:** Tag viewer UI patterns, bulk tag operation workflows, and auto-label configuration UX

The following features were inspired by Civitai's training interface:
- Tag frequency viewer with search and multi-select
- Bulk tag remove/replace operations
- 3-way overwrite mode (ignore/append/overwrite) for auto-labeling
```

---

## 9. Session Notes

### 2026-04-30 — Beta Bug Bash

**Completed this session (commits on `dev`):**
- Next.js 15 → 16 upgrade (PR #355, merged to main)
- PostCSS CVE-2026-41305 patch
- Dev-branch provisioning scripts for VastAI and RunPod
- CT-5: CheckpointTrainingConfig TypeScript type now includes SD35/CHROMA/ANIMA
- MG-1: Removed deceptive alpha input from LoRA resize (resize_lora.py auto-calculates it)
- MG-2: Commented the --save_precision vs --saving_precision naming difference
- MG-3: asyncio.wait_for() timeouts on all merge/resize subprocesses (resize=30min, merges=1hr)
- MG-4: _validate_device() CUDA availability check before any subprocess gets --device cuda
- UI-4 (#346): pollLogs fixed-cadence + visibility-aware polling + visibilitychange listener
- HF-1 (#344): HF upload page pre-fills token from saved settings; owner/repoType persist in localStorage
- LT-2: enable_bucket now reads from config instead of hardcoded True
- LT-3: network_train_unet_only guarded to False in checkpoint mode

**Up next:**
- LT-1 + UI-1 (#343): Wire wandb_key as WANDB_API_KEY env var + add LoggingCard to training form
- PYTHONUNBUFFERED=1 on Kohya subprocess (UI-4 backend part)
- MG-5: SD3 merge (deferred but part of #342)
- #349: Upload progress indicator (needs XHR refactor, waiting a few days)
- Tag Viewer + Bulk tag ops (larger feature work)

**Ease assessments (2026-05-07):**
- **#349 upload progress** — Easy. Infrastructure already exists (progress state, file status tracking). Two steps: (1) swap remote zip upload from `fetch` to `XMLHttpRequest` for `onprogress` events, (2) add `<Progress>` component to the UI. ~30-40 lines. Deferred, not tonight.
- **#343 WandB logging UI** — Easy-Medium. Entire data layer exists (Zod schema, types, defaults, validation, API client, backend Pydantic). Only missing: rendered inputs. Add a `LoggingCard` with `log_with` dropdown + conditional WandB fields. One focused session. Pair with LT-1 backend wire-up.
- **#340 tag viewer + bulk ops** — Medium. Part 1 (tag viewer): new `GET /api/dataset/{name}/tag-summary` endpoint + frontend chip/badge component. Logic is simple (`flatMap` → `reduce` → sort). Part 2 (bulk ops) builds directly on Part 1. Larger than 343 but very well scoped. Good candidate for a web session.
- **#347 dashboard redesign** — Large. Visual design decisions + missing pages + workflow grouping hierarchy. Needs a dedicated focused session with component work. Not a quick fix — confirmed big lift.

**CT-2 closed as N/A:** `/training` and `/checkpoint-training` are already separate pages — no unified form exists where LoRA fields would need to be hidden.

---

## 10. Notes (Original)

- Checkpoint training backend appears functional but needs real-world testing with actual full fine-tune runs
- Merging tool is mostly solid - the issues are UX and reliability, not correctness
- The tag system upgrades are the highest-impact changes for Beta since they directly improve the dataset preparation workflow
- All Civitai-inspired features are UI patterns only - we do NOT use their cloud orchestrator, S3 upload, or SignalR approach

---

## 11. Ecosystem Integration

The goal is a healthy, interconnected set of tools running on the same VastAI/RunPod instance — not a monolith, but a set of apps that are aware of each other and hand off naturally. Training a LoRA should flow directly into testing it. Uploading a model should be one click, not a separate workflow.

---

### 11.1 ComfyUI Frontend Integration

**Priority:** Beta+ (after core beta bugs closed)  
**Status:** Template acquired, architecture planned. **Shipping decision: all-in-one.**
**Source:** v0-generated template, repo at `duskfallcrew/KNX-ComfyUI`, to be adapted — not dropped in wholesale

#### Shipping decision (2026-05-09)

ComfyUI tab ships **bundled in the main app**. No install flag, no optional clone, no separate repo dependency. Reasons:

- Next.js code-splits routes — the ComfyUI bundle costs **zero RAM until someone actually navigates to the tab**
- The only RAM that matters is the ComfyUI Python backend, which the user controls independently
- Simpler onboarding: one install, one URL, one app
- "Throw wide the gates" — don't gate features behind install complexity

**UX for users without ComfyUI:**
- `/comfyui` tab is always visible in the navbar
- If the backend isn't reachable, the page shows a friendly skeleton/disconnected state (not an error)
- Settings page has a "ComfyUI URL" field (default: `http://localhost:8188`) so users who already have ComfyUI running somewhere can point the app at it — no re-install needed
- Connection status badge on the tab so it's obvious at a glance whether it's live



#### What the template provides

A complete, well-structured ComfyUI client layer:

| Layer | Files | Notes |
|-------|-------|-------|
| API client | `lib/comfy/client.ts` | REST + WebSocket, auto-reconnect, all ComfyUI endpoints |
| Workflow builders | `lib/comfy/workflows/` | txt2img, img2img, upscale — clean function-based API |
| State stores | `lib/stores/` | Zustand: connection, generation params, queue |
| UI components | `components/comfy/` (14 files) | All shadcn/ui — prompt editor, model selector, LoRA stack, sampler settings, image gallery, queue display, dimension picker, seed control, batch controls, denoise slider, upscale settings, workflow tabs, image input, connection status |
| Types | `lib/comfy/types.ts` | Full typing for all ComfyUI API responses |

All UI components use shadcn/ui — zero styling conflicts with the existing project.

#### The one real integration problem

The client defaults to `http://localhost:8188` and opens a WebSocket directly from the browser. On VastAI/RunPod through Cloudflare tunnel, **the browser cannot reach port 8188** — the tunnel only exposes port 3000.

**Fix:** Add a `/comfyui` reverse proxy to `server.js` (same pattern as the existing FastAPI proxy). Change the client's default `baseUrl` to `/comfyui`. One env var (`COMFYUI_PORT`, default `8188`) controls the target. WebSocket upgrades need a second handler in `server.on('upgrade')`.

#### Implementation sequence

**COMFY-1: Proxy layer (prerequisite for everything else)**
- `frontend/server.js` — add `/comfyui` HTTP proxy block (mirror the FastAPI proxy pattern)
- `frontend/server.js` — add `/ws/comfyui` WebSocket upgrade handler
- Env var: `COMFYUI_PORT=8188` (add to `start_services_vastai.sh`, `start_services_runpod.sh`, `restart.sh`)
- `lib/comfy/client.ts` — change default `baseUrl` from `http://localhost:8188` to `/comfyui`

**COMFY-2: Drop in the library layer**
- Copy `lib/comfy/` into `frontend/lib/comfy/`
- Copy `lib/stores/` into `frontend/lib/stores/` (check for Zustand dep — add to package.json if missing)
- Copy `hooks/use-generation.ts` into `frontend/hooks/`

**COMFY-3: UI components**
- Copy `components/comfy/` into `frontend/components/comfy/`
- New page: `frontend/app/comfyui/page.tsx` (adapt from template's `app/page.tsx`)
- Add ComfyUI to navbar under a new "Generate" section (or alongside Tools)
- Add to `server.js` `nodeApiPrefixes` if any Next.js API routes are needed

**COMFY-4: Settings integration**
- Add ComfyUI URL field to existing settings page (`frontend/app/settings/page.tsx`)
- Read from settings store rather than hardcoding; show connection status badge

**COMFY-6 (long-horizon): Custom node plugin system**
- Custom node packs map to UI component "plugins" — similar to A1111's extension system
- Installing a custom node pack (e.g. ControlNet, Impact Pack) surfaces a friendly UI panel for it rather than raw node inputs
- Foundation already exists: `lib/comfy/types.ts` has full ComfyUI API typing to build node→component mapping on
- Research: Dataset-Tools already has `lib/comfyui-node-registry.ts` and `lib/comfyui-github-search.ts` — may be the starting point
- **Do not design this until COMFY-1 through COMFY-4 are shipped and stable**

**COMFY-5 (dream feature): "Test in ComfyUI" post-training shortcut**
- When a training job completes, show a "Open in ComfyUI" button on `TrainingMonitor`
- Clicking it navigates to `/comfyui` with the trained LoRA pre-loaded into the LoRA stack
- Requires: COMFY-1 through COMFY-4 complete + ComfyUI actually running on the instance

#### Notes
- ComfyUI the **backend** is not bundled — the user installs and runs it separately. The frontend tab is always present; it shows a disconnected skeleton state if the backend isn't reachable. The provisioning scripts (`vastai_setup.sh`, `provision_runpod.sh`) may optionally install it, but that's a separate decision.
- The template has workflow builders for `inpaint`, `controlnet`, and `adetailer` types referenced in `types.ts` but not yet built in `workflows/`. Those are future scope.
- Check whether `zustand` is already a dep before adding it — the stores use it.

---

### 11.2 Dataset Tools Integration

**Priority:** Beta+ (after COMFY-1 proxy pattern is proven)  
**Status:** App is working and maintained — integration plan drafted  
**Source:** `C:\Users\dusk\Development\Dataset-Tools` / [Ktiseos-Nyx/Dataset-Tools](https://github.com/Ktiseos-Nyx/Dataset-Tools) (same org)  

#### What it is

Dataset Tools is a **local-first image and model browser** with deep AI metadata extraction. Key capabilities not currently in the Trainer:

| Capability | Value to the Trainer ecosystem |
|------------|-------------------------------|
| Image metadata viewer | Inspect reference images before training — see what settings produced them (A1111, ComfyUI, NovelAI, Fooocus, InvokeAI, DrawThings, SwarmUI, etc.) |
| Safetensors inspector | View training metadata embedded in a just-trained LoRA — steps, dataset hash, network args — without leaving the UI |
| ComfyUI workflow viewer | Show the full node graph for ComfyUI-generated reference images; pairs with COMFY-5 |
| Thumbnail viewport | Fast thumbnail browsing with server-side `sharp` WebP generation + disk cache (`.thumbcache/`) |
| Custom node classifier | Identifies ComfyUI custom nodes in a workflow — which are built-in vs. which require extensions |

#### Tech stack compatibility

- Next.js 16 + React 19 — **identical to Trainer** ✓
- shadcn/ui + Radix — **identical to Trainer** ✓
- Pure Node.js API routes — no Python required for the web app
- The `dataset_tools/` Python package in the repo is a **separate CLI tool**, not a web dependency

#### Integration architecture

Same proxy approach as ComfyUI (section 11.1). Dataset Tools runs as a second Next.js process on its own port; the Trainer's `server.js` proxies `/dataset-tools/*` to it.

**DT-1: Proxy layer**
- `frontend/server.js` — add `/dataset-tools` HTTP proxy block, pointing to `http://127.0.0.1:${DATASET_TOOLS_PORT}`
- Env var: `DATASET_TOOLS_PORT=3001` (add to all startup scripts)
- `start_services_vastai.sh` — start Dataset Tools process (`cd /workspace/Dataset-Tools && npm start`)
- Same for `start_services_runpod.sh` and `restart.sh`
- Add to provisioning scripts: `git clone` + `npm install` + `npm run build` of Dataset-Tools repo

**DT-2: Navbar link**
- Add "Dataset Tools" entry to the trainer navbar under a new "Ecosystem" section (or alongside "Files")
- Links to `/dataset-tools`

**DT-3: Handoff buttons (the good stuff)**
- **"Inspect in Dataset Tools"** on the files page — deep-link to Dataset Tools with the current folder pre-set
- **"Inspect LoRA"** on training completion — opens the trained `.safetensors` directly in Dataset Tools' safetensors panel
- **"View reference metadata"** in the dataset image gallery — opens a selected image in Dataset Tools' metadata panel

These handoffs are URL-based (no shared state stores), keeping the projects loosely coupled.

**DT-4: Shared folder awareness**
- Dataset Tools has a `settings.currentFolder` that the user sets manually
- Ideally the Trainer can deep-link with `?folder=/path/to/dataset` so Dataset Tools opens to the right place
- Check if Dataset Tools' settings API accepts a folder override via query param, or whether we need to add it

#### Notes
- Dataset Tools has its own settings system and thumbnail cache — these are self-contained, no conflict with Trainer settings
- The Python `dataset_tools/` CLI is a separate tool; ignore it for web integration
- Dataset Tools' `app/api/fs/route.ts` restricts file access to a configured base folder — on VastAI the default base should be `/workspace`
- `.thumbcache/` directory generates WebP thumbnails via `sharp` — on VastAI this lives inside the Dataset-Tools repo directory, which is fine
- **Dataset-Tools already has ComfyUI integration:** `ComfyUIWorkflowViewer.tsx`, `app/api/comfyui-nodes/route.ts`, `lib/comfyui-node-registry.ts`, `lib/comfyui-github-search.ts`. These may be reusable directly in the trainer's ComfyUI tab — check before building from scratch.
- **UI component goldmine:** glassmorphism components (`glass-notification`, `glass-popover`), glowing effects (`glowing-effect`, `glowingbordercard`), `smooth-cursor`, `vercel-card`, `vercel-tabs`, `kokonutui/ai-loading`. All Next.js 16 + shadcn/ui — direct transplant candidates for the UI unjankification work (Section 14).

---

### 11.3 Ecosystem Architecture Principles

As more tools are integrated, these rules keep things from becoming a mess:

1. **One port, one tunnel.** All ecosystem tools proxy through port 3000 via `server.js`. No second Cloudflare tunnel endpoints. Users access everything from one URL.
2. **One settings page.** External tool URLs/ports live in the existing settings system, not scattered `.env` files.
3. **Handoff buttons, not deep integration.** Tools stay loosely coupled. A "Test in ComfyUI" button is fine; sharing state stores between tools is not.
4. **Each tool is optional.** If ComfyUI isn't running, the ComfyUI page shows a friendly "not connected" state — it does not break anything else.

---

### 11.4 Priority Matrix — Ecosystem Features

| Feature | Category | Effort | Status |
|---------|----------|--------|--------|
| COMFY-1: server.js proxy for ComfyUI | Infrastructure | Small | ⏳ Not started |
| COMFY-2: Copy lib/stores/hooks layer from KNX-ComfyUI | Integration | Tiny | ⏳ Not started |
| COMFY-3: UI page + navbar link + skeleton disconnected state | Integration | Small | ⏳ Not started |
| COMFY-4: Settings integration (ComfyUI URL field) | UX | Tiny | ⏳ Not started |
| COMFY-5: "Test in ComfyUI" post-training button | Feature | Medium | ⏳ Not started |
| DT-1: server.js proxy + startup scripts for Dataset Tools | Infrastructure | Small | ⏳ Not started |
| DT-2: Navbar link to Dataset Tools | Integration | Tiny | ⏳ Not started |
| DT-3: Handoff buttons (inspect LoRA, view reference, files) | Feature | Small | ⏳ Not started |
| DT-4: Deep-link folder awareness | Enhancement | Small | ⏳ Not started |

---

## Section 12 — Security Review Backlog

### 12.1 CWE-23 Path Traversal (Snyk removed, low urgency, not today)

> **2026-05-07 note:** Snyk has been removed from the project (rules were SaaS-calibrated noise for a local tool). Path traversal sanitisation is a *good idea in principle* but is **not imperative right now** — this is a single-user local tool, not a public web app. We could add it for belt-and-suspenders safety, but it's firmly in "yeah we should, but not today" territory. DeepScan is still active on the JS side; Bandit is noted as a future candidate for Python security scanning if we ever want coverage again.

~~Snyk flags~~ Path traversal (CWE-23) exists across several Python service files. Partially false-positive for a single-user local tool, but worth a sanitisation pass before any public/multi-user deployment.

**Current state:**
- `api/routes/files.py` — already protected: `is_safe_path()`, `ALLOWED_DIRS`, `is_relative_to()` used before every `open()`. Snyk false-positive here.
- `services/lora_service.py`, `services/caption_service.py`, `services/tagging_service.py` — user-supplied paths go straight to `Path()`. Intended behaviour (user specifies their own model/dataset paths) but technically traversable.
- `api/routes/config.py` — preset file paths from API requests, no allowlist check. Lowest-hanging real concern.

**Why it's low risk now:** App is single-user, accessed via private tunnel on VastAI/RunPod. Path traversal = user accessing their own files, which they already can.

**Why it matters eventually:** If the app ever supports multiple users, shared instances, or public access, these become real attack surfaces.

**Investigation items (SEC-1):**
- Check if service-layer paths can be validated against a configurable `WORKSPACE_ROOT` without breaking VastAI/RunPod users who store files outside the project directory
- Consider `Path.resolve()` + `is_relative_to(WORKSPACE_ROOT)` pattern in services — only viable if `WORKSPACE_ROOT` is user-configurable in settings (default: `/workspace` on cloud, `PROJECT_ROOT` locally)
- `api/routes/config.py` preset paths — add same `is_relative_to(presets_dir)` guard that `files.py` already uses for a quick partial win

**Effort:** Small–Medium | **Priority:** Low (pre-public-release gate) | **Status:** ⏳ Not started

---

### 12.2 CWE-78 Command Injection (Snyk, medium — mostly false positives)

Snyk flags command injection (CWE-78) in Python files that pass user-controlled values into subprocess calls.

**Current state:**
- All subprocess calls use `asyncio.create_subprocess_exec` or list-form `subprocess.run` — neither passes through a shell. OS `exec()` does not interpret shell metacharacters, so these are not injectable regardless of argument content.
- No `shell=True`, `os.system()`, or `os.popen()` anywhere in `api/` or `services/`. The one `shell=True` in `job_manager.py` was removed in commit `ab2fb6b`.
- `model_service.py` — aria2c and wget calls build arg lists from user-supplied URLs and API tokens. List-form, not shell-interpolated. Safe.
- `services/trainers/`, `lora_service.py`, `captioning_service.py` etc. — training/processing commands built as lists with user-supplied paths. Same pattern, safe.

**Why Snyk flags it:** Static analysis sees user-controlled values flowing into subprocess args and flags conservatively, even when list-form exec is used. It cannot always prove the list won't later be joined into a shell string.

**Why it's low real risk:** List-form exec is the correct mitigation for CWE-78. No shell is invoked, no metacharacter interpretation occurs.

**Investigation items (SEC-2):**
- Audit that no call site ever does `" ".join(args)` and passes the result to a shell-based subprocess — this would re-introduce the vulnerability
- Verify `model_service.py` wget/aria2c header args (`Authorization: Bearer {token}`) are never shell-interpreted if the download backend is swapped in future
- Consider adding a lint rule or comment convention to flag any future `shell=True` additions at review time

**Effort:** Tiny (audit only, no fixes expected) | **Priority:** Low | **Status:** ⏳ Not started

---

---

## Section 13 — In-App Documentation Cleanup

### 13.1 Integrate upstream docs + remove hand-holding bias

**Priority:** Low (post-in-house-testing)
**Status:** ⏳ Not started

The in-app `/docs` section needs a proper cleanup pass before beta. Two things to tackle together:

**Source material to pull in:**
- LyCORIS documentation — copy relevant parameter/network type explanations into the in-app docs
- sd-scripts documentation — training flags, optimizer notes, scheduler behaviour etc.
- Strip anything that's already outdated or contradicted by our vendored versions

**Bias/tone cleanup:**
- Remove hand-holdy warnings that assume the user doesn't know what they're doing (e.g. hardware requirement nags on pages where the user has already made the choice to be there)
- "Unbiasify" any Claude-written docs that over-explain or hedge excessively
- Changelog page (`app/changelog/`) exists but is removed from nav — either populate it from git history or delete the route entirely

**Note:** Do not tackle this during active testing phases. Brain goes to FF9 Quina frog-catching, docs rot. Schedule for a dedicated docs sprint.

**Effort:** Medium | **Blocked by:** Stable alpha + in-house testing complete

---

---

## Section 14 — UI Unjankification: The Gate-Ripping

**Priority:** Beta (parallel to feature work — can start any time)  
**Status:** Planning phase — 2026-05-09  
**Codename:** "Throw wide the gates" (yes, this is a Crystal Exarch reference, we're keeping it)

### 14.1 The Problem

The trainer works. Performance is excellent — lighter than Gradio alone by a wide margin. But visually the UI is "Gradio adjacent": dense, muted, flat, functional-first but aesthetically anonymous. This happened because we stopped worrying about the UI once the trainer itself needed fixing. Now the trainer is mostly working. Time.

The goal is NOT a ground-up redesign. The component architecture is good, shadcn/ui is good, the structure is sound. The goal is: make it **feel like a product someone chose to use**, not a tool someone had to use.

### 14.2 What "Gradio Adjacent" Actually Means (and what to fix)

| Current state | Target state |
|--------------|--------------|
| Muted blue/gray palette with no identity | A visual identity that's distinctly KNX |
| Cards look identical regardless of importance | Visual hierarchy — primary workflow stands out |
| Rainbow icon colors (`text-cyan-400`, `text-pink-400`) with no semantic meaning | Semantic color use — colors mean something |
| Dense form fields stacked with minimal breathing room | Better spacing, section grouping, visual separation |
| No "alive" feeling — dashboard is a static grid | At least one live element (active job, recent dataset) |
| Placeholder text that explains nothing or explains too much | Tight, useful placeholder copy |
| Error states that look the same as empty states | Distinct states: loading / empty / error / disconnected |
| Training form is one very long scroll with no orientation | Clear sections, maybe sticky section nav or progress steps |

### 14.3 Specific Areas

**A. Visual identity / color**
- **Don't build a theme system — port one.** Dataset-Tools (`Ktiseos-Nyx/Dataset-Tools`) already has a complete, polished theme system: `ThemeCustomizer` (floating toolbar), `color-swatch-selector`, `theme-toggle`, `useThemeColor()` hook. Port these directly. See Dataset-Tools issue #196 "UI/UX Theory!" for context.
- The system uses `data-theme-color` attribute on `documentElement` + CSS variables, with localStorage persistence. 7 accent colors (zinc, red, orange, green, blue, violet, pink). Pairs with `next-themes` for dark/light.
- After porting: pick which accent color(s) to default to for KNX's visual identity. The site's meteor hero has character — the inner pages should match it.
- Semantic colors: green = active/running, amber = warning/needs attention, red = error/stopped, blue = info. Don't use them decoratively.

**B. Dashboard (already tracked as UI-2)**
- Workflow grouping (Dataset → Tag → Train → Upload) instead of 9 undifferentiated tiles
- Live "active jobs" widget at top
- All routes actually listed, not just the 9 currently there

**C. Training form**
- The form is extremely long. Users lose orientation.
- Options: sticky sidebar with section links, collapsible card sections, a step-indicator at the top
- Progressive disclosure: hide rarely-used advanced fields behind an "Advanced" expander per card. Most users never touch `lr_power`, `rank_dropout`, `module_dropout` etc.
- Required fields vs optional fields should look different

**D. Empty/loading/error states**
- Every page that fetches data (models list, dataset list, files) needs a proper skeleton loader, empty state with a CTA, and error state that tells you what to do
- Currently most pages go from loading spinner to content with nothing in between
- Empty states should be friendly: "No datasets yet — upload one to get started" with a button, not just blank space

**E. Feedback and toasts**
- Toast messages are functional but terse. Slightly warmer copy ("Training started! You can close this tab — it'll keep running." not just "Training started.")
- Success toasts feel the same as info toasts. Green for success, neutral for info.

**F. The nav**
- Navbar is functional but dense. On wider screens there's room to show more.
- Consider: active route highlighting is too subtle currently
- Ecosystem tools (ComfyUI, Dataset Tools when added) should have a distinct "Ecosystem" group in the nav

### 14.4 What NOT to do

- Don't do a full design system overhaul. Keep shadcn/ui, keep the component structure.
- Don't add animations for their own sake. Motion only where it aids understanding (loading states, transitions, not decorative spinning).
- Don't make it look like every other AI tool (dark blue + purple gradient = "I was made in an AI studio"). Avoid that.
- Don't touch the meteor background bleed on the hero. It's intentional, it has character.
- Don't redesign and then lose all the accessibility work that's already in place.

### 14.5 Approach

This is iterative, not a big-bang redesign. Work page by page, card by card:

1. **Color/identity** — establish the palette first, apply globally. One session.
2. **Dashboard** — already scoped as UI-2. Pair with color work.
3. **Training form** — progressive disclosure + section orientation. Largest single effort.
4. **Empty/loading/error states** — page by page, can be spread across sessions.
5. **Copy pass** — placeholder text, toast messages, descriptions. Fast, high-impact.
6. **Nav polish** — last, once content is stable.

### 14.6 Priority Matrix

| Task | Effort | Impact | Status |
|------|--------|--------|--------|
| Port theme system from Dataset-Tools (customizer + swatches + hook) | Small | High | ⏳ Not started |
| Dashboard redesign (UI-2) | Medium | High | ⏳ Not started |
| Training form: progressive disclosure | Medium | High | ⏳ Not started |
| Training form: section nav / orientation | Small | Medium | ⏳ Not started |
| Empty/loading/error states (all pages) | Medium | High | ⏳ Not started |
| Toast copy pass | Tiny | Medium | ⏳ Not started |
| Nav: active states + ecosystem group | Small | Medium | ⏳ Not started |
| Semantic color pass (icons, badges) | Small | Medium | ⏳ Not started |

---

**Document maintained by:** Ktiseos-Nyx-Trainer Project
