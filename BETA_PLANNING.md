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
**Status:** ✅ Done (2026-05-25) — tags/page.tsx frequency chips with counts (confirmed in UI)

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
**Status:** ✅ Done (2026-05-25) — bulk remove/replace in tags/page.tsx Actions menu

Once the Tag Viewer exists, add bulk actions on selected tags:
- **Remove tags** - delete selected tags from ALL images in dataset (with confirmation)
- **Replace tags** - find-and-replace: show each selected tag with input for replacement, apply across all images

**Implementation notes:**
- Civitai's replace modal is ~85 lines, very clean pattern
- Needs API endpoints: `POST /api/dataset/{name}/tags/remove` and `POST /api/dataset/{name}/tags/replace`
- Should operate on .txt files on disk, not just in-memory

### 1.3 Upgrade Overwrite Mode from Bool to 3-Way
**Priority:** Medium
**Status:** ✅ Done (2026-05-25) — auto-tag/page.tsx `overwriteMode` Select (ignore/append/overwrite)

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
**Status:** ✅ Done (2026-05-25) — per-image inline tag editor (badge chips + add) in tags/page.tsx

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

**Queue compatibility note:** Design the Active Jobs widget to show 1-or-N jobs from day one (a list, not a single slot). The job queue system (Phase 1 Beta, Section 5.1) will slot in as a backend change — the widget won't need to be redesigned. If the widget only handles one job at launch, adding queue support later requires a UI rewrite. The contract is: widget takes `Job[]`, renders each with progress + cancel; today that array has one item, later it has many.

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

### 5.0.95 Validation Schema / UI Dropdown Single Source of Truth

**Issue PR-0: OptimizerSchema and LRSchedulerSchema drift from UI dropdown**
- **Severity:** Medium (recurring bug class — caused #369)
- **Flagged by:** CodeRabbit on PR #370
- **Problem:** `OptimizerSchema` and `LRSchedulerSchema` in `validation.ts` are separate string arrays from the dropdown options in `OptimizerCard.tsx`. Every time a new optimizer/scheduler is added to the UI, it must be manually added to the schema too — and that drift is exactly what broke CAME/Compass/schedule-free optimizers before PR #370.
- **Fix:** Export `OPTIMIZER_VALUES` and `LRSCHEDULER_VALUES` as `as const` tuples from `validation.ts`. Build the Zod schemas from those constants. `OptimizerCard.tsx` dropdown keeps its own labels/descriptions but TypeScript can enforce values only come from `OPTIMIZER_VALUES`. Future additions require touching one place.
- **Status:** ⏳ This week (deferred from PR #370 — GPU was off)

---

### 5.0.96 Legacy Preset Audit — bmaltais Format Migration

**Issue PR-2: ~30 built-in presets still use old nested `config:{}` format with legacy field names**
- **Severity:** Medium (all old-format presets were silently broken before PR #370)
- **Background:** Most built-in presets were imported from bmaltais's Kohya SS gradio scripts. They use legacy field names (`optimizer`, `epoch`, `learning_rate`, `batch_size`, `lr_warmup`, `max_resolution`, `dataset_repeats`) nested under a `config:` block — a completely different schema from the current `TrainingConfig`.
- **Current state:** PR #370 added `normalizeLegacyPresetFields()` so these presets now *load* correctly via the mapping layer. But the files themselves are still in the old format, which is fragile.
- **Fix:** Audit all presets in `presets/` and convert any still using the old nested format to the flat format used by newer presets (e.g. `lora_SDXL - Illustrious-XL CAME Conservative v1.0.json`). Remove obsolete fields, fix legacy field names, ensure types are correct (strings → numbers where needed).
- **Status:** ⏳ Not started — low urgency since the mapping layer handles it, but good hygiene before beta

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
| Wire up wandb_key environment variable (LT-1) | Bug Fix | Tiny | ✅ Done (confirmed 2026-05-25) |
| Add WandB/Logging UI section (UI-1) | New Feature | Small | ✅ Done — LoggingCard.tsx (confirmed 2026-05-25) |
| Dashboard redesign with all routes (UI-2) | UX | Medium | 🔵 Design work, deferred |
| HF upload form persistence (HF-1) | UX/Bug Fix | Small | ✅ Done 2026-04-30 |
| Tag Viewer with frequency counts | New Feature | Medium | ✅ Done — tags/page.tsx frequency chips with counts (confirmed in UI 2026-05-25) |
| Bulk tag remove/replace | New Feature | Medium | ✅ Done — tags/page.tsx Actions menu (2026-05-25) |
| Fix alpha parameter UX in LoRA resize (MG-1) | Bug Fix | Small | ✅ Done 2026-04-30 |
| Add subprocess timeouts to merge operations (MG-3) | Reliability | Small | ✅ Done 2026-04-30 |
| CUDA availability check for merges (MG-4) | Error Handling | Small | ✅ Done 2026-04-30 |
| Update CheckpointTrainingConfig types (CT-5) | Bug Fix | Tiny | ✅ Done 2026-04-30 |

### Should Have (Beta quality)
| Feature | Category | Effort | Status |
|---------|----------|--------|--------|
| 3-way overwrite mode for tagging | Enhancement | Small | ✅ Done 2026-05-25 |
| Checkpoint-specific validation (CT-1) | Enhancement | Small | ⏳ Not started |
| Hide LoRA fields in checkpoint mode (CT-2) | UX | Medium | 🚫 N/A — separate pages, no unified form |
| Merge progress reporting (MG-7) | UX | Medium | ⏳ Not started |
| SD3 merge support (MG-5) | Feature | Small | ⏳ Deferred (part of #342) |
| Clean up redundant TOML generation (CT-6) | Tech Debt | Small | ⏳ Not started |
| Respect enable_bucket user setting (LT-2) | Bug Fix | Tiny | ✅ Done 2026-04-30 |
| LyCORIS algorithm-specific validation (LT-5) | Enhancement | Medium | ⏳ Not started |
| Clarify "Full" LoRA type semantics (LT-4) | UX | Small | ⏳ Not started |
| Silence AbortError console noise (UI-3) | Polish | Small | ✅ Done 2026-05-20 |
| Fix training log polling cadence + visibility (UI-4) | UX/Bug Fix | Small | ✅ Done 2026-04-30 |
| Add PYTHONUNBUFFERED to Kohya subprocess (UI-4 part) | Bug Fix | Tiny | ✅ Done (via -u flag in kohya.py:124) |
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

- Large textarea: paste URLs one per line, with **inline hashtag routing per
  line (A1111 BatchLinks style)**: `#model <url>`, `#lora <url>`, `#vae <url>`,
  `#dataset <url>`, `#output <url>`
- Global destination dropdown as the *default* for lines with no hashtag
- Progress list showing each download's status as it runs
- aria2c is already installed on VastAI/RunPod instances (required by existing workflow)

### Inspiration

Inspired by the A1111 `BatchLinks` extension which used `#destination` hashtag syntax to route downloads. **Dusk's vision (confirmed 2026-05-25): keep the hashtag syntax as the primary routing UX** (`#model <url>`, `#lora <url>`, etc.) — it's faster for power users pasting mixed lists — with a global dropdown only as the default for un-hashtagged lines. (Earlier draft proposed replacing hashtags with a dropdown; that was wrong — the hashtag flow is the point.)

**Reality check (2026-05-25):** the current `/models` download UI is just a card of links, NOT this paste-and-route batch tool. BD-1 is genuinely not started.

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

### Calculator Enhancement — Optimizer + LR Aware Step Guidance
The step calculator currently uses basic Kohya math. Proposal: make it optimizer- and LR-aware so it gives rough but useful guidance on target step counts:

**LR → steps relationship (the big one):**
- LR and step count are in direct tension: higher LR = more work per step = fewer steps needed to reach convergence
- The calculator should take LR as an input and adjust recommended steps/epochs accordingly
- This is more impactful than optimizer choice — LR drives the curve, optimizer affects how cleanly you ride it

**Optimizer adjustment (rough, vibes-based until we have more data):**
- AdamW8bit: baseline (1.0x)
- CAME: converges faster and burns in more aggressively — approx 0.6–0.7x steps for equivalent result; small datasets especially vulnerable to overtrain
- Adafactor (fixed LR): slower per step, more conservative — approx 1.2–1.3x
- Prodigy/DAdaptation: self-adjusting LR so step count guidance is less applicable

**Other inputs already planned:**
- Dataset size × repeats × batch size → total steps
- Resolution effect on VRAM and speed

Not exact science — label everything as "rough guide." Even a "CAME + small dataset: consider reducing epochs by 30%" hint would have saved two undertrained LoRAs. Discovered from real training runs May 2026.

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

### 2026-05-25 — ComfyUI Generate UI polish + doc reconciliation

**Shipped to `dev`:**
- ComfyUI Generate UI: queue-runs field (1–8), cleaner 400 error parsing (extracts `node_errors[*].details`), navbar "LoRA Manager" link → ComfyUI:8188, removed misplaced model-picker footer link, stripped author/fork attribution from UI strings (belongs in README).
- `extra_model_paths.yaml` written by installer: `output/`→loras, `pretrained_model/`→checkpoints, `vae/`→vae. Relative `..` paths, works local/VastAI/RunPod.
- ZIP upload fix: removed `keepalive:true` (Fetch spec caps keepalive bodies at 64 KiB → 10 MB chunks failed with "Failed to fetch"). Chunking itself solves the original 300 MB-into-RAM problem.
- Models page delete → shadcn AlertDialog (replaced raw `window.confirm()`).

**Findings (sdxl-knx / ComfyUI):**
- `Checkpoint Loader (LoraManager)` is willmiao's LoRA Manager node, NOT fearnworks (the `cnr_id` in KNX's workflow JSON was wrong). It only validates checkpoints LoRA Manager has **indexed** (with metadata) — raw HF-downloaded `.safetensors` 400 with `value_not_in_list` until indexed via CivitAI hash lookup. Dusk chose to keep the node + index-first rather than swap to `CheckpointLoaderSimple`.

**Doc reconciliation (statuses were stale):**
- **Section 1 (Tag/Caption) is 100% done** — 1.1 frequency chips, 1.2 bulk remove/replace, 1.3 3-way overwrite (auto-tag Select), 1.4 per-image inline editor. "Per-Image Visual Tag Editor" was just claude-speak for the inline editor already shipped.
- **WandB (LT-1/UI-1) done** — LoggingCard.tsx.
- **Merging tool is half-done** — core works (LoRA/ckpt/resize) but MG-5 (SD3 not wired), MG-6 (stdout not logged), MG-7 (no progress on multi-GB merges), MG-8 (no output-exists check) all still open.
- **BD-1 Batch Downloader not started** — current `/models` UI is just a link card. Vision corrected: keep BatchLinks-style hashtag routing (`#model <url>`), dropdown only as default.
- Reflow violations (memory): possibly low-RAM, not a confirmed code bug — don't chase until it reproduces.

**New idea — post-training "Test in ComfyUI →" button:**
After a training run finishes, offer a button to jump straight to testing the new LoRA in ComfyUI. KEY: `extra_model_paths.yaml` already maps `output/`→ComfyUI loras, so the trained LoRA is **already visible** in ComfyUI's picker — no file copy needed. Implement as a deep-link to the Generate page (ideally pre-filling the LoRA), not a copy operation. Small frontend job.

**Considering — forking / KNX-inspired custom ComfyUI nodes:**
- Currently the Save Image node in the sdxl-knx workflow is willmiao's **Save Image (LoraManager)** (saves CivitAI info + thumbnails + workflow — genuinely great).
- Open question (Dusk): can we embed a custom **"software" tag** identifying this trainer into saved images WITHOUT writing our own node?
- Hoped-for free path: ComfyUI's **`extra_pnginfo`** mechanism. We *already* pass `extra_data.extra_pnginfo.workflow` on submit (see `templateInjector.ts`); a spec-compliant SaveImage iterates ALL keys and embeds each as a PNG text chunk.
- **VERIFIED (2026-05-25) — does NOT work with the LoraManager node.** Reading willmiao's `py/nodes/save_image.py`, it writes at most two chunks: a `"parameters"` chunk (A1111-style string built internally from the metadata collector) and `"workflow"` from `extra_pnginfo["workflow"]` only. It explicitly ignores every other `extra_pnginfo` key. So a custom `software` tag is node-level, confirming Dusk's instinct.
- **Key constraint (Dusk, 2026-05-25):** the stock ComfyUI SaveImage is **NOT A1111/Civitai compatible** — it writes the `workflow` chunk but no A1111-style `parameters` string, so Civitai can't auto-read generation params from its output. The LoraManager node's entire value is that `parameters` chunk (`pnginfo.add_text("parameters", metadata)`). So the save node MUST stay A1111/Civitai compatible.
- **Options to add a software tag:**
  1. **Fork / KNX-inspired save node (the real path)** — start from the LoraManager save logic (keep the A1111 `parameters` chunk + thumbnails + CivitAI info), add a `software` text chunk (and any other KNX metadata). Only option that keeps Civitai compat AND adds the tag.
  2. ~~Switch to stock SaveImage~~ — REJECTED: gives a free `extra_pnginfo` tag but loses A1111/Civitai compatibility (no `parameters` chunk). Dealbreaker.
  3. Post-save server-side PNG text injection — awkward, fights ComfyUI's flow; would also have to re-implement the A1111 string. Skip.
- **Conclusion: forked/KNX-inspired save node is the only viable path** — and it's the natural anchor for the broader "fork or build KNX-inspired nodes" direction.

### 2026-05-20 — Reflow Fixes + Log Stream Cutout + ComfyUI Planning

**Completed this session:**
- **PR #375** — Reflow violations (#374): TrainingMonitor auto-scroll fully deferred into rAF with cancelAnimationFrame on cleanup; auto-tag SelectContent → `position="item-aligned"` (removes Floating UI getBoundingClientRect on mousedown); 4 raw `<button>` elements → shadcn `<Button>`; `MAX_LOGS` 1000→500; `aria-label`, `gap-2`, `type="button"` added per CodeRabbit review.
- **PR #376** — Log stream cutout bug: `Job.logs` is a `deque(maxlen=1000)`. Once full, `get_logs(since)` always returned `[]` because it used a buffer-relative index — `since` reached `maxlen` and `start >= len(logs)` was permanently true. Root cause of the 5-7 minute log cutout on a 4090. Fixed with `total_lines_written` absolute counter; same bug fixed in `stream_logs` WebSocket path. `deque` maxlen raised to 2000. 13 regression tests added.
- **BETA_PLANNING.md §11.1** — ComfyUI architecture finalised: submodule approach, B2 workflow-template state model, extension system (workflow templates + node packages), LoRA Manager button → new tab, ANIMA ↔ SDXL switcher.
- **UI-3** confirmed done — AbortError caught and suppressed in both `pollLogs` paths in `api.ts`.
- **UI-4 PYTHONUNBUFFERED** confirmed done — achieved via `-u` flag in `kohya.py:124`.

**Status check findings:**
- PR-0 (OptimizerSchema single source of truth) — still genuinely pending. No `OPTIMIZER_VALUES as const` in `validation.ts`.
- LT-1 + UI-1 (WandB logging UI) — still pending.

**Up next:**
- Merge PR #375 (reflow) → PR #376 (log stream) onto main
- LT-1 + UI-1 (#343): WandB key env var + LoggingCard
- PR-0: Export OPTIMIZER_VALUES/LRSCHEDULER_VALUES as `as const` from validation.ts

---

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
**Status:** Architecture finalised 2026-05-20. **Shipping decision: all-in-one.**
**Source:** v0-generated template, repo at `duskfallcrew/KNX-ComfyUI`, used as reference — not dropped in wholesale. Code lives in the main trainer repo.

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

#### ComfyUI backend: submodule (2026-05-20)

ComfyUI the Python backend ships as a **git submodule** in the trainer repo. It runs co-located on `localhost:8188` — always same machine, no SSH, no remote ComfyUI. Multi-GPU / remote-ComfyUI is explicitly future scope.

Provisioning scripts (`vastai_setup.sh`, `provision_runpod.sh`, `install.bat`) do `git submodule update --init --recursive` to pull it down and start it alongside FastAPI and Next.js. The frontend tab is always present; the disconnected skeleton state covers the case where the user hasn't initialised the submodule yet.

#### Workflow state architecture: B2 (2026-05-20)

We wrap ComfyUI's API — we do not re-implement node logic. Source of truth is always ComfyUI.

**Chosen approach (B2 — workflow-aware template mapping):**
- Ship known workflow templates: each template = `workflow.json` + `node-map.json` (node type → UI control binding)
- Templates live in `frontend/comfy/workflows/` — same spirit as `presets/`
- On connect: load the active template's node map; bind UI controls to node types (not node IDs — IDs shift, types don't)
- On generate: inject current UI values into the template JSON → `POST /comfy/prompt`
- On result: poll `/comfy/history` for output images

**Why not full live-graph reading (B1) yet:** B2 is the right starting point. B1 (reading every node dynamically via `/object_info`) is the long-horizon evolution as the node library grows — the architecture doesn't close that door, it just doesn't require it on day one.

**Architecture switcher (ANIMA ↔ SDXL):**
- Switcher in the UI header swaps which template is loaded — same resizable-panel UI, different workflow JSON + node map underneath
- ANIMA template is first (reference: `guy90sVerySimpleAndEasyTo_v10.json`, tested and verified 2026-05-19 — AuraFlow model sampling node, UNET/CLIP/VAE separate loaders, KSampler, Adetailer, Ultimate SD Upscale)
- SDXL template is the second built-in; in progress, not blocking ANIMA shipping

**Extension model:**
- Community contributions come in two forms: **workflow templates** (new architecture support) and **node packages** (new node type → UI control bindings)
- Both drop into `frontend/comfy/workflows/` — no app code changes needed to add a new architecture
- This is the same philosophy as `presets/` and maps to ComfyUI's own custom node ecosystem

**LoRA Manager integration:**
- The LoRA Manager custom node (`Lora Loader (LoraManager)`) ships with guy90s's workflow and provides a browseable popup UI for LoRA selection
- Our UI has a **LoRA Manager button** — clicking it opens ComfyUI's LoRA Manager in a new browser tab (A1111-style "open extra networks" pattern)
- LoRA selection state reads back from live workflow via the proxy — no separate sync needed

**Node presence check:**
- On connect, `GET /comfy/object_info` returns all loaded node types
- Soft-check: if a node type required by the active template is missing, surface a friendly warning ("This workflow requires LoRA Manager — install it via ComfyUI Manager")
- We do NOT auto-install nodes. Users manage their own ComfyUI custom nodes for now.

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

**COMFY-1: Proxy layer (prerequisite for everything else)** — ✅ Done 2026-05-23 (commit `dfc8a1e`)
- `frontend/server.js` — `/comfyui/*` HTTP proxy block strips prefix and forwards to ComfyUI
- `frontend/server.js` — `/comfyui/ws` WebSocket upgrade handler (matches the client's relative `baseUrl + '/ws'` pattern)
- Env var: `COMFYUI_PORT=8188` honored in `server.js` (still TODO: add explicit export in `start_services_vastai.sh`, `start_services_runpod.sh`, `restart.sh` for documentation)
- `lib/comfy/client.ts` — change default `baseUrl` from `http://localhost:8188` to `/comfyui` (handled in COMFY-2 when the file lands in the repo)

**COMFY-2: Library layer** — ✅ Done 2026-05-23 (commit `5ebd646`)
- `lib/comfy/types.ts` — full TypeScript types for all ComfyUI API surfaces (workflow, queue, history, system stats, WebSocket message discriminated union)
- `lib/comfy/client.ts` — `comfyClient` singleton: `submitPrompt`, `interrupt`, `getQueue`, `getHistory` (overloaded), `deleteHistory`, `getObjectInfo`, `getSystemStats`, `getImageUrl`, `ping`, `connectWebSocket` with auto-reconnect
- `lib/comfy/workflows/txt2img.ts` — `buildTxt2ImgWorkflow()` targeting SDXL / SD 1.5 (CheckpointLoaderSimple chain; see **ANIMA note** below)
- `lib/comfy/workflows/img2img.ts` — `buildImg2ImgWorkflow()` with LoadImage + VAEEncode
- `lib/comfy/workflows/index.ts` — barrel re-export
- `lib/comfy/useComfyConnection.ts` — `useComfyConnection()` hook: manages WS lifecycle, status, progress, queue; client ID from `sessionStorage` (stable per tab); `submitTxt2Img` convenience wrapper
- `lib/comfy/index.ts` — top-level barrel export

**COMFY-3: UI components** — ✅ Done 2026-05-23 (commit `5ebd646`)
- `components/comfy/ComfyConnectionStatus.tsx` — pill badge (emerald/yellow/zinc/red) for header
- `components/comfy/GenerateUI.tsx` — full two-panel resizable UI (Splitter); left: prompts, checkpoint, dimensions, sampler settings, LoRA stack, denoise/VAE/CLIP skip; right: image gallery + progress
- `app/comfyui/page.tsx` — connection-aware page shell (connecting / disconnected / error / connected states); BorderGlow disconnected card
- `components/blocks/navigation/navbar.tsx` — added "Generate" top-level nav item → `/comfyui`

**COMFY-4: Settings integration** — ✅ Done 2026-05-23 (commit `5ebd646`)
- `lib/node-services/settings-service.ts` — `comfyui_url?: string` field; GET returns `comfyui_url: settings.comfyui_url ?? 'http://localhost:8188'`; POST merges update
- `app/api/settings/user/route.ts` — accepts `comfyui_url` string in POST body
- `app/settings/page.tsx` — ComfyUI GradientCard section with URL Input before API Keys section; note: changes take effect within 5 s (proxy cache)

**ANIMA workflow vs SDXL — node structure differences**

The bundled SDXL workflow (`guy90sVerySimpleAndEasyTo_v10.json`, adapted from Guy90s) uses a `CheckpointLoaderSimple` chain — a single node loads MODEL + CLIP + VAE. ANIMA uses a completely separate loader pattern:

| Aspect | SDXL / SD 1.5 (`buildTxt2ImgWorkflow`) | ANIMA (`buildAnimaWorkflow` — TODO) |
|--------|----------------------------------------|--------------------------------------|
| Model loader | `CheckpointLoaderSimple` (outputs MODEL[0], CLIP[1], VAE[2]) | `UNETLoader` (MODEL only) |
| CLIP loader | Output [1] of checkpoint | `DualCLIPLoader` (loads two CLIP models for AuraFlow — CLIP-L + T5XXL) |
| VAE loader | Output [2] of checkpoint | `VAELoader` (separate VAE file) |
| Sampling | `KSampler` | `KSampler` + `ModelSamplingAuraFlow` (patches the model's sigma schedule for AuraFlow's non-standard distribution) |
| Detailer | Adetailer via Impact Pack | same |
| Upscaler | Ultimate SD Upscale | same |

`buildAnimaWorkflow()` needs to live at `lib/comfy/workflows/anima.ts` and wire:
- Node 1: `UNETLoader` → `model`
- Node 2: `DualCLIPLoader` → `clip` (clip_name1 = CLIP-L file, clip_name2 = T5XXL file, type = "stable_diffusion" or "flux" depending on AuraFlow version)
- Node 3: `VAELoader` → `vae`
- Node 4: `ModelSamplingAuraFlow` (patches Node 1's model output) → patched `model`
- Nodes 5, 6: `CLIPTextEncode` positive/negative (using Node 2's clip)
- Node 7: `EmptyLatentImage`
- Node 8: `KSampler` (uses patched model from Node 4)
- Node 9: `VAEDecode`
- Node 10: `SaveImage`
- LoRA nodes 20+: `LoraLoader` chain injected between UNETLoader and ModelSamplingAuraFlow

The `GenerateUI` architecture switcher (header toggle) will call `buildAnimaWorkflow` vs `buildTxt2ImgWorkflow` depending on user selection — same resizable panel, different builder underneath. This is tracked in the architecture switcher todo below.

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

**COMFY-7: Auto-install required custom nodes (revised stance, 2026-05-23)**

The earlier "we do NOT auto-install nodes" position was conservative scope-trimming, not a technical decision. Since we ship a custom UI template that *requires* specific nodes to function (LoRA Manager, rgthree, Impact Pack, Ultimate SD Upscale), missing nodes = silently broken feature from the user's perspective. Auto-installing is a correctness requirement, not a power-user convenience.

**Required nodes for the bundled SDXL / ANIMA workflow templates:**
- `rgthree-comfy` — https://github.com/rgthree/rgthree-comfy (Seed, Fast Groups Bypasser, Image Comparer)
- `ComfyUI-Lora-Manager` — https://github.com/willmiao/ComfyUI-Lora-Manager (LoRA loader + Save Image)
- `ComfyUI-Impact-Pack` — https://github.com/ltdrdata/ComfyUI-Impact-Pack (DetailerForEach, SAMLoader, ImpactSimpleDetectorSEGS, SEGSPreview)
- `ComfyUI-Impact-Subpack` — https://github.com/ltdrdata/ComfyUI-Impact-Subpack (UltralyticsDetectorProvider)
- `ComfyUI_UltimateSDUpscale` — https://github.com/ssitu/ComfyUI_UltimateSDUpscale (UltimateSDUpscale)
- `comfyui_fearnworksnodes` — **KNX SDXL fork only**: `Checkpoint Loader (LoraManager)` node that loads MODEL+CLIP+VAE from a single checkpoint. Required by `sdxl-knx-v1.json`. Not needed for ANIMA template.
- `ComfyUI-Manager` — https://github.com/ltdrdata/ComfyUI-Manager (optional but useful so users can add extras themselves)

**Provisioning script approach (preferred):** `vastai_setup.sh` / `provision_runpod.sh` / `install.bat` clone each repo into `ComfyUI/custom_nodes/` and `pip install -r requirements.txt` for each. No ComfyUI Manager dependency, no chicken-and-egg problem, deterministic.

**In-app fallback (later):** On `/comfyui` page load, check `/object_info` for missing node types. If any required node is missing, show a dialog with an "Install missing nodes" button that POSTs to a Next.js API route which runs the equivalent `git clone` server-side. Skip if ComfyUI Manager handles it.

**COMFY-8: ComfyUI backend location — DECIDED 2026-05-23: direct clone everywhere**

`.gitmodules` exists but is empty; `ComfyUI/` directory does not exist. Direct-clone approach chosen over git submodule:

- **All provisioning paths clone ComfyUI consistently** — `vastai_setup.sh`, `provision_runpod.sh`, `install.bat`, `install.sh` each `git clone https://github.com/comfyanonymous/ComfyUI` into the platform-appropriate directory (e.g. `/workspace/ComfyUI` on remote, `./ComfyUI` next to the trainer locally).
- **No submodule overhead** — no `.gitmodules` config, no `git submodule update --init --recursive` step, no submodule pin to bump.
- **Graceful degrade if missing** — if a local user runs the trainer without having gone through the installer (or deletes their `ComfyUI/` dir), the `/comfyui` page shows the disconnected skeleton state already planned. App stays functional; ComfyUI tab is just inert.
- **Same install loop for custom nodes** — the COMFY-7 custom node install routine runs the same way regardless of how ComfyUI got there.

Rejected: git submodule (extra complexity for no real benefit in this use case; we don't need a version pin since ComfyUI's `main` branch is stable enough and breaking changes are rare).

**COMFY-9: `knx-nodes` ComfyUI custom node package**

A small package owned by KNX that ships alongside the bundled workflow templates. Auto-installed by provisioning scripts the same way as third-party nodes. Initial scope is two nodes:

- **`KNXSaveImage`** — saves images with `Software: KNX Ecosystem` PNG metadata so downstream tools (Dataset-Tools, Discord bots, Civitai) tag the source correctly. Written from scratch using ComfyUI core's `SaveImage` pattern (MIT) — no fork. Optionally embeds a structured `knx_metadata` JSON chunk (template name, workflow version, KNX trainer build).
- **`KNXMetadataReader`** — loads an image and extracts PNG text chunks as ComfyUI `STRING` outputs (positive prompt, negative prompt, seed, model, etc.). Lets users feed an existing image's prompt directly into `CLIPTextEncode` without re-tagging via WD14. Intentionally narrow — read chunks, return strings, no format-scoring heuristics. (Lesson from the vendored sdpr in Dataset-Tools-main: numpy scoring stacks become unmaintainable fast.)

Package layout:
```
knx-nodes/
├── __init__.py          # NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS
├── knx_save_image.py    # KNXSaveImage
├── knx_metadata_reader.py  # KNXMetadataReader
├── requirements.txt
└── README.md
```

Lives in its own GitHub repo (`Ktiseos-Nyx/knx-nodes` or similar) so the provisioning loop clones it the same way it clones rgthree/Impact Pack. Growth path: any custom node KNX needs that doesn't exist upstream goes here.

**Future ideas (not blocking):** dataset folder save node, training reference metadata embedder, ArcEnCiel-aware Civitai uploader node.

#### Notes
- ComfyUI backend is **directly cloned** by all install paths (local + remote), always co-located on `localhost:8188`. See COMFY-8 above for the decision rationale; the "ComfyUI backend: submodule" subsection further up is superseded.
- The template has workflow builders for `inpaint`, `controlnet`, and `adetailer` types referenced in `types.ts` but not yet built in `workflows/`. Those are future scope.
- Check whether `zustand` is already a dep before adding it — the stores use it.
- **Issue #374 (Reflow Violations)** fixed in PR #375 (2026-05-20). TrainingMonitor auto-scroll moved fully into rAF with cleanup; auto-tag page SelectContent → position="item-aligned"; raw buttons → shadcn Button.

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

**DT-5: "KNX Ecosystem" source tag (pairs with COMFY-9)**

Today, images generated by the bundled ComfyUI workflows show up as `Automatic1111` in Dataset-Tools' viewer because the LoRA Manager save node writes a `parameters` PNG chunk in A1111 format, and detection at `lib/parseImageMetadata.ts:181` keys off that chunk. Once `KNXSaveImage` (COMFY-9) writes `Software: KNX Ecosystem`, Dataset-Tools needs a matching detection block:

- **Next.js viewer** (`lib/parseImageMetadata.ts`): add `'KNX Ecosystem'` to the `format` union type; add detection block `if (textChunks['Software'] === 'KNX Ecosystem')` before the A1111 check (mirroring the NovelAI pattern at line 172).
- **Python CLI** (vendored sdpr at `dataset_tools/vendored_sdpr/format/`): add a `KNXFormat` class with `tool = "KNX Ecosystem"` and register it in `image_data_reader.py:PARSER_CLASSES_PNG` ahead of A1111.

Tags both ends of the same ecosystem with one consistent name.

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
| COMFY-1: server.js proxy for ComfyUI | Infrastructure | Small | ✅ Done 2026-05-23 |
| COMFY-2: lib/comfy layer (client, types, connection hook, workflows) | Integration | Tiny | ✅ Done 2026-05-23 |
| COMFY-3: UI page + navbar + architecture switcher + template injector | Integration | Small | ✅ Done 2026-05-24 |
| COMFY-4: Settings integration (ComfyUI URL field) | UX | Tiny | ✅ Done 2026-05-23 |
| COMFY-5: "Test in ComfyUI" post-training button | Feature | Medium | ⏳ Not started |
| COMFY-7: Auto-install required custom nodes (provisioning) | Infrastructure | Small | ✅ Done 2026-05-24 (fearnworksnodes URL still TODO) |
| COMFY-8: ComfyUI submodule vs direct clone (decision) | Decision | n/a | ✅ Decided: direct clone everywhere |
| COMFY-9: knx-nodes package (KNXSaveImage + KNXMetadataReader) | New Repo | Small | ⏳ Not started |
| COMFY-10: Model picker (useComfyModels + /models API + combobox UI) | UX | Small | ✅ Done 2026-05-24 |
| COMFY-11: Model download to ComfyUI folder (from HuggingFace/Civitai) | Feature | Medium | ✅ Done 2026-05-24 |
| COMFY-12: Auto-download Ultralytics bbox/segm models for Impact-Pack | Infrastructure | Tiny | ⏳ Not started |
| DT-1: server.js proxy + startup scripts for Dataset Tools | Infrastructure | Small | ⏳ Not started |
| DT-5: KNX Ecosystem source tag detection (pairs with COMFY-9) | Integration | Tiny | ⏳ Not started |
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

## Section 16 — ArcEnCiel Model Browser

**Priority:** Beta+ / "Nice to Have"
**Status:** Not started
**Permission:** Confirmed with ArcEnCiel team
**Source:** https://github.com/Anzhc/ArcEnCiel-Extension-for-WebUI
**API:** `https://arcenciel.io/api` — public, no authentication required

### What it is

ArcEnCiel (arcenciel.io) is a community model platform hosting LoRAs, checkpoints, VAEs, and embeddings — primarily anime/illustration focused, with strong Illustrious/NoobAI/Pony coverage. Adding it as a second model source gives users an alternative to Civitai, which can be geoblocked or rate-limited.

### API

No API key required. Key endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/models/search?search=&sort=&page=&limit=&base_model=&type=` | Search models |
| `GET /api/models/{id}` | Model details + versions |
| `GET /api/models/{id}/versions/{version_id}/download` | Download URL for a version |
| `GET /api/models/{id}/gallery` | Preview images |

Response includes model metadata, all versions, activation tags, base model, and direct download URLs. No OAuth, no rate limit enforced in the extension code.

**"Link versions"** = model versions (same LoRA trained on different base models, different training epochs). Not a separate concept — just standard versioning like Civitai.

### Implementation

Same shape as the existing Civitai browser (`/models/browse`). Options:

1. **New route** `/models/arcenciel` — mirrors Civitai browser structure, hits ArcEnCiel API instead. Cleanest separation.
2. **Second tab** on `/models/browse` — toggle between Civitai and ArcEnCiel. More compact.

Backend barely needs touching — the existing download endpoint already handles arbitrary URLs. Only additions needed:
- New Next.js API route: `GET /api/arcenciel/models` (proxy to ArcEnCiel API)
- Frontend page/tab with search, base model filter, model cards, download button

Base model filter values that map to our existing model types: Illustrious, NoobAI, Pony, Flux, SDXL, SD1.5.

### Attribution

Add to `ATTRIBUTIONS.md` when implemented:
```text
ArcEnCiel Extension for WebUI by Anzhc
https://github.com/Anzhc/ArcEnCiel-Extension-for-WebUI
Used with permission from the ArcEnCiel team.
```

### Priority matrix

| Item | Effort | Status |
|------|--------|--------|
| Next.js API proxy route for ArcEnCiel search | Small | ⏳ Not started |
| Model browser page/tab (search, filters, cards) | Medium | ⏳ Not started |
| Download wiring (reuses existing endpoint) | Tiny | ⏳ Not started |
| ATTRIBUTIONS.md entry | Tiny | ⏳ Not started |

---

## Section 15 — Alternative Training Backends

### 15.1 SimpleTuner

**Priority:** Beta+ / "Big Additions" — full form UI required before shipping  
**Status:** Not started  
**Source:** https://github.com/bghira/SimpleTuner  
**Install:** `pip install simpletuner[cuda]` — standard package, no vendoring needed

#### Why SimpleTuner

SimpleTuner supports 33+ model families that Kohya doesn't cover or covers poorly:

| Notable models | Notes |
|---|---|
| Flux.1 / Flux.2 | Better LoRA support than Kohya's flux path in some community tests |
| Chroma | Full support (Kohya does partial via flux_train.py --model_type chroma) |
| Wan 2.x (video) | Video LoRA — not in Kohya at all |
| LTXVideo / HunyuanVideo | Video fine-tuning |
| HiDream, Lumina2, Sana | Modern image architectures |
| SDXL / SD1.x | Covers the basics too |

It also handles full fine-tuning, LyCORIS (lokr etc.), and quantised training (int8/fp8/nf4) natively.

#### Architecture fit

`BaseTrainer` (`services/trainers/base.py`) already defines the interface — adding SimpleTuner is a new implementation, not a restructure:

```text
services/trainers/
  base.py                 ← existing ABC, no changes needed
  kohya.py                ← existing
  kohya_toml.py           ← existing
  simpletuner.py          ← NEW: BaseTrainer subclass
  simpletuner_config.py   ← NEW: generates config.json + multidatabackend.json
```

Training is invoked via `accelerate launch simpletuner/train.py` (or `simpletuner train`) — same subprocess pattern as Kohya.

#### Config format differences from Kohya

SimpleTuner uses two JSON files instead of one TOML:

1. **`config.json`** — training hyperparams (lr, optimizer, model path, output dir, etc.)
2. **`multidatabackend.json`** — array of dataset backend objects, each with its own resolution, caption strategy, caching dirs, repeat count, and sampling probability

The multidatabackend concept is richer than Kohya's single `dataset_config` — it allows mixing multiple datasets with different settings and weights in one run. For the initial implementation, we can simplify this to a single local backend.

#### Key config fields to expose in the UI

**Core (must have):**
- `model_family` — dropdown of supported architectures (maps to ST's `--model_family`)
- `pretrained_model_name_or_path` — base model path/HF ID
- `model_type` — `lora` or `full`
- `lora_rank` / `lora_alpha`
- `lora_type` — `standard` (PEFT) or `lycoris`
- `learning_rate`, `optimizer`, `lr_scheduler`, `lr_warmup_steps`
- `max_train_steps` or `num_train_epochs`
- `train_batch_size`
- `mixed_precision`, `base_model_precision` (quantisation)
- Dataset dir, caption strategy, resolution

**Dataset backend (simplified for v1):**
- Single local backend only (no AWS/GCS)
- `instance_data_dir`, `instance_prompt`, `caption_strategy`
- `resolution`, `repeats`

**Advanced (progressive disclosure):**
- `gradient_checkpointing`, `gradient_checkpointing_interval`
- `checkpoint_step_interval`, `checkpoints_total_limit`
- Validation: `validation_steps`, `validation_prompt`, `validation_resolution`
- `report_to` (wandb/tensorboard), `tracker_project_name`
- `lora_format` — `diffusers` or `comfyui`
- `lycoris_config` (JSON editor, shown only when lora_type=lycoris)

#### Installation

SimpleTuner is a pip package — install alongside existing deps:
```text
requirements.txt: simpletuner[cuda]
```

It has different torch/diffusers version requirements than Kohya. May need a separate venv or careful dependency alignment — verify before committing to shared venv. If separate venv is needed, use the same subprocess env-injection pattern (`SIMPLETUNER_VENV` path in settings) that could be added alongside `kohya.py`.

#### Frontend scope

Full dedicated training page at `/training/simpletuner` (or a tab switcher on the existing `/training` page). The Kohya form is not reusable — SimpleTuner has fundamentally different fields (model family selector, multidatabackend concept, quantisation options). A new form is the right call.

The `TrainingConfig` Pydantic model is Kohya-specific. Options:
- **Recommended:** A separate `SimpleTunerConfig` Pydantic model + new API route `/api/training/simpletuner/start`
- Alternative: add a `trainer_backend` discriminator to `TrainingConfig` and union the models — more complex, only worth it if we want a single form endpoint

Route in `server.js` `nodeApiPrefixes`: add `/api/training/simpletuner` to the whitelist so it hits Next.js (if we ever migrate) or leave as FastAPI (current pattern).

#### Implementation sequence

1. **ST-1:** `services/trainers/simpletuner.py` + `simpletuner_config.py` — JSON config generation + subprocess invocation. No UI yet. Manual testing via API.
2. **ST-2:** `services/models/simpletuner_config.py` — Pydantic model for SimpleTuner config. New FastAPI route `/api/training/simpletuner/start`.
3. **ST-3:** Frontend form — model family selector, core fields, dataset section, progressive disclosure for advanced.
4. **ST-4:** Preset system — ST presets live in `presets/simpletuner/` to keep them separate from Kohya presets.
5. **ST-5 (future):** Multi-backend dataset config UI — allow defining multiple dataset backends with different weights.

#### Priority matrix

| Item | Effort | Status |
|------|--------|--------|
| ST-1: Backend trainer + config generation | Medium | ⏳ Not started |
| ST-2: Pydantic model + API route | Small | ⏳ Not started |
| ST-3: Frontend form (full) | Large | ⏳ Not started |
| ST-4: Preset system | Small | ⏳ Not started |
| ST-5: Multi-dataset backend UI | Medium | ⏳ Future |

---

### 15.2 musubi-tuner (Future)

**Priority:** Future / Beta++  
**Status:** Needs scoping — research first  
**Source:** https://github.com/kohya-ss/musubi-tuner (same author as kohya-ss)

Kohya's newer tuning framework. Promising but overlap with existing Kohya backend is significant — needs a proper scope session to determine what it unlocks vs. what we already cover. Do not design until SimpleTuner (§15.1) is shipped and stable.

---

### 15.3 Chroma / Flow Training Improvements (Future)

**Priority:** Future  
**Status:** Not started

Chroma is already partially supported via `flux_train.py --model_type chroma` in the vendored Kohya backend. SimpleTuner (§15.1) adds native Chroma support as a side effect of its broader model family coverage. Evaluate what's still missing after SimpleTuner lands before scoping a dedicated Chroma training path.

---

---

## Section 17 — Image → Prompt Helper (WD-Tagger + Florence-2)

**Priority:** Beta / Phase 2  
**Status:** Not started  
**Depends on:** Nothing — WD-tagger and transformers are already in requirements

A standalone utility that takes a single image and returns two things: booru-style tags from WD-tagger and a natural-language caption from Florence-2. The intent is prompt generation — give the tool a reference image, get usable prompt text back. Distinct from the dataset auto-tagger (batch, writes `.txt` sidecar files); this is interactive, single-image, no file writes.

**Natural fit with Dataset-Tools (§11.2):** once Dataset-Tools is embedded, an "Extract Prompt" button on any image in the browser can deep-link here with the image pre-selected.

---

### 17.1 Backend

**New route:** `POST /api/utilities/image-to-prompt`

Accepts a multipart image upload, runs both models, returns:

```json
{
  "tags": ["1girl", "solo", "long_hair", "..."],
  "tag_scores": {"1girl": 0.98, "solo": 0.96, "...": 0.0},
  "caption": "A young woman with long silver hair standing in a forest.",
  "detailed_caption": "..."
}
```

**WD-tagger:** reuse `custom/tag_images_by_wd14_tagger.py` logic but as a Python function call, not a subprocess — or expose a thin single-image endpoint that calls the same ONNX model. Model is already cached by the auto-tag workflow so no extra download on first use.

**Florence-2:** `microsoft/florence-2-base` (~232 MB, fast) or `florence-2-large` (~770 MB, better captions). Use the transformers pipeline with task token `<MORE_DETAILED_CAPTION>`. Lazy-load on first request so startup time isn't affected. Cache the loaded model in a module-level variable (same pattern as the existing BLIP captioner in `caption_service.py`).

Available Florence-2 caption tasks (selectable by user or returned together):
- `<CAPTION>` — one sentence
- `<DETAILED_CAPTION>` — two to three sentences  
- `<MORE_DETAILED_CAPTION>` — paragraph

**New service file:** `services/image_prompt_service.py` — keeps the logic out of the route handler and mirrors the pattern in `captioning_service.py`.

---

### 17.2 Frontend

**Location:** new tab or card in `/utilities`, or its own page `/utilities/image-prompt`.

**UI flow:**
1. Image drop zone (reuse or adapt the one in `DatasetUploader.tsx`)
2. Optional: confidence threshold slider (same as auto-tag page, default 0.35)
3. Optional: Florence-2 model size toggle (base vs large)
4. Submit → spinner → results

**Results panel — two sections side by side:**
- **Tags (WD-tagger):** tag chips (same style as the tag editor), with a "Copy as comma list" button
- **Caption (Florence-2):** text display with length selector (short / detailed / more detailed), copy button

**Combine mode (nice to have):** single "Copy as prompt" button that produces `tags..., caption sentence` — useful for trainers who want structured + natural text together.

---

### 17.3 Models and Deps

| Model | Source | Size | Notes |
|-------|--------|------|-------|
| WD-tagger | Already in project (onnxruntime-gpu) | ~350 MB | No new dep |
| Florence-2 base | `microsoft/florence-2-base` via HF hub | ~232 MB | transformers already a dep |
| Florence-2 large | `microsoft/florence-2-large` via HF hub | ~770 MB | Optional; user chooses |

No new Python packages needed — `transformers`, `Pillow`, `onnxruntime-gpu`, `huggingface-hub` are all already in `requirements_base.txt`.

Florence-2 requires `flash_attn` for best performance but falls back cleanly to standard attention if not installed — don't add it as a hard dep.

---

### 17.4 Implementation Sequence

1. `services/image_prompt_service.py` — WD-tagger single-image wrapper + Florence-2 lazy loader
2. `api/routes/utilities.py` — add `POST /image-to-prompt` endpoint
3. Frontend page/tab — drop zone + results panel
4. Wire confidence threshold and caption length controls
5. (Later) Dataset-Tools deep-link when §11.2 is in progress

---

### 17.5 Open Questions

- **Model size default:** base is fast enough for interactive use; large adds noticeable latency on CPU fallback. Default to base, let user opt into large.
- **WD-tagger model variant:** the project currently uses whatever `wd14_tagger_model_dir` is set to in settings. For the prompt helper, auto-select the best available cached model or download `wd-eva02-large-tagger-v3` if nothing is cached.
- **Florence-2 on VastAI:** model downloads automatically via HF hub — no special VastAI handling needed beyond ensuring `HF_HOME` points somewhere with disk space.

---

**Document maintained by:** Ktiseos-Nyx-Trainer Project
