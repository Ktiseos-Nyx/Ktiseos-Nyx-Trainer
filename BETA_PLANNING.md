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

## 8. Attribution Requirements

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
**Status:** Template acquired, architecture planned  
**Source:** v0-generated template (`b_xNk9Kzq6SWm`), to be adapted — not dropped in wholesale  

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

**COMFY-5 (dream feature): "Test in ComfyUI" post-training shortcut**
- When a training job completes, show a "Open in ComfyUI" button on `TrainingMonitor`
- Clicking it navigates to `/comfyui` with the trained LoRA pre-loaded into the LoRA stack
- Requires: COMFY-1 through COMFY-4 complete + ComfyUI actually running on the instance

#### Notes
- ComfyUI is NOT bundled — the user installs and runs it separately on the same instance. The provisioning scripts (`vastai_setup.sh`, `provision_runpod.sh`) may eventually install it, but that's a separate decision.
- The template has workflow builders for `inpaint`, `controlnet`, and `adetailer` types referenced in `types.ts` but not yet built in `workflows/`. Those are future scope.
- Check whether `zustand` is already a dep before adding it — the stores use it.

---

### 11.2 Second Ecosystem App

**Priority:** TBD  
**Status:** Location TBD — user to provide path  

*(Placeholder — details to be filled in once the second app is shared.)*

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
| COMFY-2: Copy lib/stores/hooks layer | Integration | Tiny | ⏳ Not started |
| COMFY-3: UI page + navbar link | Integration | Small | ⏳ Not started |
| COMFY-4: Settings integration | UX | Tiny | ⏳ Not started |
| COMFY-5: "Test in ComfyUI" post-training button | Feature | Medium | ⏳ Not started |
| App 2: TBD | TBD | TBD | ⏳ Placeholder |

---

**Document maintained by:** Ktiseos-Nyx-Trainer Project
