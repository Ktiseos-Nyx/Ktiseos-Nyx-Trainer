# Beta Planning Document

**Current Version:** Alpha (v0.1.0-dev)
**Target:** Beta Release
**Last Updated:** 2026-05-30

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

### 1.5 Delete Image from Tag Editor (NEW FEATURE)
**Priority:** Medium (Beta QOL)
**Status:** ✅ Done (already implemented — full stack: backend endpoint, API client, frontend button with toast confirm and state removal).

### 1.6 Breadcrumbs on Tag Editor / Dataset Sub-Pages
**Priority:** Low (Beta QOL)

The dataset sub-pages (`/dataset/[name]/tags`, `/auto-tag`, `/tag-processing`) lack a breadcrumb trail (e.g. Home › Dataset › [name] › Tags). Bigger than it looks: requires threading the dataset name + current sub-page context through these routes rather than a single static breadcrumb.

### 1.7 Auto-Tagger Page Redesign — Match LoRA-Card Structure
**Priority:** Medium (UX — long-intended, never executed)
**Status:** Done (Unsure when, but it's done)

The **auto-tagger** page (`frontend/app/dataset/[name]/auto-tag/page.tsx`) — the batch WD14/BLIP/GIT tagging interface — was always meant to be fleshed back out to mirror the **LoRA/training card structure** (the shadcn `Card`-based layout in `frontend/components/training/cards/*.tsx`), but never got it. This is the **auto-tagger**, NOT the tag editor (`tags/page.tsx`, §1.1–1.6, done).

**Approach — rip-and-replace, not iterate.** If reformatting the existing page into the card structure is awkward, **don't** — delete it and build a fresh card-based page. The prior "that's too much work to format over time" read was wrong: the cost lives in *iterating on a bad base*, not in a clean rebuild. Greenfield it.

**Target:** same `Card` / `CardHeader` / `CardContent` componentry and visual rhythm as the training cards, so the auto-tagger reads as part of the same app instead of an older flat form.

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

**Issue CT-1: No checkpoint-specific validation** — 🚫 **WON'T DO (2026-06-10)**
- **Severity:** Medium → **Rejected**
- **Location:** `api/routes/training.py:33-253`
- **Problem:** Proposed VRAM warnings and LR range checks for checkpoint mode
- **Rationale:** This is conventional-wisdom nannying — exactly the bias pattern rejected in **LT-7** (removed DatasetCard VRAM warning) and **LT-5** (rejected algorithm-specific LR hints). Users on 48GB cards know their VRAM; LR ranges are empirical, not rules. "Warning" = quiet should = bias by suggestion.
- **Non-biased validation that DOES belong:** structural/hard errors only (missing required files, invalid enums, path traversal, negative/zero where structurally impossible). Everything else → docs/wiki.

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
- **Status:** ✅ **Done — verified in code 2026-06-06.** `kohya.py:301` now maps `ModelType.ANIMA: "anima_train.py"` in the CHECKPOINT `script_map`. No longer falls back to `fine_tune.py`.
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
- **Status:** ✅ Fixed (2026-07-02) — removed `target_alpha` from `LoRAResizeRequest` service model + API model, cleaned commented-out code, removed `new_alpha` from frontend result display

**Issue MG-2: Parameter naming inconsistency**
- **Severity:** Medium (confusion)
- **Location:** `services/lora_service.py:298` vs `services/lora_service.py:400`
- **Problem:** LoRA merge uses `--save_precision` but checkpoint merge uses `--saving_precision` (different Kohya scripts expect different arg names). This is correct behavior but confusing in the codebase.
- **Fix:** Add comments explaining the naming difference is intentional per-script
- **Status:** ✅ Fixed (2026-07-02) — annotated every `--save_precision` / `--saving_precision` usage with the target script name

**Issue MG-3: No subprocess timeout**
- **Severity:** Medium (reliability)
- **Location:** `services/lora_service.py:320-327, 428-435`
- **Problem:** `await process.communicate()` has no timeout. A hung merge process blocks the worker indefinitely.
- **Fix:** Use `asyncio.wait_for(process.communicate(), timeout=3600)` with proper cleanup
- **Status:** ✅ Fixed (2026-07-02) — all subprocess calls wrapped in `wait_for` with timeouts (1800s resize, 3600s merge)

**Issue MG-4: No CUDA availability check**
- **Severity:** Medium
- **Location:** `services/lora_service.py:312, 420`
- **Problem:** If user requests `device: "cuda"` but no GPU is available, the merge script will fail with an unhelpful error.
- **Fix:** Check `torch.cuda.is_available()` before passing cuda device, return clear error
- **Status:** ✅ Fixed (already had `_validate_device()` which checks `torch.cuda.is_available()`, verified 2026-07-02)

**Issue MG-5: SD3 merge not exposed**
- **Severity:** Low (missing feature)
- **Location:** `trainer/derrian_backend/sd_scripts/tools/merge_sd3_safetensors.py` exists but isn't wired up
- **Problem:** SD3 users can't merge models via the web UI
- **Fix:** Add SD3 merge option to the API and frontend
- **Note:** Deferred — `merge_sd3_safetensors.py` is a parts-to-full compositor (DiT + VAE + CLIP-L/G + T5XXL → single file), not a checkpoint averaging tool. Needs its own UI tab. Not a quick patch.

**Issue MG-6: Stdout not logged during merges**
- **Severity:** Low (observability)
- **Location:** `services/lora_service.py:320-327`
- **Problem:** Subprocess stdout is captured but never logged or surfaced
- **Fix:** Log at debug level, optionally stream to frontend for progress
- **Status:** ✅ Fixed (2026-07-02) — all 4 subprocess calls log stdout via `logger.debug()` after completion

**Issue MG-7: No merge progress reporting**
- **Severity:** Low (UX)
- **Problem:** Large checkpoint merges (2-7GB each) can take minutes with no progress indicator beyond a spinner
- **Fix:** Parse stdout for progress info, send via job status polling
- **Note:** Deferred — would need converting from `communicate()` to stdout streaming + job-system integration. ~1-2 day feature, not a quick patch.

**Issue MG-8: Missing output file existence check**
- **Severity:** Low
- **Location:** `services/lora_service.py:334`
- **Problem:** Reads file size without verifying output file actually exists
- **Fix:** Add `output_path.exists()` check before stat()
- **Status:** ✅ Fixed (2026-07-02) — `output_path.is_file()` check after all subprocess calls

**Issue MG-9: LoRA → Checkpoint merging (bake LoRA into base model)** *(feature request — 2026-05-28)* — **FIRST SLICE when picked up**
- **Severity:** Feature request (backlog)
- **Problem:** No path to bake a LoRA's weights directly into a base checkpoint and save a standalone merged model. Current "LoRA merge" combines LoRAs; "Checkpoint merge" is a checkpoint↔checkpoint weighted merge.
- **Backend status (verified in source 2026-05-30, NOT yet run):** the capability already exists. `networks/merge_lora.py` and `networks/sdxl_merge_lora.py` have `merge_to_sd_model()` which applies a LoRA onto the base model and saves a full checkpoint. This is an *expose-existing-capability* job, not a new feature.
- **Fix:** Add a "merge LoRA into checkpoint" mode (LoRA + base ckpt → merged ckpt) in the API + UI. Wire `merge_to_sd_model` path. See §3.3 for the agreed shape.
- **Status:** 🔧 **Built 2026-06-16 — wired, NOT yet run.** Full stack: `LoRAToCheckpointRequest/Response` (`services/models/lora.py`), `lora_service.merge_lora_to_checkpoint()` (clones `merge_lora()` + passes `--sd_model` → `merge_to_sd_model()` saves a full checkpoint; SD1.5 + SDXL only), route `POST /utilities/lora/merge-to-checkpoint` (`api/routes/utilities.py`), `utilitiesAPI.mergeLoraToCheckpoint()` (`api.ts`), and a new **"LoRA → Checkpoint"** tab (base-ckpt picker + LoRA multi-select w/ ratios + sd/sdxl toggle, sourced from trainer + ComfyUI dirs). py_compile/tsc/eslint clean. **Needs a real GPU run to verify the bake produces a valid checkpoint** (Dusk's test). MBW/`--lbws` (MG-10) still deferred.

**Issue MG-10: Block-weight (LBW-style) control for LoRA merge** *(feature request — 2026-05-28)*
- **Severity:** Feature request (backlog)
- **Problem:** No per-block (layer block weight) control when merging LoRAs — only a single global model/clip strength. Block weighting tunes individual UNet blocks (IN/MID/OUT) for finer style vs. likeness control.
- **Backend status (verified in source 2026-05-30, NOT yet run):** the LBW engine already exists. `networks/sdxl_merge_lora.py` accepts `--lbws`; `networks/svd_merge_lora.py` holds the machinery — `format_lbws`, `get_lbw_block_index`, and presets at **12/17/20/26 blocks** (`ACCEPTABLE = [12, 17, 20, 26]`; 26 = full `BASE/IN00-11/M00/OUT00-11` SuperMerger MBW layout). The frontend currently sends only a global strength and discards this. SD1.5 path (`merge_lora.py`) lacks `--lbws` — route SD1.5 block merges through the SVD path.
- **Fix:** Expose `--lbws` in the merge flow, **presets first** (see §3.3).
- **Status:** ❌ **Superseded/removed 2026-06-23.** Block weights are a *checkpoint-merge* concept, not a LoRA one (Dusk's call): the MBW presets encode an A↔B per-block blend, meaningless on a single LoRA. LBW removed from Merge-LoRAs + LoRA→Checkpoint. The real work moves to **MG-11** (clean-room checkpoint MBW). Dead `block_weights` fields + `BlockWeightPicker` component cleaned 2026-07-02.

**Issue MG-11: Block-weighted (MBW) checkpoint merge — clean-room SuperMerger port** *(2026-06-23)*
- **Why:** Block weights belong ONLY on the Merge Checkpoints tab (per-block A/B/C blend). MG-10 misplaced them on LoRA merge/bake — removed.
- **License constraint (HARD):** SuperMerger (`hako-mikan/sd-webui-supermerger`) is **AGPL-3.0**; this repo is **MIT**. Do NOT copy its source — that forces the whole project to AGPL. **Clean-room only:** read it to understand the algorithm, reimplement fresh in our style. Safe to lift as *data*: preset weight arrays (already in `block_weights.json`) and the `BLOCKID`/`BLOCKIDXL` name lists (facts/tables). Credit hako-mikan in `ATTRIBUTIONS.md` (courtesy). Reference: `mergers/mergers.py` — `blockfromkey`, modes list `["Weight","Add","Triple","Twice"]`, per-key blend loop. (Dusk has a local clone; path in memory.)
- **Reuse our kohya:** SuperMerger bundles its own kohya (`scripts/kohyas/`); we already vendor kohya in `derrian_backend/sd_scripts`. Build on ours; don't duplicate.
- **Target (full feature set, all exposed — no "advanced drawer" rationing; only genuinely-hard bits are *sequenced*, never hidden):**
  - Faithful `blockfromkey`: SD1.5 (26 = BASE/IN00-11/M00/OUT00-11) and SDXL (20 + VAE), `base_alpha` as element 0. NOTE the real mapping: SD1.5 `time_embed`→IN00, `.out.`→OUT11, unmatched→BASE (not what a naive guess gives — verify against SuperMerger output).
  - Modes: Weight `A·(1−α)+B·α`, Add `A+(B−C)·α`, Triple `A·(1−α−β)+B·α+C·β`, Twice (sequential). Triple/Twice need a second per-block curve (β for C) → second picker in UI for those modes.
  - calcmodes (cosine A/B, train-difference, extract, tensor/self), deep/elemental merge, block exclusions.
- **Build order (forced by dependency, not by hiding):** (1) own credited module: load/save via our kohya + faithful `blockfromkey` + Weight/Add/Triple/Twice + `base_alpha` + existing presets, wired to MergeCheckpointTab (mode selector, up to 3 model slots, 1–2 block pickers). (2) calcmodes. (3) deep/elemental/exclusions. Each layer ships exposed as it lands.
- **Verify gate:** merge-correctness — every layer needs a real GPU merge + generation check (Dusk) before "done"; cross-check block mapping against a known SuperMerger result.
- **Status:** 🔧 Phase 1 done (block weights removed from LoRA flows, `ae0e922`). **Foundation (step 1) built 2026-07-02** — `services/block_weight_merge.py` (clean-room, Weight/Add/Triple/Twice modes, `blockfromkey` for SD1.5/SDXL, preset support, `base_alpha`), `POST /checkpoint/merge-weighted` endpoint, frontend MergeCheckpointTab now has Basic/Block-Weighted toggle with mode selector, 2-3 model slots, dual preset pickers, and alpha sliders. Dead `block_weights` fields cleaned from LoRA merge APIs.

**🔴🔴 BUG MG-12 — EXTREMELY IMPORTANT (flagged 2026-06-24): Merge Checkpoints tab doesn't list ComfyUI checkpoints**
- **Symptom:** Merge Checkpoints shows only `pretrained_model/` models, NOT the ComfyUI checkpoints — even though they exist on disk and the page throws no error.
- **Established — DO NOT re-derive (Claude burned a session on wrong theories):** the ComfyUI checkpoints folder is HEALTHY. `ls -la {ComfyUI}/models/checkpoints/` showed a normal dir with 4 real checkpoints (`T3RR4KNXN30NSKRUNKLE_v05`, `naiComicsNAIXL_v10`, `ultraComix_v20`, `virtualDiffusion_v20`) + the `put_checkpoints_here` placeholder. **NOT a symlink, not missing, not a perms issue.** `/api/utilities/directories` returns JSON with no error. The "Jupyter can't open the folder" symptom is a SEPARATE Jupyter-sandbox quirk — IRRELEVANT (the merge tab uses the FastAPI backend, not Jupyter). The symlink / `is_dir()` / broken-FS / cwd theories were all WRONG.
- **TWO facts needed to localize — GET THESE FIRST, do not theorize without them:**
  1. Does the `/api/utilities/directories` JSON contain a `comfyui_checkpoints` key (and what path)?
  2. Do the 4 checkpoints actually render in the Merge Checkpoints UI list?
- **Maps cleanly:**
  - key present + UI empty → the *listing* step (`loadModelFiles` → `utilitiesAPI.listLoraFiles(dir, 'safetensors,ckpt', …)`) or a frontend filter is dropping them (backend found the folder fine).
  - key absent → backend `_comfyui_model_dirs()` didn't include it despite the folder existing → base/cwd path mismatch.
- **Code pointers:** `api/routes/utilities.py` → `_comfyui_model_dirs()` (~L36) + `get_directories()` (~L241); `api/routes/settings.py` → `get_comfyui_models_path()` (~L145, fallback `{cwd}/ComfyUI/models`); frontend `frontend/app/utilities/page.tsx` → `loadModelFiles()` (~L93) + `MergeCheckpointTab`.
- **ROOT CAUSE FOUND + FIXED 2026-06-24 (`a94854f`) — pure code asymmetry, no box data needed:** trainer dirs (`MODELS_DIR` etc.) come from `services.core.validation.PROJECT_ROOT` = `Path(__file__)...resolve()` (cwd-independent → `pretrained_model` always showed), but `get_comfyui_models_path()`'s fallback used `os.getcwd()`. When the backend's working dir ≠ project root, `{cwd}/ComfyUI/models/checkpoints` resolved wrong → `is_dir()` False → `_comfyui_model_dirs()` silently dropped `comfyui_checkpoints`/`comfyui_loras`. Fix: fallback now anchors on `PROJECT_ROOT`. NOT zustand (merge page has no store), NOT settings, NOT the filesystem (folder was healthy). **Verify the 4 checkpoints list on next deploy.**
- **Sibling latent bug (NOT fixed):** `settings.py:19` `SETTINGS_DIR = os.path.join(os.getcwd(), "user_config")` has the SAME `os.getcwd()` fragility → wrong settings dir if backend cwd ≠ project root. Anchor on `PROJECT_ROOT` when convenient.
- **Status:** ✅ Fixed (`a94854f`), pending deploy verification.

**🟠 MG-13 — De-bias the model-merging page copy (flagged 2026-06-24):** the merge page reads as AI-marketing-algo speak and ignores what Dusk's actually been asking for for *months*. Needs a copy/UX de-biasing pass — strip the marketing tone, align with his real merge workflow. **Spell the "months" asks out WITH Dusk before doing — don't guess them.** Ties to welcoming-not-corporate + the "don't write bias into the UI" rule.
- **Status:** ✅ Done (2026-07-02) — stripped marketing fluff from header, LoRA→Checkpoint banner, and resize info box. Functional/no-nonsense tone throughout.

**Issue MG-14: LoRA → Checkpoint bake for Anima (DiT)** *(feature request — 2026-06-29)*
- **Problem:** MG-9's bake path covers SD/SDXL only (`lora_service.merge_lora_to_checkpoint` is gated to `sd`/`sdxl`). Anima is a DiT (loaded via `UNETLoader`, with a separate Qwen text encoder + VAE), so there's no path to bake an Anima LoRA into a standalone diffusion-model checkpoint.
- **Backend status (verified in source 2026-06-29):** the merge engine already exists in the vendored backend — this is an *expose-existing-capability* job like MG-9, not new math, and needs no clean-room port or external tool (it's upstream's own Anima code, Apache-2):
  - `networks/lora_anima.py` → `LoRAInfModule.merge_to` (~L150): per-module bake, `weight += multiplier * (up @ down) * scale` (linear + conv variants), written back in-place.
  - `networks/lora_anima.py` → `LoRANetwork.merge_to(text_encoders, unet, weights_sd)` (~L603): bakes a whole LoRA into the loaded model in-place, routing `lora_unet_*`→DiT / `lora_te_*`→TE.
  - `networks/lora_anima.py` → `create_network_from_weights` (~L342): builds the network sized from the LoRA file.
  - The only genuinely new code is the write-out: `save_weights` saves the LoRA, not the baked base, so the merged checkpoint must be saved from the model's own `state_dict`.
- **Fix (4 steps):**
  1. Read `load_dit_model` (`anima_minimal_inference.py:221`) + `create_network_from_weights` (`lora_anima.py:342-378`) bodies to lock exact args.
  2. Add `custom/anima_merge_lora.py`: load DiT → `create_network_from_weights` → `network.merge_to` → `save_file(dit.state_dict())`. CLI `--base --models --ratios --save_to`, mirroring `sdxl_merge_lora.py`.
  3. Add an `anima` lane to `lora_service.merge_lora_to_checkpoint` (currently gated `sd`/`sdxl`), invoking the script via subprocess.
  4. Test: bake a real Anima LoRA into the Anima base, confirm it loads in ComfyUI `UNETLoader`.
- **Output:** save in the base's own key format so the result loads wherever the base did (no key conversion).
- **UI:** surface as a tab in the existing tabbed-card merge UI (`frontend/app/utilities/page.tsx`), LoRA → Checkpoint with model-type Anima.
- **Status:** 🔧 Built (2026-07-02) — `custom/anima_merge_lora.py` (load DiT → `create_network_from_weights` → `network.merge_to` → `save_anima_model` for DiT + separate TE save). Wired into `lora_service.merge_lora_to_checkpoint` as `anima` lane. API model accepts `"anima"` + `text_encoder_path`. Frontend LoRAToCheckpointTab shows 3-way arch toggle (SD1.5 / SDXL / Anima) with TE path input when Anima selected. **Needs a real GPU run.**

### 3.3 SuperMerger-lite Vision *(captured 2026-05-30)*

North star: a merge experience inspired by A1111 **SuperMerger** + the batteriesincluded merger, scoped to what a training tool can do safely. Guiding rule from Dusk: **"mix the best of proven tools, borrow presets that already work, never invent untested merge math."** Homegrown merge algorithms silently corrupt models — only wire proven paths.

**Decided scope:**
- ✅ **Presets-first block weights.** Ship named block presets (GRAD_V/FLAT/COSINE-style) as the entry point. Full 26-block manual entry is an opt-in *Advanced* mode added later. Easy first, difficult later. (Dusk personally uses presets; power users will want full manual eventually.)
- ✅ **First slice = MG-9** (bake LoRA → checkpoint). Backend ready.
- ✅ **Then MG-10** block-weight LoRA merge (presets), backend ready via `--lbws`.
- ❌ **Out, permanently:** merge-in-RAM and live gen/test (Dusk finds it confusing; also would need a ComfyUI bridge). Do not build.
- ❌ **XY/grid ratio plots** — out (needs a gen pipeline). Future only, would pair with a ComfyUI bridge.
- ❌ **SD3 (MG-5)** — **perma-deferred** (SD3 **deprecated per Stability AI**; also deprecated on Civitai + licensing uncertainty). No SD3 work unless a user explicitly asks. Don't rip out existing SD3/SD3.5 enum entries — just no new investment.

**Preset/merge-mode references (borrow proven values from these, don't reinvent):**

*Tier 1 — canonical preset sources (lift preset definitions from here):*
- `bbc-mc/sdweb-merge-block-weighted-gui` — **the canonical 25-block MBW GUI.** Provides all the named presets: GRAD_V, GRAD_A, FLAT, WRAP, MID12_50, OUT07/12, RING, SMOOTHSTEP, COSINE, cubic Hermite. `base_alpha` controls TE/VAE separately from UNet. 25 comma-separated values input. **Primary source for preset arrays.**
- `Faildes/Chattiori-Model-Merger` — CLI merger, ships a ready-made **`mbwpresets_master.txt`** (readable preset dictionary). 24+ modes (WS/SIG/GEO/MAX, Add/Smooth-Add/Multiply/Similarity/Train Difference, Triple/Tensor Sum, Sum Twice, DARE, Orthogonalized Delta, Sparse Top-k, cosine structure modes). 19/25-length block weights + elemental syntax. CPU-default, supports Flux. **Architecturally closest to how we call Kohya scripts (CLI/subprocess).**

*Tier 1.5 — Anima-relevant:*
- `kiygskr/sd-webui-supermerger-forgeneo-anima` — SuperMerger fork with **Anima + ForgeNeo support** (relevant to our Anima work). Caveat per its README: changes mostly authored by Codex, correctness unverified — reference only, don't trust blindly.

*Tier 2 — architecture / mode references:*
- `Ktiseos-Nyx/sdwebui-batteriesincluded-merger` — Dusk's own fork. 15 merge modes (Weight-Sum, Triple/Quad Sum, Sum Twice, Add/Multiply/Train Difference, DARE Power-up, Extract, interpolation variants), per-block weights, regex layer filtering, YAML weight editor.
- `silveroxides/sd-webui-untitledmerger` — fork of groinge's merger; roadmap includes "block weights reformatted for SuperMerger preset compatibility." Calc reuse for fast merges. References sd-webui-supermerger, safetensors-merge-supermario, MergeLM.
- `wkpark/sd-webui-model-mixer` — modern; sequential merge of up to 5 models, block-level rebasin, LoRA/LyCORIS↔checkpoint, no mandatory save. Good architecture model.
- `ddPn08/maji-merger` — JSON per-key alpha + longest-key-match targeting. Pattern reference for a future Advanced/elemental mode (early repo, ~2 commits).

*Parked — out of agreed scope (need a generation/scoring pipeline; revisit only with a ComfyUI bridge):*
- `s1dlx/sd-webui-bayesian-merger` — auto-tunes the 26 block params + base_alpha by generating & scoring images (Bayesian optimization). Same category as the cut live-gen/XY features. Future-only.
- `ashen-sensored/stable-diffusion-webui-vae-merger` — VAE-only merging (up to 3 VAEs, per-key alpha). Self-described experimental. Possible niche side-feature someday, not core.

- ⚠️ Several references are **AGPL-3.0** — reimplementing preset *values*/arrays is fine; copying source needs license care. (`bbc-mc`'s preset arrays and Chattiori's `mbwpresets_master.txt` are the safest lifts as data, not code.)
- ⚠️ **Confirmed 2026-06-23:** SuperMerger is **AGPL-3.0**, this repo is **MIT** → **clean-room reimplementation only** (see MG-11); never paste their source. Dusk is open to relicensing to (A)GPL *eventually* if enough copyleft deps accumulate — at which point direct source ports would become possible, but that's a deliberate future decision, not now.
- *(A third repo — a fork of Dusk's batteriesincluded-merger with extra presets — was sought but not located.)*

**Note:** none of the §3.2 backend findings have been *run* — all "backend ready" claims are verified-in-source only. First implementation step is a manual smoke test of `merge_to_sd_model` and `--lbws` before building UI on top.

> **Note (2026-05-28):** the merge-tool fixes in commit `80144fd` (paths, `size_formatted`, dirs) are **not yet verified** — needs a fetch-restart + manual run before MG items are re-audited as done.

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
- **Status:** ✅ **Done — verified in code 2026-05-30.** `kohya_toml.py:111` now reads `dataset["enable_bucket"] = self.config.enable_bucket`; the old hardcode survives only as a commented line (95) in the `general` block.

**Issue LT-3: network_train_unet_only added in checkpoint mode**
- **Severity:** Medium
- **Location:** `services/trainers/kohya_toml.py:346`
- **Problem:** `network_train_unet_only` is added to args unconditionally in `_get_training_arguments()`, but it's a LoRA-only parameter. For checkpoint/fine-tune training (`fine_tune.py`, `sdxl_train.py`, etc.) this argument is invalid and may cause Kohya to error out or warn.
- **Fix:** Wrap in `if self.config.training_mode != TrainingMode.CHECKPOINT:` like the network args section already does at line 184.
- **Status:** ✅ **Done — verified in code 2026-05-30.** `kohya_toml.py:384` now reads `if self.config.training_mode != TrainingMode.CHECKPOINT and self.config.network_train_unet_only:` — properly guarded against checkpoint mode.

**Issue LT-4: "Full" LoRA type semantic confusion**
- **Severity:** Low (UX/conceptual)
- **Location:** `services/models/training.py:36`, `services/trainers/kohya_toml.py:270-272`
- **Problem:** `LoRAType.FULL = "Full"` is exposed as a LoRA algorithm option (mapped to `lycoris.kohya` with `algo=full`), but "Full" means native fine-tuning (DreamBooth) which conceptually belongs in checkpoint training mode. Users selecting `training_mode=lora` + `lora_type=Full` get an unusual config that may conflict with `network_train_unet_only=True`.
- **Fix:** ~~(a) auto-switch to checkpoint mode~~ (rejected — there is **no** training-mode toggle on the LoRA tab; checkpoint training is a separate page `/checkpoint-training`, and Dusk confirms there should NOT be a checkpoint switch on the LoRA tab). ~~(b) add validation warning~~ (rejected — bias, see LT-5). **Decision (Dusk, 2026-05-30): (c) relabel for clarity — label change only, keep `algo=full` available.**
- **Status:** ✅ **Done — verified (tsc clean) 2026-06-06.** `LoRAStructureCard.tsx` `Full` description (line ~49) and `TYPE_HINTS.Full` (line ~55) rewritten: now "Full-rank LyCORIS network (algo=full) … still outputs a LoRA-format file — NOT a standalone checkpoint … For native checkpoint fine-tuning, use the Checkpoint Training page." No more "DreamBooth-style" checkpoint misread; `algo=full` still available.
- **Bonus (LT-5 applied to existing hints):** while in this file, pulled the **value-range nudges** from `TYPE_HINTS` — `IA3` ("requires 5e-3–1e-2 LR") and `DyLoRA` ("rank 64+") removed entirely; `Diag-OFT`/`BOFT` **trimmed** to keep the genuine disambiguation ("orthogonal — different from LoRA") and drop the "keep rank 4–16" nudge. Kept the disambiguation/behavioural-fact hints (`Full`, `TLoRA`). The IA3 hint was verified to be half-true lore (not in `lycoris/modules/ia3.py`, contradicted by `lora_ia3-sd15.json` LR 1.0 + Prodigy) — see CLAUDE.md "Empirical Lore — Interrogate It."

**Issue LT-5: No LyCORIS algorithm-specific validation**
- **Severity:** Low (UX)
- **Location:** `api/routes/training.py:33-253`
- **Problem:** No validation warns users about algorithm-specific learning rate ranges (e.g., IA3 needs 5e-3 to 1e-2, much higher than standard LoRA), or about unused parameters (e.g., `conv_dim`/`conv_alpha` only apply to LoCon/LoHa, `factor` only to LoKR).
- **Status:** 🚫 **WON'T DO in the UI — confirmed (Dusk, 2026-06-06).** Re-examined: not just LR-range *warnings* but **even purely informational UI hints** are out. Two parts considered:
  - **(a) per-algorithm LR scale** (IA3 ~5e-3–1e-2, etc.) — algorithm-intrinsic *fact*, not opinion, so it isn't "bias" in the strict sense. **But** as a UI hint it still sends the wrong signal: a "typical: X" next to a field reads as a quiet *should* and nudges behavior — bias-by-suggestion. Out.
  - **(b) no-op-parameter hints** (`conv_dim`/`conv_alpha` only for LoCon/LoHa; `factor` only for LoKR) — knowable from LyCORIS source, but same verdict: more UI hints clutter and confuse more than they help.
- **Where this info SHOULD live:** the project's **own docs/wiki** — reference material users go *find*, not in-form nudges. Deferred to the documentation re-jig (no date). Principle: a *few* good UI hints help; layering on more is net-negative. Generalizes [[feedback-stop-forcing-bias]] — the form must not editorialize, even with true facts.

**Issue LT-6: network_module field is misleading**
- **Severity:** Low (API clarity)
- **Location:** `services/models/training.py:197-199`
- **Problem:** `network_module` is a writable field with default `"networks.lora"` but its docstring says "derived from lora_type". The TOML generator always overrides it via `_get_network_config()`. API users who set this field will see it silently ignored.
- **Status:** ✅ **Resolved via documentation — verified in code 2026-05-30.** `training.py:210-212` now carries an explicit comment: *"network_module is always derived from lora_type… This default is for serialization only; the user-set value is not used in TOML generation."* No longer a silent trap. (Field is still writable-but-ignored by design; could be made read-only later if desired, but no longer misleading.)

**Issue LT-7: Conventional-wisdom value caps & precision limits baked into training UI/validation** *(Dusk, 2026-06-05)*
- **Severity:** Medium (violates the project's own bleeding-edge rule; blocks legitimate expert values)
- **Background:** A pile of academic/"safe-range" guardrails (the Furkan-flavored kind that leaked in from LLM training data, **not** added by us deliberately) are baked into the training form and Zod schema. They directly contradict `CLAUDE.md` → *"Bleeding Edge — Non-Standard Parameters Are Intentional"* (no artificial min/max/step on training params; use `step="any"`; only validate hard errors). This is the same class already rejected in **LT-5** — generalize that decision across the whole form.
- **Symptom Dusk hit (2026-06-05):** on the **Advanced tab**, the **noise** fields can't take precise values like `.357` — they snap because of `step={0.01}`. Confirmed: `AdvancedCard.tsx` `noise_offset` (line 75, `step={0.01}`), and siblings `adaptive_noise_scale` (86) and `multires_noise_discount` (124) same. Separately, **Min SNR gamma won't go above 20** (`min_snr_gamma .max(20)`) — the arbitrary ceiling.
- **Offenders — hard value caps (`frontend/lib/validation.ts`):**
  - `min_snr_gamma` `.max(20)` (the "can't burn it" cap) — line ~292
  - `unet_lr` / `text_encoder_lr` `.max(1, 'seems too high')` — ~189-197
  - `network_dim` / `network_alpha` `.max(1024)` **and `network_alpha` `.int()`** (rejects fractional alpha, which is legitimate) — ~214-216
  - `lr_power` `.max(10)`, `weight_decay` `.max(1)`, `max_grad_norm` `.max(10)`, `clip_skip` `.max(12)`, `guidance_scale` `.max(30)`, `lr_scheduler_number` `.max(100)`, `network_dropout`/`rank_dropout`/`module_dropout` `.max(1)`, etc.
- **Offenders — precision-limiting UI props:** ~137 `min=`/`max=`/`step=` props across the 9 cards in `frontend/components/training/cards/` (AdvancedCard 56, LoRAStructureCard 18, LearningRateCard 13, CaptionCard 12, DatasetCard 11, AugmentationCard 9, OptimizerCard 8, SavingCard 6, MemoryCard 4). The `step={0.01}`/`step={0.1}` ones are what block 3-decimal precision.
- **Offenders — value-judgment nags (remove):** `frontend/components/training/cards/DatasetCard.tsx:103` — *"⚠️ This batch size may exceed your VRAM! Consider reducing to 1-2."* That's conventional-wisdom nagging, not a hard error.
- **NOT bias — keep these:** destructive-action confirms are safety, not bias — `TrainingMonitor.tsx:298` (stop training), `PresetManager.tsx:337` (delete preset), `TrainingDefaults.tsx:48` (reset defaults). Leave them (though the raw `confirm()` calls should become shadcn `AlertDialog` per the a11y rule — separate concern). The `(recommended)` labels in dropdowns are soft informational defaults, not blocks — lower priority, audit only if they read as pushy.
- **Fix:** (1) In `validation.ts`, drop the value-range `.max()`/`.min()` opinions on training params; keep only structural/required checks (non-empty paths, correct type, divisible-by-64 resolution, enum membership). Remove `.int()` from `network_alpha`. (2) In the cards, switch numeric training fields to `step="any"` and remove artificial `max`/`min` (keep `min={0}`/`min={1}` only where a negative/zero is genuinely structurally invalid). (3) Delete the DatasetCard VRAM batch-size warning.
- **Cross-ref:** LT-5 (same anti-bias precedent, already accepted), `CLAUDE.md` Bleeding Edge section, `5.0.95` (validation single-source-of-truth — coordinate so the enum schemas aren't disturbed while loosening the numeric ones).
- **Status:** ✅ **Done — verified (tsc clean) 2026-06-06.** Three parts landed:
  1. **`validation.ts`** — stripped every conventional-range `.max()` ceiling on training params (min_snr_gamma, unet/text_encoder_lr, network_dim/alpha, conv_dim/alpha, dropouts, weight_decay, max_grad_norm, clip_skip, guidance_scale, lr_power, lr_scheduler_number, repeats/epochs/batch/grad_accum, bucket/token caps, etc.); removed `.int()` from `network_alpha`/`conv_alpha` (fractional alpha is legit); made `noise_offset`/`adaptive_noise_scale` bare numbers (negatives are legit). Kept only structural floors (≥1 epoch/batch/repeat, positive LR, divisible-by-64 resolution), enum membership, and required paths.
  2. **`FormFields.tsx` `NumberFormField`** (single choke point for all ~138 numeric fields) — decimal `step` props (0.01/0.1/0.00001) that forced 1–2 decimal places are now upgraded to `step="any"` for full precision (fixes the `.357` snap); whole-number structural steps ≥1 (e.g. resolution's `step={64}`) are still honored; `max` is accepted-but-ignored (documented).
  3. **`DatasetCard.tsx`** — deleted the "⚠️ batch size may exceed your VRAM, reduce to 1-2" nag (kept the neutral VRAM estimate in the field description).
- **Follow-up (cosmetic, low priority):** the now-inert `max={…}` props still sit on ~138 call sites (ignored by the component, documented). A mechanical sweep to delete them would tidy the cards but changes no behavior.

### 4.3 Training Dataset Features (backlog)

These are new capabilities (not bugs) for how datasets feed a LoRA run.

**LT-FEAT-1: Regularization image support**
- **Priority:** Medium (Beta)
- Let a run include a **regularization / class-image** set (Kohya `reg_data_dir` style) to curb overfitting and preserve the base model's prior. Needs: a way to point at / upload a reg-image folder, plumb it into the dataset TOML as a separate subset with `is_reg = true` (and its own `num_repeats`), and a UI field on the training config. Confirm exactly how the vendored sd-scripts expects reg subsets before wiring.

**LT-FEAT-2: Multiple folders per dataset (multi-concept / per-folder activation tags)**
- **Priority:** Medium (Beta)
- Support multiple subfolders within one dataset, each with its **own activation/trigger tag and `num_repeats`**, mapping to Kohya's multi-`subset` dataset config. Touches three layers: the dataset uploader/structure (create + manage subfolders), the TOML generator (emit one `[[datasets.subsets]]` per folder with its `class_tokens`/caption + repeats), and the training UI (per-folder trigger + repeats inputs). Larger than it looks because it changes the dataset → TOML shape, not just one field.

**LT-FEAT-3: Progress bar for ZIP uploads** *(Dusk, 2026-06-28)*
- **Priority:** Low–Medium (Beta polish)
- Image uploads show a progress bar; ZIP uploads don't. Mirror the image uploader's XHR `upload.onprogress` for the **upload** phase (% of bytes sent). The server-side **extraction** phase (`dataset_service.upload_zip`) only returns on completion, so after upload hits 100% show an indeterminate "Extracting…" state rather than faking extraction %. (Real per-file extraction progress would need the service to stream events — out of scope for v1.) Pairs with the existing image-upload progress UI.

**LT-FEAT-4: Preserve captions (and optionally latent caches) from an uploaded ZIP** *(Dusk, 2026-06-28)*
- **Priority:** Medium (Beta)
- Today `dataset_service.upload_zip` extracts **images only** — any `.txt` captions (and `.npz` caches) in the archive are silently skipped (`"Not an image file, skipped"`, dataset_service.py:339-340), so a user who zips a captioned dataset loses their captions **without warning**. Add an opt-in prompt on upload: *"This ZIP includes N caption files — import them too?"*
  - **`.txt` captions:** the common, wanted case → offer, default **on**. Pair to images by stem.
  - **`.npz` latent/TE caches:** offer separately, default **off**, with a note they're model/resolution/sd-scripts-version specific — stale caches can silently corrupt a run.
- **Security: not a concern in this flow (verified 2026-06-28).** The extractor flattens every entry to its basename (`dataset_path / file_path.name`, dataset_service.py:343), so there's no Zip Slip / path-traversal vector for any file type, captions included. The backend only streams bytes to disk — it never `np.load`s uploaded files, so the one real `.npz` danger (`numpy.load(allow_pickle=True)` → code execution) doesn't apply (sd-scripts reads latent `.npz` with `allow_pickle=False` at train time). Adding `.txt`/`.npz` to the allowlist is just "write more bytes to a validated dir."
- **Real gotcha — collision rename breaks pairing:** the current dedup renames a colliding image to `name_1.jpg` (lines 344-350) but would **not** rename its sibling `.txt`. If captions are imported, the rename must move the image **and** its `.txt` together (by stem) or caption→image pairing silently breaks.

### 4.4 torchao Optional Dependency (vendored optimizers)

**Priority:** Low (decision needed)

The vendored optimizer package (`trainer/derrian_backend/custom_scheduler/LoraEasyCustomOptimizer/low_bit_optim/`) is torchao-based, and `train_util.py`'s optimizer-signature introspection imports that chain — but `torchao` is in **none** of our requirements. Result: a **non-fatal** `WARNING ... determine default orthograd ... No module named 'torchao'` on every run (seen with CAME), after which it falls back to defaults (orthograd off, torchao state-storage auto-config skipped). Harmless for CAME and standard optimizers.

Verified: `orthograd`/`torchao` appear **only** in the vendored backend — nothing in our generators, presets, or frontend forces them. The warning is pure upstream optimizer-package behavior.

**Decision:** add `torchao` (enables the low-bit/torchao optimizers + orthograd auto-detect, silences the warning) vs. leave it out (heavy, torch-version-coupled dep that CAME doesn't need). If added: pin to our torch version and test on Windows + VastAI + RunPod first. This is ours to own — the vendored backend carries **our own patches** (applied ~Mar–May 2026 after pulled upstream updates broke things), so don't wait on upstream.

**Code findings (verified in `library/train_util.py:prepare_optimizer`, ~line 7589–7650, on 2026-05-30):**
- **orthograd does NOT depend on torchao.** `apply_orthograd` is detected (line 7646) and applied at the **network/param level** via `network.prepare_optimizer_params(..., apply_orthograd=...)`; the implementation `_paper_orthograd` is pure torch (imported from `LoraEasyCustomOptimizer/.utils`). The torchao ImportError only makes `inspect.signature(optimizer_class.__init__)` fail → fallback to `{}` (line 7642–7644) → can't auto-detect an orthograd *default* from the signature. Explicit `use_orthograd=True` still applies. So **orthograd is a red herring for the torchao decision.**
- **What torchao actually unlocks: `state_storage_dtype`** (line 7648+) — low-bit/quantized **optimizer-state storage** (8/4-bit, fp8). This is precisely the `state_storage_dtype=bfloat16 state_storage_device=cuda` that 4070-class **CAME** users need for VRAM (cross-ref the Anima §7.2 note + PR-1). Without torchao the low_bit_optim chain (`subclass_8bit/4bit/fp8`) can't import.
- **Why torchao surfaces — NOT a CAME dependency (clarified 2026-06-06):** `came.py` imports only torch + pytorch_optimizer; CAME is kozistr-based (`came.py:1-3`) + neggles stochastic rounding. torchao gets pulled in purely because the vendored `LoraEasyCustomOptimizer` collection's **eager `__init__`** imports its whole roster, including AO low-bit optimizers (copied from pytorch/ao) that hard-import torchao. That eager `__init__` is **67372a's fork's** (`refresh` branch); derrian's *original* had an **empty `__init__`** and never pulled torchao. So it's not derrian's packaging, not CAME — it's the fork's roster.
- **Per research (2026-06-06): the upstream we vendor has already moved on.** 67372a migrated installs pip→**uv** and now installs torchao via `--index-strategy unsafe-best-match` (2026-05-30) — resolving it against the installed torch instead of a hard pin, neutralizing the version-coupling risk. Our vendored snapshot is simply **behind**; the `No module named 'torchao'` warning is a *staleness artifact*, not a real problem (training proceeds; it's caught at `train_util.py:7639-7643` during orthograd-default introspection).
- **DECISION (Dusk, 2026-06-06) — SUPERSEDED 2026-06-22 (torchao WAS added standalone as `torchao==0.7.0`; see Status below): do NOT add torchao standalone / by hand.** Chasing torchao via more hand-vendoring is the exact "chase vendored updates by hand" pain we're trying to escape. **The right answer is the submodule strategy → see Section 20.** Submodule the backend and we inherit torchao (and everything else 67372a ships) naturally, far more often, **and align better with GPL-3.0 licensing** (reference upstream rather than copy GPL source into our tree). torchao's fate is folded into the §20 backend-delivery decision — resolved there, not here. The earlier "add it for CAME VRAM" lean is **withdrawn** (it also fought the CAME bf16-state NaN guard at `kohya_toml.py:515`).
- **Status:** ✅ **ADDED standalone (2026-06-22) as `torchao==0.7.0` in `requirements_base.txt`.** Supersedes the 2026-06-06 "won't add / fold into §20" decision: the submodule (§20) is now PARKED, and patch-update is the chosen delivery method (§20 revised 2026-06-22), so adding torchao directly IS the sanctioned path — not the hand-chasing it was once feared to be. **Why 0.7.0 specifically:** earliest version exposing both `torchao.utils.TorchAOBaseTensor` + `get_available_devices` (both on the eager import path via `adam.py → low_bit_optim → cpu_offload`); hard-pinned because unpinned grabs latest (0.17+, needs torch≥2.6, breaks on our torch 2.4.1 base = the "too high" install failure). **Cross-platform verified:** Linux=manylinux wheel, Windows/Mac=`py3-none-any` pure-Python wheel; torch-2.4.1-safe because its `_C` load is `try/except`'d and we use only the pure-Python `torchao.utils`. Zero torch dependency → never drags/pins torch. **Usage:** `state_storage_dtype`/`state_storage_device` via the freeform `optimizer_args` field (not auto-injected — CAME bf16-state NaN guard at `kohya_toml.py:515` stands).

### 4.5 Custom Optimizer Audit — Wiring Solid, Pins Drift *(Dusk, 2026-06-05)*

Full trace of the schedule-free / CAME / custom-optimizer chain on 2026-06-05. **Headline: the wiring is complete and correct end-to-end; the only real gap is dependency-pin hygiene.**

**✅ Verified solid (no action):**
- **Schedule-free fully wired:** frontend dropdown → `validation.ts` `OPTIMIZER_VALUES` → backend `OptimizerType` (`services/models/training.py:68-70`) → `CUSTOM_OPTIMIZER_PATHS` (`services/trainers/kohya_toml.py:236-238`) → vendored native support (`train_util.py:5454-5468`). (Supersedes the stale "schedule-free backlogged, needs enum/path entries" note — that work is done.)
- **The SF saved-LoRA gotcha is handled:** native schedule-free optimizers must switch to `.eval()` before save/sample or the saved weights are the wrong (train-mode) iterate. `train_network.py` calls `optimizer_eval_fn()` before every save/sample (lines 1880, 2080, 2217, 2264) and `optimizer_train_fn()` after (1889, 2132, 2248). Same pattern in `flux_train.py` / `anima_train.py`. SDXL/SD15 SF training produces correct weights.
- **CAME NaN guard present:** `kohya_toml.py:449-450` deliberately does not inject `state_storage_dtype/device` for CAME (bf16 state → NaN; fp32 default is correct). See §4.4 tension note above.

**⚠️ LT-OPT-1: Dependency pins disagree across three requirements files**
- **Severity:** Low–Medium (reproducibility / CLAUDE.md "pin versions" rule)
- **Background:** the installer editable-installs the vendored dirs (`installer.py:606-612`) and installs `requirements_base.txt` — so **`requirements_base.txt` is the file that actually sets versions**; the pins inside `sd_scripts/requirements.txt` and `lycoris/requirements-kohya.txt` are effectively decorative (the vendored backend was never a real submodule, so those loose `.txt` pins don't drive the install). CodeRabbit recently added pins (good) — finish making them consistent.
- **Drift (effective = `requirements_base.txt`):**
  - `came-pytorch` — **unpinned** in base; vendored says `~=0.1.3`. Pin base to `~=0.1.3`. (A breaking release would silently break installs.)
  - `schedulefree` — base `==1.4`, vendored `~=1.4.1`; we install the *older* one than the code targets. Bump base to `~=1.4.1`.
  - `pytorch_optimizer` — base/sd_scripts `3.10.0`, but `lycoris/requirements-kohya.txt` carries a **stale `==3.1.2` pin + comment** ("init_group abstract method added in 3.6.0") implying LyCORIS wanted `<3.6.0`. We run 3.10.0 and `sd_scripts` agrees, so the fork was presumably updated — but the stale pin/comment is a trap. Confirm LyCORIS runs clean on 3.10.0, then delete the stale pin/comment.
- **Unconfirmed (1-line check):** assumed `pip install -e` ignores the vendored `.txt` pins (reads `setup.py`/`pyproject` `install_requires` instead). Confirm LyCORIS's `setup.py` doesn't `-r requirements-kohya.txt` before relying on "base wins."
- **Status:** ✅ **Done — verified 2026-06-06.** `requirements_base.txt`: `came-pytorch` → `came-pytorch~=0.1.3` (was unpinned), `schedulefree==1.4` → `schedulefree~=1.4.1` (matches vendored). **Stale LyCORIS pin resolved by confirmation, not edit:** `lycoris/setup.py:13` declares `install_requires=["torch", "einops", "toml", "tqdm"]` and does NOT read `requirements-kohya.txt`, so that file's `pytorch_optimizer==3.1.2` pin is **decorative** (pip never sees it on editable install) — "base wins" confirmed. Left the vendored file untouched (it's overwritten on every upstream sync; editing it is pointless + against the don't-touch-vendored rule). `pytorch_optimizer==3.10.0` in base stands.

**🟡 LT-OPT-2: Schedule-free scheduler default (optional, low priority)**
- Schedule-free optimizers self-schedule and want a `constant` LR scheduler; the frontend default is `cosine_with_restarts` with no smart default. **Per the bleeding-edge/anti-bias rule (LT-7, LT-5): a smart *default* when an SF optimizer is selected is acceptable; a *warning/block* is not.** Also note `flux_train_network.md:580` — validation loss is unsupported with SF optimizers, so that UI combo silently no-ops. Low priority; smart-default only, never a nag.

**🟢 LT-OPT-3: Auto-populate optimizer dropdown from a single source (low priority, Dusk 2026-06-22)**
- Today adding an optimizer is a TWO-file edit: `OPTIMIZER_VALUES` (`validation.ts`) for the schema/type, AND the hardcoded `{ value, label, description }` list in `OptimizerCard.tsx` (constrained via `satisfies` but not generated). The "only change needed" comment was doc-rot — fixed 2026-06-22.
- **Refactor:** make `OPTIMIZER_VALUES` an array of `{ value, label, description }` objects; derive the Zod enum (`z.enum(values.map(v => v.value))`) AND the dropdown from it. Not free because the labels/descriptions carry real UX value and must relocate, not vanish. Clean afternoon job, behind trainer correctness.

**🟡 LT-OPT-4: Wire the 3 deferred schedule-free/wrapper optimizers (Dusk 2026-06-22)**
- `AdamWScheduleFreePlus`, `NorMuonScheduleFree`, `SODAWrapper` were pulled from 67372a + staged in-tree (commit `f479989`) but NOT exposed. (`AMUSE`/`MODA`/`SODA` — standard — shipped exposed in `2e7a156`.)
- **The dispatch problem:** kohya detects schedule-free by name → routes to the external `schedulefree` package (`train_util.py:5454`); `NorMuonScheduleFree`'s name would misroute, and SF opts need the `optimizer_eval_fn()`/`optimizer_train_fn()` train/eval toggle that native SF gets (see §4.5 line re: `train_network.py` save/sample eval calls) — custom-path resolution may not provide it → saved weights would be the wrong train-mode iterate. Verify/handle that toggle for custom-path SF before exposing, or they train silently wrong. `SODAWrapper` purpose still unconfirmed (possibly SF too).

---

## 5. Miscellaneous Bug Fixes for Beta

### 5.0 WandB / Logging UI Missing Entirely

**Issue UI-1: WandB and logging fields have no UI in the training form** — ✅ **Done 2026-05-25**
- **Status:** `LoggingCard.tsx` added; WandB/logging fields configurable in UI. Paired with LT-1 (wandb_key wired to env).

### 5.0.5 Dashboard — Bulldoze + Rebuild

**Issue UI-2: Dashboard is a first-day, zero-context wrapper artifact** — ⏳ **Not started (later — ~2 months)**
- **Severity:** Low (current dashboard is acceptable/cleaner now — just incomplete, not blocking)
- **Location:** `frontend/app/dashboard/page.tsx`
- **Plan:** Don't patch — **rebuild from scratch.** Current dashboard predates real project context; incremental fixes aren't worth it.
- **Rebuild requirements (carry forward):**
  - Surface all real routes — currently missing: `/checkpoint-training`, `/dataset/auto-tag`, `/dataset/tags`, `/dataset-uppy`, `/huggingface-upload`, `/changelog`, `/models/browse`.
  - If it includes an Active Jobs widget: take `Job[]` from day one (one job now, many once §5.1 queue lands) — single-slot = a UI rewrite later.
- **Blocked by:** backend work finished + a database layer landing first. Sequence after those.

### 5.0.7 Listener/Request Cancellation Console Noise

**Issue UI-3: "Listener cancelled" / AbortError messages on page navigation** — ✅ **Done 2026-05-20**
- **Status:** Resolved — AbortController refactor landed, errors caught and suppressed. Residual noise attributed to low-RAM thrashing (not reproduced on adequate RAM). Won't chase further.

### 5.0.8 Training Log Polling - Updates Feel Inconsistent

**Issue UI-4: Training logs only update "when they feel like it"** — ✅ **Done 2026-04-30**
- **Status:** Fixed-cadence + visibility-aware polling implemented in `api.ts:pollLogs()`. Backend uses Python `-u` flag. Residual sporadicity = backend emit cadence (Kohya/accelerate buffering), not poller. Re-check only if regresses.

### 5.0.9 Training Monitor — tqdm ETA and Step Progress Parsing

**Issue UI-5: No ETA, step count, or percentage in the training monitor** — ✅ **Done 2026-05-30**
- **Status:** Backend parses tqdm → `step_progress` events; `TrainingMonitor.tsx` consumes them with `getTimeRemaining()`. Minor residual: progress "bursts at end" (drain logic, 15s window) — expected, low priority.

### 5.0.95 Validation Schema / UI Dropdown Single Source of Truth

**Issue PR-0: OptimizerSchema and LRSchedulerSchema drift from UI dropdown**
- **Severity:** Medium (recurring bug class — caused #369)
- **Flagged by:** CodeRabbit on PR #370
- **Problem:** `OptimizerSchema` and `LRSchedulerSchema` in `validation.ts` are separate string arrays from the dropdown options in `OptimizerCard.tsx`. Every time a new optimizer/scheduler is added to the UI, it must be manually added to the schema too — and that drift is exactly what broke CAME/Compass/schedule-free optimizers before PR #370.
- **Fix:** Export `OPTIMIZER_VALUES` and `LRSCHEDULER_VALUES` as `as const` tuples from `validation.ts`. Build the Zod schemas from those constants. `OptimizerCard.tsx` dropdown keeps its own labels/descriptions but TypeScript can enforce values only come from `OPTIMIZER_VALUES`. Future additions require touching one place.
- **Status:** ✅ **Done — verified 2026-06-06.** `validation.ts` exports `OPTIMIZER_VALUES` + `LR_SCHEDULER_VALUES` as `as const` tuples and builds both Zod enums from them (`OptimizerSchema`/`LRSchedulerSchema`). Both dropdowns are typed against those tuples — `OptimizerCard.tsx:70` `satisfies Array<{ value: OptimizerValue }>` and `LearningRateCard.tsx:95` `satisfies Array<{ value: LRSchedulerValue }>` — so any drift between a dropdown and the schema is a **compile error**. Single-source-of-truth confirmed.

---

### 5.0.96 Legacy Preset Audit — bmaltais Format Migration

**Issue PR-2: ~30 built-in presets still use old nested `config:{}` format with legacy field names**
- **Severity:** Medium (all old-format presets were silently broken before PR #370)
- **Background:** Most built-in presets were imported from bmaltais's Kohya SS gradio scripts. They use legacy field names (`optimizer`, `epoch`, `learning_rate`, `batch_size`, `lr_warmup`, `max_resolution`, `dataset_repeats`) nested under a `config:` block — a completely different schema from the current `TrainingConfig`.
- **Current state:** PR #370 added `normalizeLegacyPresetFields()` so these presets now *load* correctly via the mapping layer. But the files themselves are still in the old format, which is fragile.
- **Fix:** Audit all presets in `presets/` and convert any still using the old nested format to the flat format used by newer presets (e.g. `lora_SDXL - Illustrious-XL CAME Conservative v1.0.json`). Remove obsolete fields, fix legacy field names, ensure types are correct (strings → numbers where needed).
- **Status:** 🟡 **Partially done — verified 2026-06-06.** The `normalizeLegacyPresetFields()` shim (PR #370) is present, so legacy presets **load** correctly. BUT the migration deliverable itself isn't done: **31 preset files still use the nested `config:{}` legacy format** (e.g. `lora_SDXL - LoRA AI_characters standard v1.0.json`, the `EDG_*` set, `lora_lokr-sd15.json`, …). They work only via the shim. **Outstanding:** convert those 31 files to flat format (mechanical, low-risk, no GPU) so they don't depend on the compatibility layer. Not done in this session to keep the PR focused.

---

### 5.0.97 Categorical Preset Library Expansion — Peer-LoRA Study *(Dusk, 2026-06-09)*

**Goal: add new *categorical* presets covering training patterns that are common across real-world peer LoRAs but that our `presets/` folder doesn't represent yet.**
- **Severity:** Low (library enrichment, not a bug) — but high *value* for users picking a starting point.
- **Status:** ⏳ Parked — corpus read + tooling built, the actual gap-walk and preset authoring are not started. (Captured on a flat-brain day; resume fresh.)

**Method — IMPORTANT, this is the part that's easy to get wrong:**
- **Dusk leads, section by section, qualitatively.** The empirical layer is *his knowledge of what each LoRA actually is* (subject, what worked, output quality) — the metadata can't see that. He walks a section, names the gap.
- **The metadata stats are BACKING, not the driver.** They sanity-check a hunch; they don't lead and they shouldn't clog the flow. No aggregate tables / histograms unless explicitly asked — that's what derailed the first pass.
- **Authority order:** Dusk's hands-on training results > paper-reading of preset fields. Community experts (Novowels, Citron, kudou-reira) are authoritative — don't second-guess their values.
- **Presets don't pin training outcome:** repeats are dataset-side, steps are derived, batch matters — so matching another LoRA's LR/dim does **not** make a new preset redundant. Coverage is about *kind of recipe*, not exact numbers.

**Tooling already built (local scratch in `temp/`, header-only, no deps):**
- `read_loras.py` — reads safetensors `__metadata__` header-only; emits per-file summary + `lora_study_full.csv`.
- `lora_study_full.csv` — all 123 peer LoRAs, one row each (optimizer / dim·alpha / unet_lr / scheduler / network).
- `preset_gaps.py` — flags optimizer×model combos peers use that our `presets/` has **no** preset for (the new-preset hunt). Backing only.
- `provenance.py` — checks whether a cluster (e.g. the Adafactor-fixed group) is one source or many.

**Corpus:** 123 peer LoRAs at `C:\Users\dusk\Downloads\Loras to study for Claude`.

**Candidate clusters spotted as starting backing (NOT conclusions — Dusk's read decides):**
- **Adafactor-fixed workhorse** — `dim32 / a16 / lr 5e-4 / cosine_with_restarts`, plain `lora` on SDXL/Illustrious. Most common signature in the corpus; likely one tool/author (that's what `provenance.py` is for). Check coverage vs `finetune_adafactor.json` / `folk_horror_style_adafactor.json`.
- **Flux/Chroma low-dim** — `dim2 / a16 / lr 5e-4 / cosine_restarts`, `lora_flux`. Distinctive high alpha:dim. Check vs `chroma_style_experimental.json` / `flux_*`.
- **Prodigy Illustrious/PDXL** — `lr 1.0` adaptive, dim 8–32, cosine/constant. Check vs `kudou-reira_prodigy.json` / `faetastic_sdxl_prodigy.json`.
- **CAME NoobAI** — `dim8 / a4 / lr ~3–6e-5 / cosine` — likely already covered by `came_character_*`.

---

### 5.1 Preset Optimizer Args Contamination

**Issue PR-1: optimizer_args field picks up general training args from community presets**
- **Severity:** Medium (causes cryptic training failures)
- **Status:** ✅ **Done — verified 2026-06-06.** Inspected every `optimizer_args` value across all presets — none carry contaminated training-level args anymore (no `state_storage_dtype=`/`state_storage_device=`/`mixed_precision=`/`fp8_base=` etc.); all values are legitimate optimizer args (`weight_decay`, `betas`, `eps`, `decouple`, `d0`, `d_coef`, `scale_parameter`, …). The Citron-style precision/device contamination is gone. (Tokenizer robustness for *parsing* those args is tracked separately in PR-1b.)
- **User-reported:** Citron's Adafactor preset stuffed precision/device args (`state_storage_dtype=bfloat16 state_storage_device=cuda` style) into `optimizer_args`. Adafactor then threw `ValueError: not enough values to unpack` because those aren't valid optimizer args.
- **Root cause:** Community presets (and possibly our own) miscategorise general training args as optimizer_args. These are actually top-level training fields (`mixed_precision`, `fp8_base`, etc.) that got bundled into the freetext `optimizer_args` blob.
- **Workaround:** Manually clear `optimizer_args` before training if switching optimizers or loading community presets.

**Fixes:**
1. **Preset cleanup** — audit all bundled presets, move any precision/device args out of `optimizer_args` into proper top-level fields
2. **UX warning** — when `optimizer_args` is non-empty and optimizer type changes, warn the user: "These args may be specific to a different optimizer — clear them?"
3. **Validation** — before training starts, validate that `optimizer_args` entries look like actual optimizer args (key=value pairs that the selected optimizer recognises), not training-level settings
4. **Community preset naming** — if a preset bundles optimizer-specific args, the name should make that clear (e.g. "Citron Adafactor - SDXL" not just "Citron")

**Issue PR-1b: optimizer_args copied from LoRA metadata crash the trainer** *(Dusk, 2026-06-06)*
- **Status:** ✅ **Done — verified 2026-06-06.**
- **Trigger:** copying optimizer args from a reference LoRA's metadata to emulate its training style. Metadata formats them **comma-separated** with **spaces inside tuple values** — e.g. `weight_decay=0.08,betas=(0.99, 0.999, 0.99995)`. Our generator tokenised with `shlex.split` (whitespace-only), which shattered the tuple into fragments lacking exactly one `=`, so the vendored parser (`train_util.py:7610`, `key, value = arg.split("=")`) crashed with `ValueError: ... unpack`. Same crash *class* as PR-1, different *trigger* (metadata paste, not preset contamination) — so the earlier fix didn't cover it.
- **Fix (BOTH config generators — Python and Node):** a paren-aware tokenizer that splits on commas/whitespace only at the **top level** (separators inside `()`/`[]`/`{}` are preserved), then strips inner whitespace → clean `key=value` tokens. Handles metadata-style (comma + spaces) AND Kohya CLI-style (space, no inner spaces); clean presets unchanged. Structural-hard-error hardening, not bias.
  - **Python:** `services/trainers/kohya_toml.py` — new `_tokenize_optimizer_args()` replaces `shlex.split`; removed the now-unused `import shlex`.
  - **Node:** `frontend/lib/node-services/config-service.ts` (the "ported from kohya_toml.py" training-config generator) had the **same bug** — `config.optimizer_args.trim().split(/\s+/)` shattered tuples AND its `weight_decay`-injection check then ran on the fragments. New `tokenizeOptimizerArgs()` (line-for-line port of the Python one) replaces it.
- **`network_args` checked, NOT affected:** verified on both backends — it's built from hardcoded per-algo lists (`['algo=…']`) + individually-typed conv/block fields, never a freetext paste blob, so it can't hit this crash. No fix needed (corrected an earlier guess that it was a freetext blob).
- **Regression caught during PR re-validation (2026-06-06):** the first cut of the tokenizer (paren-aware only) would have **broken two existing preset formats** that the old `shlex.split` handled — **quoted args** (`"weight_decay=0.1" "betas=0.9,0.99"`) and **paren-less comma tuples** (`betas=0.9,0.99`). Revised the tokenizer to also be **quote-aware** (strip quotes, never split inside them) and to **merge `=`-less fragments** back into the preceding token (re-joining paren-less comma tuples). Both Python (`kohya_toml.py`) and Node (`config-service.ts`) updated to match.
- **Verified:** tested the real Python function against a **13-case matrix of actual preset formats** (quoted, paren tuple, bracket tuple, bare-comma tuple, AdaFactor flags, the metadata paste, empty) — all produce clean `key=value` tokens AND survive the simulated vendored `arg.split("=")` + `ast.literal_eval`. Node tokenizer parity-tested against the same cases + `tsc --noEmit` clean. ⏳ **Not yet run end-to-end on a real CAME job** — that's Dusk's to run (only he can launch CAME training).

---

### 5.1 HuggingFace Upload - Form State Doesn't Persist

**Issue HF-1: HF upload form loses all data on page navigation** — ✅ **Done 2026-04-30**
- **Status:** Form state migrated to Zustand persist; token pre-filled from settings; owner/repoType persist in localStorage.

### 5.2 Cross-Tab Form Persistence — App-Wide Gap

**Issue FP-1: Most forms lose their state on navigation (broader than HF-1)** — ⏳ **Not started**
- **Severity:** Medium (recurring UX pain across the app)
- **Problem:** HF-1 fixed HF upload page, but ComfyUI Generate, training form, dataset/auto-tag, batch/util forms still lose state on navigation.
- **Approach:** Standardize Zustand `persist` pattern per-form (matching HF-1). Storage per field: localStorage for convenience defaults; sessionStorage/settings for sensitive; never persist per-run transient state.
- **Priority:** ComfyUI Generate persistence most-wanted.

### 5.3 SyntaxWarning Log Noise — Escape Sequences *(Dusk, 2026-06-05)*

**Issue SW-1: `invalid escape sequence` warnings flood training/tagging logs** — ✅ **Done 2026-06-06**
- **SW-1a:** `tag_images_by_wd14_tagger.py:881` → raw string; `python -W error::SyntaxWarning -m py_compile` clean.
- **SW-1b:** `kohya.py:_build_env()` sets `PYTHONWARNINGS=ignore::SyntaxWarning` for training subprocess only (vendored silenced, ours still surface). No vendored patches.

### 5.4 Generate Tab Falls Back to Literal ComfyUI *(Dusk, 2026-06-05)*

**Issue GEN-1: Generate page route collides with ComfyUI proxy prefix** — ✅ **Done 2026-06-06**
- **Root cause:** `/comfyui` route shadowed by `server.js` proxy exact-match.
- **Fix:** `app/comfyui/page.tsx` → `app/generate/page.tsx` (route `/generate`), navbar link updated. LoRA Manager links and proxy rules unchanged. `tsc` clean.
- **Follow-up:** tighten `COMFYUI_ROOT_PREFIXES` (`/checkpoints`, etc.) so generic names can't shadow future routes.

### 5.5 Form State & Field-Linkage Bugs *(Dusk, 2026-06-28)*

**Issue PRESET-1: Preset load is non-deterministic — sticky omits + wrong clears** — ✅ **Fixed 2026-07-02**
- **Severity:** Medium (recurring config-correctness pain; can silently ship a run with hyperparameters the user didn't choose)
- **Problem (bidirectional):** loading a preset (1) does **not clear** fields the preset omits → stale values ride along (e.g. the "Train UNet Only" tick stays set; a `lr_warmup_ratio` set by one preset persists when you next load one that omits it), AND (2) **does clear** some fields it shouldn't — **confirmed live: swapping to a CAME preset wipes the user's pre-selected base model (Illustrious/NAI)**, a job field that should survive the load. Compounded by `useTrainingForm.ts` hydrating last-saved state from localStorage (falling back to server) on mount (`readStoredConfig` ~L211, hydration `useEffect` ~L294), so the "default" a user sees is their **persisted prior state**, not `defaultConfig` (L49). Net effect: the warmup/optimizer/unet-only values feel "sticky" and un-attributable.
- **Root cause:** `loadPreset` (`useTrainingForm.ts:348`) merges a *partial* preset over current values — there is no model of what a preset owns vs job/user state.
- **Fix (no runtime dirty-tracking needed):** statically classify each `TrainingConfig` field as **recipe** (preset owns: LR, optimizer, scheduler, warmup, dim/alpha, dropout, …) or **job** (this-run: dataset/model/output paths, project name, trigger words). Preset load = reset **recipe** fields to `(defaultConfig → then preset overrides)`, leave **job** fields untouched. Deterministic; kills both failure modes at once. (v1 deliberately does not preserve a manual recipe tweak across a preset swap — overriding recipe edits is expected when loading a recipe.)
- **Files:** `frontend/hooks/useTrainingForm.ts` (loadPreset, defaultConfig, hydration effect), `frontend/components/training/PresetManager.tsx`.

**Issue SCHED-1: `cosine_annealing` & `rex` schedulers hide their restart-count field** — ✅ **Fixed 2026-07-02**
- **Severity:** Low–Medium (silent loss of UI control over a real hyperparameter)
- **Repro:** pick **Cosine Annealing (Warm Restarts)** or **Rex (Warm Restarts)** in the LR scheduler dropdown → the "Number of Restarts" field disappears, so the user can't set it and is stuck with whatever `lr_scheduler_number` is already in the config (default 3 / stale).
- **Root cause:** `LearningRateCard.tsx:99` gates the `lr_scheduler_number` field on `scheduler === 'cosine_with_restarts' || scheduler === 'polynomial'` only — it omits the two vendored warm-restart schedulers. But `kohya_toml.py:409` passes `lr_scheduler_num_cycles = lr_scheduler_number` for **every** scheduler, and `rex`/`cosine_annealing` are restart-based (`CUSTOM_SCHEDULER_PATHS`, kohya_toml.py:291-294), so they DO consume the cycle count.
- **Fix:** add `'cosine_annealing'` and `'rex'` to the `LearningRateCard.tsx:99` conditional and extend the label/description branch (L103-104) to read "Number of Restarts" for them. Confirm the vendored `CosineAnnealingWarmRestarts` / `RexAnnealingWarmRestarts` consume `num_cycles` (strongly implied by the unconditional pass at L409).
- **Files:** `frontend/components/training/cards/LearningRateCard.tsx`; verify `trainer/derrian_backend/.../LoraEasyCustomOptimizer/{CosineAnnealingWarmRestarts,RexAnnealingWarmRestarts}.py`.

### 5.6 ComfyUI Generate — Checkpoints Not Found (single-source discovery) *(Dusk, 2026-06-28)*

**Issue GEN-CKPT-1: Generate UI lists LoRAs but not checkpoints** — ⏳ **Not started**
- **Severity:** Medium-High — blocks generation (no base model selectable) after any `extra_model_paths` / folder change.
- **Symptom:** the Next.js Generate UI finds **loras** but **not checkpoints**, even though ComfyUI's **LoRA Manager *can* see the checkpoints**. The gen checkpoints live in the training dir `pretrained_model/`, surfaced to ComfyUI via `extra_model_paths.yaml` (`installer.py:591`); a recent folder/path change broke that mapping. (There is NO ComfyUI `/object_info` call here — discovery is via `/models/{folder}`.)
- **Root cause:** asymmetric discovery in `frontend/lib/comfy/useComfyModels.ts:86-94`. **LoRAs have TWO sources** — `safe('loras')` (ComfyUI `/models/loras`) **+ `comfyClient.getLmLoras()`** (LoRA Manager cache via `/api/lm/loras/list`, which indexes every `extra_model_paths` root incl. training dirs). **Checkpoints have ONE** — only `safe('checkpoints')` (ComfyUI `/models/checkpoints`). When the yaml/path breaks, ComfyUI's checkpoint folder returns empty and checkpoints have **no LM-cache fallback** like loras do. LM has them indexed; the frontend just never asks LM for checkpoints. (The "we broke it recently" = the LM fallback was added for loras but never mirrored for checkpoints.)
- **Fix (mirror the loras path):**
  1. `client.ts`: add `getLmCheckpoints()` mirroring `getLmLoras()` (`client.ts:117`), hitting LoRA Manager's checkpoint endpoint.
  2. `useComfyModels.ts`: fetch checkpoints from **both** `safe('checkpoints')` and `getLmCheckpoints()`, merge+dedupe, and have the consumer fall back to the LM source when ComfyUI's is empty — the exact `loraModels` / `lmLoraModels` pattern already in the hook.
- **One unknown to confirm FIRST (don't guess a ComfyUI endpoint):** LoRA Manager's exact checkpoint route — likely `/api/lm/checkpoints/list?page_size=9999` mirroring the loras endpoint; confirm from the running LM API or LM's source.
- **Files:** `frontend/lib/comfy/client.ts`, `frontend/lib/comfy/useComfyModels.ts`. ~15 lines once the endpoint string is confirmed.

**ACTUAL ROOT CAUSE found via live testing (2026-06-28) — separate from the frontend fallback above:**
- LoRA Manager *does* see `pretrained_model` (checkpoints) + `output` (loras) — it indexes the yaml mappings; SDXL workflows (LM checkpoint loader) work fine.
- **guy90 (Anima) uses `UNETLoader` (comfy-core), which loads from the `diffusion_models/` folder — and `extra_model_paths.yaml` maps `checkpoints`/`loras`/`vae` but has NO `diffusion_models` line.** So `UNETLoader` can't find the Anima base (`anima-base-v1.0`) sitting in `pretrained_model/`. SDXL (CheckpointLoaderSimple → `checkpoints`, mapped) works; Anima (`UNETLoader` → `diffusion_models`, unmapped) doesn't. Same gap hits our Generate UI's diffusion_models list (`useComfyModels.ts:87`, `safe('diffusion_models')`).
- **Root cause = a MISSING yaml mapping, not a stale scan or single-source frontend.** Anima/UNET models simply have no folder mapping.
- **Fix (the real unblock):** add `diffusion_models: pretrained_model` (and `unet: pretrained_model` for older ComfyUI naming) to `extra_model_paths.yaml` — live on the box AND permanently in **`installer.py:591`** (the yaml generator) so every install ships it. Then `supervisorctl restart comfyui`.
- The `getLmCheckpoints` frontend hardening above is still worth doing, but **this yaml line is the actual fix for "ComfyUI can't see the Anima models."**
- **Why LM "saw" it but guy90 didn't (Dusk, 2026-06-28):** LoRA Manager categorizes *everything* as "checkpoints" — it dumps the Anima diffusion model AND its text encoder into the checkpoints folder, so LM-based loaders find it. But Anima is a **diffusion-transformer, not an all-in-one checkpoint**; its correct loaders are `UNETLoader` → `diffusion_models/` (base) + a CLIP/TE loader → `text_encoders/` or `clip/` (text encoder). guy90 uses the *correct* `UNETLoader`. So the yaml fix likely also needs **`text_encoders: pretrained_model`** (and/or `clip: pretrained_model`) if guy90 loads Anima's TE from `pretrained_model/` separately — confirm from the guy90 workflow's loader nodes before finalizing the `installer.py:591` yaml template. Net: the yaml only maps the SDXL-style folders (`checkpoints`/`loras`/`vae`); the Anima/DiT folders (`diffusion_models`, `text_encoders`/`clip`) were never added.

## 6. Feature Priority Matrix (Beta)

### Must Have (Alpha -> Beta gate)
| Feature | Category | Effort | Status |
|---------|----------|--------|--------|
| Fix Anima checkpoint script mapping (CT-4) | Bug Fix | Tiny | ✅ Done (pre-existing) |
| Fix network_train_unet_only in checkpoint mode (LT-3) | Bug Fix | Tiny | ✅ Done 2026-04-30 |
| Wire up wandb_key environment variable (LT-1) | Bug Fix | Tiny | ✅ Done (confirmed 2026-05-25) |
| Add WandB/Logging UI section (UI-1) | New Feature | Small | ✅ Done — LoggingCard.tsx (confirmed 2026-05-25) |
| Dashboard redesign with all routes (UI-2) | UX | Medium | ⏳ Not started |
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
| Checkpoint-specific validation (CT-1) | Enhancement | Small | 🚫 **WON'T DO** (bias/nannying — rejected per LT-5/LT-7) |
| Hide LoRA fields in checkpoint mode (CT-2) | UX | Medium | 🚫 N/A — separate pages, no unified form |
| Merge progress reporting (MG-7) | UX | Medium | ⏳ Not started |
| SD3 merge support (MG-5) | Feature | Small | 🚫 Perma-deferred — SD3 deprecated per Stability AI; revisit only on explicit user request |
| Clean up redundant TOML generation (CT-6) | Tech Debt | Small | ⏳ Not started |
| Respect enable_bucket user setting (LT-2) | Bug Fix | Tiny | ✅ Done 2026-04-30 |
| LyCORIS algorithm-specific validation (LT-5) | Enhancement | Medium | 🚫 **WON'T DO** (bias — rejected per LT-5/LT-7) |
| Clarify "Full" LoRA type semantics (LT-4) | UX | Small | ✅ Done 2026-06-06 (label/description rewrite) |
| Silence AbortError console noise (UI-3) | Polish | Small | ✅ Done 2026-05-20 |
| Fix training log polling cadence + visibility (UI-4) | UX/Bug Fix | Small | ✅ Done 2026-04-30 |
| Add PYTHONUNBUFFERED to Kohya subprocess (UI-4 part) | Bug Fix | Tiny | ✅ Done (via -u flag in kohya.py:124) |
| MG-2: Document save_precision naming difference | Code Clarity | Tiny | ✅ Done 2026-04-30 |

### Nice to Have (Beta+)
| Feature | Category | Effort | Status |
|---------|----------|--------|--------|
| Per-image visual tag editor | New Feature | Large | ✅ Done — it's the inline badge-chip editor (shipped 2026-05-25, §1); minor tweaks may follow. NOT a separate unstarted feature. |
| Caption editor with search highlighting | Enhancement | Small | 🟡 Partial — **tag** editing is the chip editor (§1, done). Natural-language **captions** (BLIP/GIT) don't fit chips → current fallback is the text editor. **Fix: add a caption/textarea *mode toggle* to the existing `tags/page.tsx`** (same .txt file, chips ↔ textarea+search-highlight view) — NOT a new page. Per the "extend existing surfaces, don't spawn a page per feature" preference. Smaller than the old "Medium" implied. |
| Merge presets/templates | UX | Medium | ⏳ Not started — **sequenced after** the core SuperMerger work (MG-* block-weight merging) is sorted; presets sit on top of that. |
| Merge dry-run/preview mode | Feature | Medium | ⏳ Not started |
| EQ VAE support - SDXL (VAE-EQ-1) | Advanced Feature | Small | ⏳ Not started — deferred ("too hard in the moment, soon"). Niche/edge-case but **wanted** — important to *allow* it (bleeding-edge philosophy). Port of the SDXL EQ-VAE / reflection-padding work **by Anzhc & Bluvoll** (we obtained the code *via* Jelosus2's fork — integration source, NOT author); our vendored backend lacks it. **Credit Anzhc & Bluvoll in `ATTRIBUTIONS.md` when implemented.** |
| Qwen-Image VAE reflection padding - Anima (VAE-EQ-2) | Advanced Feature | Small (needs research) | ⏳ Not started — **different VAE entirely — NOT a port of VAE-EQ-1.** The Anzhc & Bluvoll EQ-VAE / reflection-padding work targets the **SDXL VAE**. Anima uses the **Qwen-Image VAE** — a separate autoencoder, not SDXL's — so this is the open question of whether equivalent EQ / reflection-padding treatment exists or applies for the Qwen-Image VAE *at all*, answered in the Qwen/Anima VAE code. (SDXL VAE ≠ Qwen VAE — don't conflate.) |
| Batch Downloader (BD-1) | New Feature | Medium | ⏳ Not started — **= the batchlinks tool (§6.5)**. Same item; see §6.5 for the full spec. |
| Training monitor tqdm ETA parsing (UI-5) | Enhancement | Small | ⏳ Not started |

---

## 6.5 Batch Downloader

**Issue BD-1: Model sourcing is fragile and region-dependent**
- **Priority:** Beta+ (nice to have, but solves a real accessibility problem)
- **Status:** Design LOCKED 2026-06-27 — implementation not started (see "v1 Design" below)
- **Motivation (1 — accessibility):** Civitai's API geoblocks certain regions (UK datacenter IPs get 451'd due to UK Online Safety Act compliance). Users shouldn't need a working Civitai API to download models — they should be able to paste any link and have it work.
- **Motivation (2 — escape LM's download UX, added 2026-06-22):** LoRA Manager's built-in download/manage flow is clunky and over-coupled to "how Civitai works" (Dusk: "the worst thing ComfyUI could've slapped in"). Batchlinks should **take over *downloading*** (paste any link, our routing) while **LM stays purely the *loader/browser*** (Dusk firmly keeps the `Lora Loader (LoraManager)` node — see §11). Coexistence hinge: after batchlinks writes a gen model into the ComfyUI folder, **trigger an LM rescan via its API** so the LM loader node sees it (same "force rescan" fix §802 found for HF-downloaded-checkpoint 400s). Net: delete LM's download flow from the user's life without losing LM's loading.

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

### v1 Design (LOCKED 2026-06-27 — supersedes the older FastAPI/SSE sketch in Implementation notes)

**Scope:** sources = public HF (`hf_hub_download`, tokenless) + Civitai (existing token) + direct HTTP via aria2c. B-tier (Google Drive / MEGA / magnet) deferred — the hashtag-routing design makes them cheap add-ons later. **No gated/private HF auth** (Dusk keeps his unreleased models on PUBLIC HF — can't get gated working and doesn't need to).

**Placement (IA):** standalone page under **File Management** nav. Rationale: it's a general file-getter (the *tool* is file-management even though its *content* is gen-flavored). Batchlinks ≠ the `/models` link-card — additive, doesn't touch it. **Separate sibling task:** relocate the *existing single-model downloader* under **Training** (those are training base models; nav/IA move, ships independently).

**Orchestration:** the **jobs system** (same machinery as tagging/captioning), NOT a bespoke FastAPI/SSE stream. A `batch_download` job runner loops items and **reuses the existing `ModelService.download_model_or_vae()`** (`services/model_service.py` — already routes HF→hf_hub_download+xet, Civitai/direct→aria2c+token, tracks bytes for progress) — **no new download code**. Per-item progress streams over the existing job WebSocket. **Per-item error isolation:** one bad link logs its error and the batch continues — never aborts the whole run.

**Flow:** textarea paste → [frontend] parse each line → `{url, type}` + live preview (counts per type, flag unparseable lines BEFORE submit) → start `batch_download` job → [runner] resolve `type → dir`, download via `ModelService`, emit progress lines → after all gen-model items, **one LM rescan** → per-item ✅/❌ summary.

**Hashtag → destination** (backend owns path resolution, PROJECT_ROOT-anchored via the just-fixed `get_comfyui_models_path()` / `a94854f` so it's correct on local/Vast/RunPod — frontend only sends the `type`):

| Tag | Destination | LM rescan? |
|-----|-------------|-----------|
| `#model` / `#checkpoint` | ComfyUI checkpoints | ✅ |
| `#lora` | ComfyUI loras | ✅ |
| `#vae` | ComfyUI vae | ✅ |
| `#dataset` | trainer dataset dir | ❌ |
| `#output` | trainer output dir | ❌ |
| *(untagged)* | global-default dropdown | per-type |

**LM coexistence (reframed 2026-06-27):** LM CAN already see HF-downloaded files — it just needs a rescan or a settings tick. So batchlinks **automates that rescan** after gen-model writes (one call at end of batch), saving the manual step. **This is automating a working manual flow, not working around a broken one.** Confirm LM's actual rescan endpoint against a RUNNING LM before wiring (ask-don't-assume). LM stays purely the loader for the SDXL + Guy90s anima workflows. (Updates the older "LM is bad at HF" framing.)

**Logging:** each download + the rescan call logged to the backend logs (also feeds the in-app log-viewer goal).

**Frontend:** `frontend/app/batch-download/page.tsx`, **shadcn only** (`Textarea`, `Select` for the global default, `Button`, progress list reusing the existing job-progress display). Parser is a pure fn → unit-testable.

**Out of scope (v1):** B-tier sources, gated/private HF auth, torrent indexing, replacing the `/models` card, cross-session resume.

**Testing:** parser unit tests (hashtag routing, untagged→default, junk lines); a dry-run that resolves routing without downloading; one small real download per source (HF / Civitai / direct).

### Implementation notes
- Backend: **jobs-system `batch_download` runner reusing `ModelService.download_model_or_vae`** (see v1 Design above) — NOT a standalone FastAPI/SSE endpoint (that earlier sketch superseded 2026-06-27)
- Frontend: new page at `frontend/app/batch-download/page.tsx`
- aria2c already present on instances — no new provisioning needed
- gdown may need `pip install gdown` added to requirements
- No torrent tracker/indexer integration — users provide their own links. Completely neutral technology.
- **LM coexistence (per Motivation 2):** for gen-model destinations (`#lora`/`#checkpoint`/`#vae` → ComfyUI folders), after the download completes, call LoRA Manager's rescan/refresh API so the new file is indexed and selectable in the `Lora Loader (LoraManager)` node without a manual ComfyUI restart. Confirm LM's actual rescan endpoint before wiring (ask-don't-assume — check while LM is running). Trainer-model destinations (`#dataset`/training output) don't touch LM.

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

**Priority:** ~~High — blocks actual training~~  
**Status:** ✅ **RESOLVED 2026-05-30** — presets fixed; loading a preset and hitting Train submits correctly.
- **Root cause:** Hardcoded presets in `useTrainingForm.ts` populated form visually but didn't wire to submit handler.
- **Fix applied:** Migrated to proper JSON preset system in `presets/`.

## 8. Anima Deep Dive — Research Session Needed

**Priority:** ~~Beta (before Anima is considered properly supported)~~  
**Status:** ✅ **RESOLVED 2026-05-30** — Anima works now (confirmed by Dusk). Historical context preserved in git history.
- **Issue was:** `size mismatch` on `layers.27.mlp.gate_proj.weight` — wrong LLM/text encoder loaded.
- **Resolution:** Correct file wiring + `networks.lora_anima` already in place.
- **ARCHITECTURE.md** still a good idea for future context — track separately.

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

When implementing Civitai-inspired features, add to `ATTRIBUTIONS.md`:

```markdown
## Civitai
**Repository:** [civitai/civitai](https://github.com/civitai/civitai)
**License:** Apache License 2.0
**Usage:** Tag viewer UI patterns, bulk tag operations, 3-way overwrite mode for auto-labeling
```

---

## 9. Session Notes

### 2026-05-25 — ComfyUI Generate UI polish + doc reconciliation

**Shipped:** ComfyUI Generate UI improvements (queue-runs, 400 error parsing, navbar LoRA Manager link), `extra_model_paths.yaml` installer support, ZIP upload chunking fix, Models page delete → AlertDialog.

**Findings:** LoRA Manager `Checkpoint Loader` validates against scanned cache (not CivitAI metadata). Fix = force rescan in LoRA Manager. Decision: build in-app model manager hooking LoRA Manager API (fold into Civitai downloader page), add ArcEnCiel source.

**Doc reconciliation:** §1 Tag/Caption 100% done; WandB/LoggingCard done; Merging tool half-done; BD-1 not started; Reflow violations likely low-RAM environmental.

**New ideas:** Post-training "Test in ComfyUI" button (deep-link, no copy); KNX-inspired save node for `software` PNG tag (fork LoraManager save logic).

### 2026-05-20 — Reflow Fixes + Log Stream Cutout + ComfyUI Planning

**Completed:**
- **PR #375** — Reflow fixes: TrainingMonitor rAF scroll, SelectContent positioning, raw buttons → shadcn, MAX_LOGS 500.
- **PR #376** — Log stream cutout: `deque` buffer-relative index bug → `total_lines_written` absolute counter, maxlen 2000, 13 regression tests.
- **§11.1** ComfyUI architecture finalised (submodule, B2 workflow templates, extension system).
- UI-3 (AbortError) and UI-4 (PYTHONUNBUFFERED via `-u`) confirmed done.

**Pending:** PR-0 (OptimizerSchema single source), LT-1+UI-1 (WandB) — both since resolved.

### 2026-04-30 — Beta Bug Bash

**Completed (dev branch):**
- Next.js 15 → 16 upgrade, PostCSS CVE patch, provisioning scripts
- CT-5: CheckpointTrainingConfig types include SD35/CHROMA/ANIMA
- MG-1: Removed deceptive alpha input (auto-calculated)
- MG-2: Documented --save_precision vs --saving_precision difference
- MG-3: Subprocess timeouts (resize=30min, merges=1hr)
- MG-4: CUDA availability check before --device cuda
- UI-4: pollLogs fixed-cadence + visibility-aware + visibilitychange
- HF-1: HF upload token from settings; owner/repoType persist localStorage
- LT-2: enable_bucket reads from config (not hardcoded)
- LT-3: network_train_unet_only guarded to False in checkpoint mode

**Since resolved:** LT-1+UI-1 (wandb_key + LoggingCard), PYTHONUNBUFFERED (via `-u` flag).
**Deferred:** MG-5 (SD3 merge), #349 (upload progress — easy, ~30-40 lines), Tag Viewer + Bulk ops.
**CT-2 closed N/A:** Separate pages already.

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

**🧭 North star (Dusk, 2026-06-22):** *Re-make what made A1111/Forge easy, on top of ComfyUI.* ComfyUI is the right **engine** (graph-as-JSON + API-first is what makes our programmatic template injection / BYO-template possible — Forge couldn't host that); its cost is the lost OOTB ergonomics. So the Generate-UI roadmap is **rebuilding Forge/A1111 comfort as a layer on top of ComfyUI**, not switching back. Every Generate-side feature (detailer dropdowns, batchlinks §6.5, future selection rework) gets one test: *does this make it feel more Forge-easy?* Incremental ergonomic bricks are encouraged (don't over-defer waiting for a grand "rethink"); the bigger rethink = workflow-dependent / BYO-template selection, tracked separately.

**Priority:** ~~Beta+ (after core beta bugs closed)~~ — **superseded: it's BUILT.** No longer a future/post-beta item.  
**Status:** ✅ **IMPLEMENTED & shipping (as of 2026-06-22).** Generate UI is live, bundled all-in-one: ANIMA + SDXL templates, model/VAE/LoRA dropdowns, LoRA Manager loader integration, 4-detailer Adetailer chains with per-detailer model pickers, UltimateSDUpscale. COMFY-1..4 shipped 2026-05-23; architecture finalised 2026-05-20. Now in **active Forge-ergonomic improvement** per the north star above — NOT "post-beta, not built." Remaining work is incremental (batchlinks §6.5, the selection rework) and tracked individually, not a release blocker.
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

#### ComfyUI backend: submodule (2026-05-20) — ⚠️ SUPERSEDED by COMFY-8 (direct clone everywhere, decided 2026-05-23)

> **Stale — kept for history only. ComfyUI is NOT a submodule;** it's direct-cloned by all install paths (see COMFY-8). The text below reflects the abandoned submodule plan.

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
- SDXL template — ✅ **shipped** (`sdxl-knx-v13pt5.json`: 4-detailer chains face/eye/hand/mouth, per-detailer model dropdowns, detailer denoise tuned 0.05→0.35 on 2026-06-22). The earlier "in progress" is done.

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

**COMFY-6 (evolving): Custom workflow templates**
- Users drop a workflow JSON into a `workflows/custom/` folder → it appears as a generation mode in the architecture dropdown
- Requires component coverage for the nodes used in the template (added over time as we encounter new node types)
- NOT a plugin system — no extension API, no dynamic node→component mapping
- Priority: low. Add template UI slots as we build workflows. No up-front abstraction effort.**

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
```text
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

**Priority:** Beta+ (NOT gated on COMFY-1 — different mechanism; see architecture below)  
**Status:** ⏳ Not started — integration plan drafted (architecture corrected 2026-05-30: in-app merge, NOT a proxied separate process)  
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

**Corrected approach (Dusk, 2026-05-30): direct in-app integration — NOT a proxied separate process.** ComfyUI uses a proxy because it's a Python app on its own port; Dataset Tools is the **same Next.js 16 + React 19 + shadcn stack as the Trainer**, so it slots straight into the single app as additional routes/components. **No second Next.js process, no `DATASET_TOOLS_PORT`, no `server.js` proxy block, no extra startup/provisioning steps.** One app, one build, one process.

**DT-1: Merge DT into the Trainer app (in-app)**
- Bring DT's pages in under a namespaced route in the Trainer's `frontend/app/` (e.g. `app/dataset-tools/...`) and its components into `frontend/components/`.
- **Namespace DT's API routes AND whitelist them in `server.js`** — the "API paths" gotcha, traced 2026-07-03. Both apps have `app/api/*`, with **direct collisions**: DT *and* the Trainer both define `/api/civitai` and `/api/settings` (both already in the Trainer's `server.js` `nodeApiPrefixes`). DT's other routes — `/api/fs`, `/api/metadata`, `/api/metadata-from-file`, `/api/metadata-write`, `/api/thumbnail`, `/api/image`, `/api/safetensors`, `/api/rules`, `/api/find-file`, `/api/comfyui-nodes`, `/api/health` — are unclaimed.
  - **Fix (two parts, both required):** (1) move all DT API routes under `app/api/dataset-tools/*`; (2) add `'/api/dataset-tools'` to `nodeApiPrefixes` at `server.js:240`. Without (2), `server.js` routes those paths to its "proxy other `/api/*` to FastAPI:8000" branch (`server.js:257`) → FastAPI has none of them → 404. This is a **one-line whitelist entry**, not a ComfyUI-style separate-process proxy block — the "no `server.js` proxy block" note above still holds; this single `nodeApiPrefixes` line is the one exception.
  - NB: the Trainer's own crop/convert Python endpoints (`/api/dataset/crop`, `/api/dataset/convert`) are **unaffected** — `next.config.js:52`'s `/api/:path*` → FastAPI rewrite proxies them through fine (verified 2026-07-03).
- **Reconcile shared pieces:** dedupe `components/ui/*` (both shadcn — keep one set, watch for version drift), merge `next.config.js` (`serverExternalPackages` for `sharp`, etc.) and any provider/layout wrappers.
- **Settings + thumbnail cache:** DT's settings system and `.thumbcache/` are self-contained — keep them namespaced so they don't clash with Trainer settings.
- Net ops change: provisioning just builds the one app; no separate clone/build/start of a second repo, no port wiring. Code can be vendored/subtree'd from the Dataset-Tools repo or copied in — decide at implementation time.

**Framing (Dusk, 2026-05-30): this is a full combination, bidirectional — not a one-way bolt-on.** The two apps are the same stack, so treat it as unifying them and **bringing DT's optimizations forward into the combined app** (and vice versa). Concretely harvest:
- **UI components** — DT's glassmorphism/glow/cursor/card/tabs set + theme system (already flagged as the §14 unjankification goldmine). Porting these *is* a big chunk of §14.
- **Performance work** — any perf optimizations DT already made (thumbnail caching/`sharp` pipeline, metadata parsing, list virtualization, etc.) apply directly to the Trainer's equivalent pain points (e.g. the dataset gallery / file manager reflow issues).
- **Reusable subsystems** — DT's ComfyUI integration (`comfyui-node-registry.ts`, `comfyui-github-search.ts`, `ComfyUIWorkflowViewer.tsx`) may feed the Trainer's ComfyUI tab directly (cross-ref §11.1).
- Decide which is canonical when both apps have a version of the same thing (shadcn components, settings patterns) and keep the better one. The merge is the opportunity to consolidate, not duplicate.

**DT-2: Navbar link**
- Add "Dataset Tools" entry to the trainer navbar under a new "Ecosystem" section (or alongside "Files")
- Links to `/dataset-tools`

**DT-3: Handoff buttons (the good stuff)**
- **"Inspect in Dataset Tools"** on the files page — deep-link to Dataset Tools with the current folder pre-set
- **"Inspect LoRA"** on training completion — opens the trained `.safetensors` directly in Dataset Tools' safetensors panel
- **"View reference metadata"** in the dataset image gallery — opens a selected image in Dataset Tools' metadata panel

With the in-app merge these become **same-app navigation** (Next router/`<Link>` to the namespaced `app/dataset-tools/...` routes) — simpler than the old cross-app deep-link idea. Can still pass the target via URL params (e.g. `?path=`/`?file=`) and even share state directly if wanted, since it's one app now.

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
- `.thumbcache/` directory generates WebP thumbnails via `sharp` — with the in-app merge it lives within the single Trainer app; just pick a stable location and ensure `sharp` is in the merged `serverExternalPackages`
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

### 11.5 SDXL Workflow — Eye Detailer Pass (COMFY-14)

**Status:** 🔧 **Workflow BUILT, UI wiring pending (updated 2026-06-09).** Dusk has already built + tested a **4-pass** Adetailer in `sdxl-knx-v13pt5.json` — Face (`bbox/face_yolov8n`), Eyes (`segm/Anzhc Eyes -seg-hd`), Hands (`segm/PitHandDetailer-v1b-seg`), Mouth (`bbox/adetailer2dMouth_v10`). Remaining work is **exposing the full control surface** in the Generate UI — NOT building or hiding a single auto-fix pass. (The single-eye-pass framing below is superseded; kept only for the bbox-vs-segm wiring reference, which is still accurate.)

**Problem:** Eyes are frequently poorly drawn even with the current detailer. **Root cause is NOT GPU or model** — the workflow only has a *face* detailer (`face_yolov8m`). A face detailer crops the whole face and re-renders it as one region, so the eyes get only a tiny fraction of the new resolution. The fix is a dedicated **eye detailer pass** (standard intermediate SDXL setup; Forge's ADetailer did this by just adding a second tab with an eye model).

**Models:** Dusk already has Anzhc's anime detection models; some are **`-seg-` (segmentation)** models. (Alt refs if needed: "Eyeful" / SnowyYukino eye models on Civitai.) Models live in `ComfyUI/models/ultralytics/{bbox,segm}/` — ties to COMFY-12 (these aren't auto-downloaded).

**Key gotcha — bbox vs segm wire differently (this is the part that wasn't obvious):**
- **bbox model** → rectangle → needs SAM → routes through `Simple Detector (SEGS)` (what the existing *face* chain uses).
- **segm `-seg-` model** → outputs the mask shape directly, **no SAM** → routes through **`SEGM Detector (SEGS)`** (`SegmDetectorSEGS`) using the detector's **SEGM_DETECTOR** output. *(The harmless little red X on the face detector's segm port is just that output being unused on a bbox model — normal, not a bug.)*

**Wiring for the (segm) eye pass:**
```
UltralyticsDetectorProvider (Anzhc eye -seg-) ── SEGM_DETECTOR ──▶ SEGM Detector (SEGS) ──▶ DetailerForEach (eyes)
```
- Splice **after** the face DetailerForEach (#74), **before** Save (#65): face first, then eyes on the cleaned face. (Find what currently feeds Save's image input; insert the eye chain between.)
- DetailerForEach eye inputs: same MODEL/CLIP/VAE/positive/negative as the face pass.
- **Settings:** `guide_size` 256–512 (biggest lever — gives tiny eyes real resolution); `denoise` ~0.4 (redraw without changing eye color/shape/style); SEGM threshold ~0.5 (lower if it misses an eye).

**Design intent (corrected 2026-06-09 — kill the simplify-and-hide bias):** the extra detail passes ARE the feature for this audience, not surface to hide. Phase 1 (wire + tune live) is **done** — v13.5 ships 4 tuned passes. Phase 2 is **expose the full control surface** in the Generate UI:
- **Per-pass on/off** for each detail pass (independent bypass — today's single `adetailerEnabled` only covers ONE of four; that's the correctness bug to fix first).
- **Per-pass detector model** picker, **bbox/segm-aware** so a segm model routes to `SEGM Detector (SEGS)` and a bbox model to `Simple Detector (SEGS)` (the wiring reference above stays authoritative).
- **Per-pass settings** users want to reach (guide_size / denoise / detection threshold) — exposed **inline**, NOT behind an "Advanced drawer." A drawer is *more* machinery (build the show/hide abstraction, decide what's "advanced," manage its state) AND it buries the feature. Inline is both less work and the honest default for this audience.
- **Add/remove passes** beyond the built-in set (see build note — the one genuinely open scope question).

**Do NOT** re-frame this as "auto-fix that hides the wiring" or "one body part at a time, future optional." Dusk already built all four; the job is *surfacing* control, not rationing it. (Recurring bias to resist — see CLAUDE.md "Bleeding Edge" + "Empirical Lore." Past sessions kept writing "nobody needs the extra surface" into this doc and leaving Dusk to force-agree. Stop.)

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
| COMFY-13: Gallery image popup — show + copy generation metadata (prompt/seed/sampler/settings) from the lightbox in `GenerateUI.tsx` | Feature | Small | ⏳ Not started |
| COMFY-14: Full Adetailer control surface — 4 passes (face/eyes/hands/mouth) built+tested in `sdxl-knx-v13pt5.json`; expose per-pass on/off + bbox/segm model picker + settings in Generate UI (see §11.5) | Feature | Medium | 🔧 Workflow done, UI pending |
| DT-1: Merge DT routes/components into the app (namespace API routes, dedupe `components/ui`, merge `next.config`) | Infrastructure | Medium | ⏳ Not started |
| DT-5: KNX Ecosystem source tag detection (pairs with COMFY-9) | Integration | Tiny | ⏳ Not started |
| DT-2: Navbar link to Dataset Tools | Integration | Tiny | ⏳ Not started |
| DT-3: Handoff buttons (inspect LoRA, view reference, files) | Feature | Small | ⏳ Not started |
| DT-4: Deep-link folder awareness | Enhancement | Small | ⏳ Not started |

---

## Section 12 — Security Review Backlog

### 12.1 CWE-23 Path Traversal (Snyk removed, low urgency)

**Status:** ⏳ Not started | **Priority:** Low (pre-public-release gate)

Snyk removed (SaaS-calibrated noise). Path traversal sanitisation is good practice but not imperative — single-user local tool, private tunnel. `files.py` already protected; `config.py` preset paths need `is_relative_to(presets_dir)` guard; service paths need configurable `WORKSPACE_ROOT` validation.

### 12.2 CWE-78 Command Injection (Snyk — mostly false positives)

**Status:** ⏳ Not started | **Priority:** Low

All subprocess calls use `asyncio.create_subprocess_exec` / list-form `subprocess.run` — no shell, not injectable. Snyk flags conservatively. Audit: no `" ".join(args)` → shell; verify `model_service.py` headers never shell-interpreted.

---

## Section 13 — In-App Documentation Cleanup

### 13.1 Integrate upstream docs + remove hand-holding bias

**Priority:** Low (post-in-house-testing) | **Status:** ⏳ Not started | **Effort:** Medium

**Pull in:** LyCORIS docs (params/network types), sd-scripts docs (flags/optimizers/schedulers). Strip outdated/contradicted content.

**Remove bias:** Hardware requirement nags on pages user already chose; Claude over-explanation/hedging. Changelog page: populate from git history or delete route.

**Blocked by:** Stable alpha + in-house testing complete.

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

**G. File Manager**
- **More contrast** — the file list is low-contrast and hard to scan. Needs clearer row separation, stronger hover/selected states, and a visible dir-vs-file distinction (structural contrast, per the page-by-page approach here).
- **Copy files (feature, do alongside the contrast pass)** — the files API has list/rename/delete/mkdir/read/write/workspace but **no copy**; `rename` only moves. Add a copy capability: a new `/api/files/copy` route reusing the existing files-route path-safety guards (`is_safe_path`/`ALLOWED_DIRS`), plus a UI action. Pairs naturally with the contrast work since both touch FileManager.

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
| **Theme-matched button glow:** faint `box-shadow` behind buttons driven by theme CSS vars (`--primary` etc.) on the existing shadcn `Button` — ship alongside expanded color themes. Keep subtle, no neon/flashing. No bare HTML / primitives needed (pure CSS via className). | Tiny | Low | ⏳ Not started |
| Dashboard redesign (UI-2) | Medium | High | ⏳ Not started |
| Training form: progressive disclosure | Medium | High | ⏳ Not started |
| Training form: section nav / orientation | Small | Medium | ⏳ Not started |
| Empty/loading/error states (all pages) | Medium | High | ⏳ Not started |
| Toast copy pass | Tiny | Medium | ⏳ Not started |
| Nav: active states + ecosystem group | Small | Medium | ⏳ Not started |
| Semantic color pass (icons, badges) | Small | Medium | ⏳ Not started |
| **Slate purge:** Training cards (`SavingCard`, `MemoryCard`, `LoRAStructureCard`, `LoggingCard`, `CaptionCard`, `AdvancedCard`, `AugmentationCard`) — `border-slate-700` dividers → `border-border` | Tiny | Medium | ✅ Done 2026-06-16 (all 7 cards: `border-slate-700`→`border-border`, `bg-slate-800/50`→`bg-muted/50`) |
| **Slate purge:** `DatasetUploader.tsx` + `UppyDatasetUploader.tsx` — raw HTML form elements + `bg-slate-800`/`border-slate-600` → shadcn components + CSS vars | Medium | High | ✅ Done 2026-06-16 (4 raw `<input>`→shadcn `<Input>`, `<label>`→`<Label>`, slate divs→`bg-muted`/`border-border`; validation borders kept as semantic yellow/red; tsc clean) |
| **Custom → shadcn audit:** `components/effects/` custom cards/borders/buttons — identify which can be replaced by installed shadcn components, retire the rest. Root cause: went custom early before knowing what shadcn had. | Small | Medium | ⏳ Not started |
| **Hero slate:** `hero-animated.tsx` gradient strings bake in `slate-950` — make theme-aware or replace with CSS var equivalents | Tiny | Low | ⏳ Held for visual call (2026-06-16) — the `slate-950` gradients are ALREADY behind `isDark ? dark : light` ternaries (not a theme bug; deliberate dark base). Swapping to `bg-background`/`bg-muted` would change the designed dark hero. Also two raw `<button>` CTAs here (lines ~150/163) — pairs with the a11y raw-button→shadcn pass. Needs Dusk's eye, not a mechanical swap. |
| **a11y: numeric inputs for Steps & CFG** — on the Generate page (`GenerateUI.tsx`) Steps and CFG are slider-only; add paired numeric `<Input>` (like width/height/seed already have) so values are typeable, precise, and exposed to keyboard/assistive-tech users. | Tiny | Medium | ⏳ Not started |
| **a11y sweep (whole app)** — dedicated pass at some stage (not urgent): keyboard nav, `focus-visible` states, ARIA labels, slider-only controls, color contrast, form-label associations. Audit + fix per page. | Medium | High | ⏳ Not started |

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

## Section 18 — Frontend Tooling & Follow-up Backlog

### 18.1 Priority Matrix

| Item | Effort | Impact | Status |
|------|--------|--------|--------|
| **Fix Next-16 lint** — ✅ **Done 2026-06-16.** Two fixes: (1) `lint` script `next lint` → `eslint .` (Next 16 removed the subcommand); (2) the `minimatch: expand is not a function` crash was the `brace-expansion` override `>=2.0.3` floating to **v5.0.5**, which moved to ESM named exports and broke `minimatch@3`'s CJS `require()` (eslint's own `@eslint/config-array` + import/jsx-a11y/react plugins all use `minimatch@3`). Capped the override to `>=2.0.3 <3.0.0` (highest CJS, CVE-patched v2 = 2.1.1). Flat config already present (`eslint.config.js` spreads `eslint-config-next@16`'s flat array); removed the dead legacy `.eslintrc.json`. `npm run lint` now runs across the whole project; `tsc --noEmit` still clean. **Follow-up (separate cleanup, NOT the tooling fix):** lint surfaces 71 pre-existing problems (47 errors / 24 warnings) — 21 `react/no-unescaped-entities` (cosmetic), 26 React-Compiler hook rules (`react-hooks/refs`, `preserve-manual-memoization`, `set-state-in-effect`) from eslint-config-next@16's strict defaults, rest warnings. Triage/disable-vs-fix is a content decision, tracked here not done. | Small | High | ✅ Done (tooling) |
| **Frontend unit tests** — no React/Next test runner exists (`tests/` is Python only). Recommend Vitest + React Testing Library (Next 16 / React 19 fit). Would let us unit-test things like the gallery keyboard accessibility instead of relying on manual visual checks. Ties into the existing "smoke tests" intent. | Medium | Medium | ⏳ Not started |
| **LyCORIS re-sync** — vendored LyCORIS is at 67372a `dev16` (synced 2026-05-05); upstream `dev` adds 7 files not yet vendored (assessed 2026-06-22) that split into TWO wiring patterns: **(a) new modules** — `lora2.py` (`LoRA2Module(LoConModule)`), `ortholora.py`, `tsm.py` (`LycorisBaseModule`) → register in `modules/__init__.py` + algo dict, then LoRAType enum + `_map_lora_type_to_network` (`kohya_toml.py:338`) + frontend dropdown, same `algo=X` pattern as ABBA/TLoRA; **(b) init / gradient-hook methods** — `pissa_utils.py` (SVD init), `ralora_utils.py` + `gora_utils.py` (gradient hooks) → NOT new types, exposed as network_args options (need exact LyCORIS arg names + which base algos accept them). Do as a wholesale re-sync per the methodology (preserve our patches). NOT "Small" — the init-method half is fiddly. Nothing breaks without them; TLoRA/ABBA already work (modules vendored). See §11.1. | Medium | Low | ⏳ Not started |
| **`train_llm_adapter` wiring (Anima)** — the arg exists in the vendored backend (sd_scripts `lora_anima.py` + LyCORIS) and is documented, but is NOT exposed in the config flow (`api.ts` / `validation.ts` / `config-service.ts` / `kohya_toml.py` / presets / UI). Currently defaults `False` with no way to enable. Mind the Anima (`networks.lora_anima`) vs LyCORIS network_args path difference. | Small | Low | ⏳ Not started |
| **Preset audit + rename** — review presets for what's actually useful; rename misleading names (e.g. the Illustrious preset labelled "Conservative" that's actually a fast/clean config). Distinct from the format-migration audit (§5.0.96) and optimizer-args contamination (§5.1) — this is a content/naming pass. | Small | Medium | ⏳ Not started |
| **`.jsx` → `.tsx` conversion** (deferred from CR on #386) — convert remaining plain-JSX components to TypeScript per the frontend TS-only policy. `ClickSpark.jsx` is in active use (GenerateUI) so it must be *converted*, not deleted; type the canvas refs, `Spark[]`, and pointer/mouse handlers. (`BorderGlow.jsx` is exempt — slated for deletion via the §14.6 "Custom → shadcn audit".) | Small | Low | ⏳ Not started |
| **Demo/showcase file audit** (deferred from CR on #386) — `*-demo.tsx` and `satori-ui/dotted-modern.tsx` look unused. Either delete them, or fix the nits CR flagged (dotted-modern CTA label/href mismatch, `gooey-input-demo` missing docstring). Decide keep-vs-delete first. | Small | Low | ⏳ Not started |
| **`upload-progress.tsx` dark-mode decision** (deferred from CR on #386) — the redundant `isDark` ternaries were collapsed to a light-only palette (lint fix). Decide whether this aicanvas upload card should actually support dark mode; if yes, supply real dark hex values for the color tokens. Pairs with the §14.6 theme-system work. | Tiny | Low | ⏳ Not started |

### 18.2 Notes

- The hand-built `components/effects/*` (and `components/BorderGlow.jsx`) are confirmed duplicates of installed shadcn/registry components (`shiny-button`, `shine-border`, `hover-border-gradient`, `rainbow-button`, `backlight`, `spotlightcard`, …). Retiring them is already tracked as the "Custom → shadcn audit" row in §14.6.
- Root cause of the duplication (per §14.6): components were built custom early, before knowing what shadcn/the installed registries already provided. Rule going forward: use the installed component; only hand-build when nothing installed/installable fits.

### 18.3 Handoff & delegation discipline *(2026-06-22)*

Work hands off cleanly to **anyone** — DeepSeek (≈ peer-tier LLM, run via **opencode** in the same terminal, no PowerShell friction), a future human coder, or a cold-started Claude — **when the context is externalized**, not held in one session's head. Capability was never the gate (framing DeepSeek as "limited to safe scraps" is bias, not analysis). The two tools that externalize context are ones we already use:
- **Docstrings as we go** (never a deferred pass) — the local "what/why" context.
- **BETA_PLANNING context** — the cross-system "how it fits / why it's sequenced this way" context.

**Honest current gap:** it's NOT all externalized yet ("hand this to someone tomorrow and they'd get lost"). So *keeping docstring + planning coverage current IS the delegation enabler* — and it serves DeepSeek, future humans, and cold-Claude identically. You never hand anything off cold to any coder; you hand off the context with it.

**The one real gate — about *verifiability*, not who's coding:** silent breakage that needs a GPU / ground-truth to confirm — training correctness, optimizer dispatch (schedule-free train/eval), ComfyUI runtime behavior. DeepSeek can absolutely *write* these; they just need **Dusk-verified before merge** (a verification step, not a capability wall). Hand ComfyUI work its ground truth (`tools/dump-workflow.js`) up front so it's not guessing the graph.

**Pre-PR review = mutual peer review** (DeepSeek ↔ Claude) before Dusk's PR mode + CodeRabbit. Two independent LLM passes over a diff catch more than one — neither is the "junior."

---

## Section 19 — Repository Root Organization

### 19.1 Problem
~48 tracked files sit at the repo root — mostly install/setup/run/diagnostic scripts plus **six** `requirements*.txt`. Hard to scan; new contributors can't tell which entry point is "the" one.

### 19.2 Proposed grouping (a *plan*, not a blind `git mv`)
- `scripts/install/` — `install.bat`, `install.sh`, `install_frontend.py`, `install_linux.py`, `installer.py`, `installer_windows_local.py`
- `scripts/provision/` — `provision_runpod.sh`, `provision_runpod_dev.sh`, `vastai_setup.sh`, `vastai_setup_dev.sh`
- `scripts/run/` — `start.sh`, `start_services_*.{bat,sh}`, `restart.{bat,sh}`, `fetch-restart.sh`, `run_backend.py`
- `scripts/diagnose/` — `diagnose.{bat,py,sh}`, `clean_slate.py`
- `requirements/` — the six `requirements*.txt`
- **Must stay at root** (auto-discovered by tooling/GitHub): `.gitignore`, `.gitattributes`, `.gitmodules`, `LICENSE`, `pyproject.toml`, `README.md`, and the agent files `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` / `QWEN.md`. GitHub special files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`) may move to `.github/` if wanted.

### 19.3 Critical caveat — NOT a free move
Every relocation breaks references that must be updated **and re-tested on all three platforms (local / VastAI / RunPod)**:
- README install steps (clone → run `install.sh`)
- Cross-script path assumptions (scripts calling siblings / assuming root as cwd)
- VastAI/RunPod provisioning expecting fixed paths
- `pyproject.toml` entry points / `run_backend.py`
- `CLAUDE.md` / `AGENTS.md` references (e.g. `start_services_local.bat`)
- Installer `project_root` / `install_complete.marker` path math

Do it **incrementally** (one group per PR), re-test each platform, update docs in the same PR.

### 19.4 Quick-win cleanup (independent of the reorg)
- `.snyk` — ✅ **Deleted 2026-06-16** (Snyk removed; policy file was dead).
- `find_fences.py` — ✅ **Deleted 2026-06-16** (one-off code-fence scanner, zero references).

---

## Section 20 — Backend Delivery & Vendoring Strategy *(decision captured 2026-06-06)*

**Status (REVISED 2026-06-22): vendored + patch-update — CHOSEN. Submodule (former Option A) REVERSED to parked ("too-hard basket, temporarily" — revisit only if update cadence ever justifies the cloud-wiring cost).**

**Decision (2026-06-22): keep the backend VENDORED; switch the update *method* from wholesale flush → targeted patch (cherry-pick the specific upstream files a given feature/fix needs).** Two assumptions the 2026-06-09 submodule lock rested on were falsified this session: (1) "pure consumer / no patch-set" is **FALSE** — we carry live fixes (SDXL-NaN `d39d249`, 5 optimizer crashes `a3b7361`, min_snr/cross-attn `65cb90d`), verified still absent from upstream `sd3-upstream@457914d`, so a fork was needed *either way*; (2) the submodule's real cost — every clone site (`vastai_setup.sh`, `provision_runpod.sh`, both `_dev` variants, installers, docs) must pass `--recurse-submodules` or the backend comes up **EMPTY on cloud** — lands on the most fragile, least-debuggable path. Vendoring keeps the backend as plain in-repo files (cannot come up empty), needs no fork hygiene, keeps `AGENTS.md` "no submodules" honest, and the cherry-pick labor is the assistant's, not Dusk's. Patch-update fits the actual cadence (event-driven — a new model/optimizer/fix — not weekly). **Weakness accepted:** big multi-week catch-up jumps are painful via cherry-pick → reserve a deliberate flush-and-re-apply-patches for those rare cases. (Original Option A reasoning preserved in 20.2/20.3 below as the record of *why* it was once leading.)

- **Correction to an earlier assumption:** ComfyUI does **not** currently prove a submodule-on-cloud pattern here — `.gitmodules` is **empty** and ComfyUI is **direct-cloned** (see COMFY-8 / §11 "ComfyUI backend is directly cloned"). So Option A *pioneers* the submodule-on-cloud pattern in this repo — which is why the 20.4 wiring smoke-check on a real box still matters (the pattern is new *here*), even though the decision itself is settled.
- **Two submodule targets:** (a) the **training backend** → our clean tracking fork of 67372a; (b) **ComfyUI** (currently direct-cloned — intermittent clone failures motivate pinning + one uniform `--recursive` pull for both).
- **Hinge FALSIFIED (2026-06-22):** the "pure consumer / no patch-set" claim was wrong — we DO carry a live patch-set (see 2026-06-22 decision above). A clean tracking fork was never actually on the table, which is a primary reason the submodule lost its edge. Plan: **carry our patches in the vendored tree indefinitely — NO upstream PR** (decided 2026-06-22: 67372a is solo-maintained and a PR would likely sit unmerged forever, so it can't be relied on to shrink the patch-set). On each patch-update, preserve the patch-set (see `upstream_sync_methodology.md`); the set is small (SDXL-NaN + 5 optimizer one-liners, mostly cold files) so the carry cost is low.
- **Robustness is a separate lever (don't conflate):** submodules fix update-cleanliness + pinning, **NOT** clone fragility — a submodule update is still a git fetch. The clone-fragility fix is shallow `--depth 1` + retry, **landed 2026-06-09** in `installer.py` / `installer_windows_local.py` for the direct ComfyUI clone; carry the same `shallow = true` into the submodule. True elimination of clone fragility still needs Docker (Option B, parked).

**Vendored-backend code patches are UN-PAUSED (2026-06-22)** — the submodule swap they were being held for is parked, so patching the vendored tree is again the normal, sanctioned workflow (it IS the chosen delivery method now). Re-apply discipline: before any wholesale flush, preserve our patch-set first (see `upstream_sync_methodology.md`).

### 20.1 The reframe — backend currency is the steady state, not an edge case
Kill the "updating the ML backend is a rare edge case" bias. 67372a ships ~weekly (1,226 commits; our vendored snapshot is ~a month stale as of 2026-06-06). ML moves constantly, so the design criterion is **"updating to latest is trivial, frequent, and low-risk"** — NOT "minimize how often we touch it." Generalizes the project anti-bias rule from training params to process/architecture (cross-ref CLAUDE.md "Bleeding Edge" + "Empirical Lore — Interrogate It").

### 20.2 What's wrong with each delivery mechanism
- **Vendoring (current):** hand-copy slog, no version marker, perpetual drift — "weird/messy." This is the pain we're escaping.
- **Plain `git clone`:** stupidly fragile on cloud (network blips, the documented install cascade).
- **pip-install-from-git — REJECTED:** it's a git fetch at deploy time, so it's the **same fragility class as clone** (Dusk's call, correct). It would only clean up *updates*, not *robustness* — not worth pretending otherwise.

### 20.3 Options
**(A) Submodule → OUR fork of 67372a — *leaning, for now*.**
- Update = bump a pinned SHA (purpose-built for constant change).
- **Licensing alignment (GPL-3.0):** 67372a is GPL-3.0. A submodule *references* their code (clean attribution, no relicensing/propagation question); vendoring *copies* GPL source into our tree — murkier. Submodule is the cleaner GPL posture. (Third reason it wins, alongside update-cleanliness and matching upstream's own pattern.)
- **Machinery is partly present, but the pattern is UNPROVEN here:** `.gitmodules` exists (though **empty**), and `--recursive` covers nesting (67372a itself submodules `sd_scripts @ 457914d`). **Correction (2026-06-09):** ComfyUI is **direct-cloned, not a submodule** — the earlier "ComfyUI backend: submodule (2026-05-20)" note was *superseded* by COMFY-8. So Option A **pioneers** the submodule-on-cloud pattern in this repo rather than widening a proven one — which is why the 20.4 wiring smoke-check on a real box is worth doing (the pattern is new *here*): a wiring confirmation, not a reason to doubt the decision.
- **Honest boundary:** fixes *update cleanliness*, NOT *clone robustness* — still git-clone-based. Clone fragility stays parked until (B).
- **Precedent ≠ proof:** the ComfyUI submodule path is itself still pending live cloud verification.
- **HINGE QUESTION — RESOLVED (2026-06-09):** we were never carrying a private patch-set; the pain was *chasing upstream by hand*, not maintaining divergent patches. So we're effectively a **pure consumer** → the submodule points at our **clean tracking fork** of 67372a (a stable pin we control + the base to PR fixes back upstream), staying close to upstream and low-maintenance. Not a fork to hold patches.

**(B) Pre-baked Docker image — *long-term, deliberately parked*.**
- The only option that truly fixes clone fragility (backend pre-installed → no runtime clone/install cascade).
- Parked by Dusk's explicit call (2026-06-06): **not because it's wrong** — it's the gold standard — but a hands-on **knowledge gap** (Docker + service offloading + security layers he hasn't worked with directly). Revisit when he's ready. **Do not push it.**

**(C) Status-quo vendoring — *fallback*.** Just clones, freely patchable, but the staleness/drift is exactly the pain. Keep only if (A) proves too fragile in the de-risk test.

### 20.4 Wiring smoke-check before flipping main (NOT a referendum on the decision)
The decision is made — submodule beats vendoring, full stop (it ends the hand-chasing; pinning + shallow make it *less* fragile, not more). This is **not a test of whether the approach works** — it's a one-box confirmation that the **provisioning scripts wire `--recurse-submodules` + `shallow = true` correctly** for both targets (backend fork + ComfyUI) on a real cloud host. We're catching a wiring typo, not re-litigating the approach. The old submodule burn was cloud-execution competence, not the mechanism. Verify GUI-first: read the install log via file-browser/Jupyter (`/workspace/Ecosystem/logs/`), open the app, run a tiny job — no SSH needed for the happy path.

### 20.5 Why staying current matters — what 67372a shipped since our ~2026-05-05 snapshot
New optimizers (SODA/MODA/AMUSE, AdamWScheduleFreePlus, nor_muon_schedulefree, OCGOptV2, fftdescent); adaptive non-uniform timestep sampling (arXiv:2411.09998); Weight Noising; Latent Wavelet Diffusion masking; `min_snr_gamma_soft` + Min-SNR-gamma for flow-matching models; ICC-aware color (`to_srgb()` replacing `.convert('RGB')`); LyCORIS T-LoRA via LoCon/ortholora; Anima leco + addift; flash-attn guard for < CUDA sm_80; REX scheduler fixes. We're missing all of it.

### 20.6 Related — torchao is now installed upstream (cross-ref §4.4)
67372a migrated installs pip→**uv** and added torchao via `--index-strategy unsafe-best-match` (2026-05-30) — upstream resolves torchao against the *installed* torch instead of a hard pin, which solves the version-coupling risk that argued against adding it standalone. **RESOLVED 2026-06-22:** with the submodule parked and patch-update chosen, torchao was added standalone as `torchao==0.7.0` in `requirements_base.txt` (see §4.4 Status) — cross-platform + torch-2.4.1-safe via the pure-Python `torchao.utils` path. The old "decide as part of the sync" framing is moot.

### 20.7 uv for provisioning installs *(implemented 2026-06-23, remote only)*
**Goal:** use **uv** for **remote/cloud** dependency installs (provisioning), keep **pip on local** — never uv on local.

- **Why remote:** uv installs are faster + parallel + cached → less GPU-rental time burned on the provisioning cascade (literal $ saved while the rented GPU idles during setup). Also matches 67372a's own pip→uv migration, and uv's resolver (`--index-strategy unsafe-best-match`) handles torch-coupled deps cleanly (the exact trick that resolves torchao against the *installed* torch).
- **Why local = pip (conservative default, NOT a proven uv limitation):** local stays on the known-good pip path because uv-on-local is **unverified**, not because it's broken. There's an **unrooted anecdote** from the old Jupyter-notebook era of *something* trying to install a **Rust toolchain** on Windows during deps — never pinned down which package (numpy? safetensors? uv itself? other). So we don't flip local (Windows) installs onto uv on a hunch. (Note: local Windows users are end users running the trainer, not developers — don't conflate "local" with "dev.") If anyone ever wants uv locally, it's worth actually testing rather than assuming it breaks. `detect_package_manager()` gates uv on `platform.system() == "Linux"`; everything else stays pip.
- **Zero conflict with current state:** uv reads our existing `requirements_*.txt` (`uv pip install -r ...`), so the `torchao==0.7.0` pin and all other pins work unchanged. Nothing to re-spec.
- **How it shipped (defensive, no container test):** implemented in `installer.py` `detect_package_manager()` (the abstraction was already scaffolded "uv → pip fallback" but returned pip). Bootstraps uv (`pip install -U uv`), resolves a runnable invocation (`python -m uv`, then PATH `uv`), routes every install site through `uv pip install --python <interp> --index-strategy unsafe-best-match`. Two safety nets so it can never make provisioning worse: (1) any uv bootstrap failure silently returns pip; (2) if a uv install still fails, `install_dependencies()` retries the same requirements with pip. `requirements_cloud.txt` carries no torch (base image pre-installs it), so the cu121-extra-index quirk doesn't bite here.
- **Live-confirmed 2026-06-23:** uv engaged on a real dev deploy and made provisioning **noticeably faster** — `--python <interp>` worked on the base image, no pip fallback needed. (The original aria2 hang that prompted this was likely *partly* a host/container condition — boot-time dpkg lock / slow host — which the aria2 fix now rides out regardless.)
- **Status:** ✅ **Implemented + live-verified (remote only).** Speedup confirmed. Still nice-to-have: a clean wall-clock pip-vs-uv comparison number, and confirm editable (`-e`) installs under uv on a deploy that exercises them.

---

## Section 20 — AMD GPU Support (ZLUDA) — Research *(captured 2026-05-30)*

### 20.1 Interest
App currently assumes NVIDIA/CUDA everywhere (local + VastAI + RunPod). Dusk wants to scope eventual **AMD GPU support**. Reference: `patientx/ComfyUI-Zluda` — a Windows-only ComfyUI distribution that uses **ZLUDA** to run CUDA workloads on AMD GPUs, with install paths for RX 400/500 through RDNA4 9000-series.

### 20.2 Scope notes (research-only, not committed)
- **Split reality (per Dusk 2026-05-30):** the two halves of the app have very different AMD outlooks:
  - **Training (sd-scripts):** *should* work on ZLUDA hardware — this is the realistic AMD path. (Unconfirmed end-to-end, but the expected-viable side.)
  - **ComfyUI / generation side:** currently effectively **NVIDIA-only** — this is the blocker, not training.
- ZLUDA is Windows-only and a CUDA-translation layer — implications for the Linux VastAI/RunPod targets are different (ROCm is the Linux AMD path). Needs separate investigation per platform.
- Unknowns to research before any commitment (training side first, since it's the viable one): does the vendored sd-scripts stack actually run under ZLUDA end-to-end? bitsandbytes / xformers / custom optimizers (CAME, etc.) AMD compatibility? onnxruntime tagging path on AMD?
- Not a beta blocker. Parked as a forward-looking platform expansion.

---

## Section 21 — Dataset-Tools Integration & Theme Picker *(captured 2026-07-03)*

### 21.1 Overview
Port high-value features from the standalone Dataset-Tools app (`C:\Users\dusk\Development\Dataset-Tools`) into Ecosystem. Dataset-Tools is a Next.js 16 app with OKLCH theming, AI image metadata parsing, and SafeTensors inspection — same tech stack as KNX (Next.js 16 + React 19 + shadcn/ui + Tailwind v4).

**Cherry-pick, don't port the whole thing.** Dataset-Tools has its own API routes that conflict with KNX's FastAPI backend. Only port components and client-side logic.

### 21.2 Theme Picker with OKLCH Accent Colors

**Priority:** Medium (Beta polish)
**Status:** ⏳ Not started

**What Dataset-Tools has that KNX lacks:**
- 7 accent colors: zinc (default), red, orange, green, blue, violet, pink
- Each accent defines 8 CSS variables for light mode + 8 for dark mode (primary, accent, ring, sidebar variants) using OKLCH format
- Applied via `[data-accent="red"]` attribute selectors on `<html>`
- 6 customizer UI variants: Pill (floating), Bar, Sidebar, Dock, Corner, Toolbar
- Settings persistence in localStorage with cross-tab sync via StorageEvent

**What KNX currently has:**
- Dark/light/system toggle only (no accent colors)
- Purple-tinted default palette (primary = `oklch(0.468 0.272 279.601)`)
- Duplicate `:root`/`.dark` blocks in `globals.css` that need consolidation

**Integration plan:**
1. Consolidate duplicate CSS variable blocks in `globals.css`
2. Add `[data-accent]` override blocks for all 7 accent colors (from Dataset-Tools `globals.css` lines 118-281)
3. Create `AccentColor` type + localStorage persistence + `useSettings` hook
4. Add inline `<script>` in `layout.tsx` to read accent from localStorage before React hydration (prevents flash)
5. Add Appearance section to settings page with color picker
6. Adapt one customizer variant (Pill or Toolbar most practical)
7. Fix Dataset-Tools bug: customizer sets `data-theme-color` but CSS uses `data-accent`

**Files to port/adapt from Dataset-Tools:**
- `components/ui/theme-customizer.tsx` (477 lines — 6 variants)
- `components/ui/color-swatch.tsx` (120 lines)
- `components/ui/color-swatch-selector.tsx` (81 lines)
- `hooks/use-outside-click.tsx` (24 lines)
- `types/settings.ts` (31 lines — AccentColor type)
- `lib/settings.ts` (29 lines — localStorage persistence)
- `hooks/use-settings.ts` (47 lines — cross-tab sync)

**Dependencies:** Zero new npm packages needed. Everything (`next-themes`, `motion`, `lucide-react`, Radix, CVA) already installed.

### 21.3 AI Image Metadata Parser

**Priority:** High (needed for dataset inspection)
**Status:** ⏳ Not started

**What Dataset-Tools has:**
- 1,832-line metadata parsing engine (`api/metadata/route.ts`)
- PNG tEXt chunk reader/writer (`lib/png-metadata.ts`, 147 lines)
- EXIF extraction (`exif-parser`)
- ComfyUI workflow graph traversal (deterministic, 90% success rate)
- A1111/Forge/NovelAI/Civitai format support
- SafeTensors metadata viewer (`components/safetensors-panel.tsx`)
- ComfyUI node registry lookup (`lib/comfyui-node-registry.ts`)

**Integration approach:**
- Metadata parser can run as Next.js API routes (already Node.js, no FastAPI conversion needed)
- SafeTensors viewer + ComfyUI node registry are client-side components
- Would add new `/api/metadata` route alongside existing KNX API routes

### 21.4 Components NOT to Port

- File tree, theme system base, settings management — already in KNX
- Background/particle effects — KNX already has many
- Glass notification — deferred (lower priority)
- All Dataset-Tools API routes that do file system access — conflicts with FastAPI backend
- `three.js` particles — heavy dep (~600KB) unless specifically wanted

---

**Document maintained by:** Ecosystem Project
