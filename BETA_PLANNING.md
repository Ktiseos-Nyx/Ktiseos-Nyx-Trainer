# Beta Planning Document

**Current Version:** Alpha (v0.1.0-dev)
**Target:** Beta Release
**Last Updated:** 2026-04-06

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

## 4. Feature Priority Matrix (Beta)

### Must Have (Alpha -> Beta gate)
| Feature | Category | Effort |
|---------|----------|--------|
| Tag Viewer with frequency counts | New Feature | Medium |
| Bulk tag remove/replace | New Feature | Medium |
| Fix alpha parameter UX in LoRA resize (MG-1) | Bug Fix | Small |
| Add subprocess timeouts to merge operations (MG-3) | Reliability | Small |
| CUDA availability check for merges (MG-4) | Error Handling | Small |

### Should Have (Beta quality)
| Feature | Category | Effort |
|---------|----------|--------|
| 3-way overwrite mode for tagging | Enhancement | Small |
| Checkpoint-specific validation (CT-1) | Enhancement | Small |
| Hide LoRA fields in checkpoint mode (CT-2) | UX | Medium |
| Merge progress reporting (MG-7) | UX | Medium |
| SD3 merge support (MG-5) | Feature | Small |

### Nice to Have (Beta+)
| Feature | Category | Effort |
|---------|----------|--------|
| Per-image visual tag editor | New Feature | Large |
| Caption editor with search highlighting | Enhancement | Medium |
| Merge presets/templates | UX | Medium |
| Merge dry-run/preview mode | Feature | Medium |

---

## 5. Attribution Requirements

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

## 6. Notes

- Checkpoint training backend appears functional but needs real-world testing with actual full fine-tune runs
- Merging tool is mostly solid - the issues are UX and reliability, not correctness
- The tag system upgrades are the highest-impact changes for Beta since they directly improve the dataset preparation workflow
- All Civitai-inspired features are UI patterns only - we do NOT use their cloud orchestrator, S3 upload, or SignalR approach

---

**Document maintained by:** Ktiseos-Nyx-Trainer Project
