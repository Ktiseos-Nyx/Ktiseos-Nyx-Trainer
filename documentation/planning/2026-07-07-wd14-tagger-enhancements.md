# WD14 Tagger Enhancement Plan

**Date:** 2026-07-07
**Status:** Draft
**Source:** [67372a/stable-diffusion-webui-wd14-tagger](https://github.com/67372a/stable-diffusion-webui-wd14-tagger)

## Current State

Our tagging system already has strong feature parity with the webui extension:

| Feature | We Have |
|---------|---------|
| WD14 v3 + v1.4 models (9 total) | Yes |
| 3-tier thresholding (overall/general/character) | Yes |
| Tag replacement (source→target pairs) | Yes |
| Undesired tags blacklist | Yes |
| Always-first tags / activation tags | Yes |
| Rating tag control (none/first/last) | Yes |
| Remove underscores | Yes |
| Character tag expansion (`name_(series)`) | Yes |
| ONNX acceleration | Yes |
| Append/overwrite/ignore modes | Yes |
| Batch size / workers | Yes |
| Recursive directory processing | Yes |
| Tag frequency reporting | Yes |
| Post-processing (add trigger, remove, replace) | Yes |
| Per-image tag editing (gallery) | Yes |
| BLIP + GIT captioners | Yes (more than webui) |

## What We're Missing

### 1. Interactive Tag Preview with Confidence Sliders

The webui extension shows a live tag table during/after interrogation with:
- Each tag's confidence score as a horizontal bar
- Per-tag threshold override (click to exclude/include)
- Color-coded categories (general/character/rating)

**Our current gap:** We show a progress bar and then write files. No interactive
tag review step before saving to disk.

**Potential approach:** Add an intermediate review step (similar to the gallery)
where users can adjust thresholds, exclude/keep tags per-image or per-batch,
then confirm to write. Could be a lightweight React component using the tag
confidence data from the interrogator's stdout.

### 2. Interrogation Cache Database (db.json)

The webui tagger maintains a `db.json` file alongside images that stores:
- sha256 checksums of processed images → interrogation results
- Deduplication: identical images picked up once
- Cumulative averaging: re-running combines with old results
- Query indexing: skip re-interrogating unchanged files

**Why it matters:** On repeated tag runs (e.g., after adjusting thresholds),
the cache avoids re-running the model on unchanged images. For large datasets
this is significant time savings.

**Files involved:**
- `tagger/uiset.py` — `QData` class, `read_json()` / `write_json()` / `apply_filters()`
- `tagger/json_schema/db_json_v1_schema.json` — schema validation

**Integration:** The cache file lives at the dataset root. The tagger runner
writes it after each batch interrogation, reads it on next run to skip
unchanged images. Would need a cache lookup step before the main inference
loop in our WD14 runner.

### 3. Additional Interrogator Models

The webui extension supports more model backends:

| Model | Engine | Our Status |
|-------|--------|------------|
| SmilingWolf WD14 v3/v2/v1 | ONNX | We have these |
| Z3D-E621-Convnext | ONNX | Missing — e621-style tags for furry/NSFW content |
| DeepDanbooru | TensorFlow | Missing — requires TF dependency |
| MLDanbooru | ONNX | Missing — Lightweight pytorch alternative |

**Most impactful missing model:** Z3D-E621-Convnext. It's ONNX (same infra as
our existing WD14 models), provides a different tag vocabulary (e621.net style)
for non-anime imagery.

### 4. Custom Output Filename Template

The webui supports `[name]`, `[hash:sha1]`, `[extension]`,
`[output_extension]` in the output filename format string:

```
[name].[hash:sha1].[output_extension]
```

Our system always writes `{image_stem}.txt`. No template customization.

### 5. Keep Tags (Threshold Override)

The webui has a "keep tags" field — tags that should always be included in
output even if their confidence falls below the threshold.

**Our equivalent:** We have `always_first_tags` and activation tags, but
those are injected post-tagging. The webui's "keep tags" works during the
interrogation itself (affects tag selection, not just ordering).

### 6. Escape Brackets Option

The webui has an "escape brackets" option that escapes `()` and `[]` in tag
names for SD prompt compatibility. We don't have this — tags with brackets
in their names could confuse SD parsers.

### 7. Weighted Tags Files

The webui can write weights to tag files:
```
(tag1:0.95), (tag2:0.87), (tag3:0.42)
```

Instead of:
```
tag1, tag2, tag3
```

Our system always writes unweighted tags.

### 8. Large Batch Mode (TF Data Generator)

The webui's `large_batch_interrogate()` method batches images via a TensorFlow
data generator with batch sizes up to 1024. This is specific to the TF-based
interrogator path. Our ONNX-based approach bates at model level (batch_size
default 8, max 16), which is more conservative but more portable.

---

## Recommended Implementation

### Phase 1 — Quick Wins (1-2 days)

| Feature | Effort | Impact |
|---------|--------|--------|
| Keep tags field | Small — add to TaggingConfig, pass to CLI | Useful for preserving specific tags |
| Escape brackets option | Small — one regex pass after tag generation | Prevents SD parsing issues |
| Weighted tags file option | Small — config flag + format switch | Useful for advanced workflows |
| Custom output filename template | Medium — new config field + filename logic | Niche but flexible |

### Phase 2 — New Model (2-3 days)

Add Z3D-E621-Convnext ONNX model to the model catalog:
- Add model config to `services/models/tagging.py` (TaggerModel enum + hf_repo)
- Add tag CSV to download list
- The model uses the same ONNX inference pipeline — just different input size
  and tag vocabulary

### Phase 3 — Cache Database (3-5 days)

Implement the db.json interrogation cache:
1. Add sha256 checksumming (already in webui code as `get_bytes_hash`)
2. Create `dataset_cache.json` reader/writer
3. Add cache lookup before inference (skip if checksum + model match)
4. Store per-image tag confidences for the interactive review UI

### Phase 4 — Interactive Tag Review UI (1-2 weeks)

Build an intermediate review step showing tag confidence sliders per image:
- Tag list with horizontal confidence bars
- Toggle to exclude/include specific tags
- Adjust threshold per-tag
- Category color coding (general/character/rating)
- Confirm to write, or go back and re-run with adjusted parameters

---

## File Changes

### Phase 1

| File | Change |
|------|--------|
| `services/models/tagging.py` | Add `keep_tags`, `escape_brackets`, `weighted_tags`, `output_filename_template` fields |
| `services/tagging_service.py` | Pass new params to CLI, post-process output for weighted/escaped format |
| `frontend/hooks/useTaggingForm.ts` | Add new fields to zod schema |
| `frontend/components/tagging/cards/TagProcessingCard.tsx` | Add keep tags input, escape toggle, weight toggle |
| `frontend/components/tagging/cards/BasicSettingsCard.tsx` | Add output filename template input |

### Phase 2

| File | Change |
|------|--------|
| `services/models/tagging.py` | Add `z3d-e621-convnext` to `TaggerModel` enum |
| `frontend/components/tagging/models.ts` | Add model metadata |
| `services/tagging_service.py` | Handle Z3D model path resolution |
| `trainer/derrian_backend/sd_scripts/finetune/tag_images_by_wd14_tagger.py` (if needed) | Or handle via wrapper |

### Phase 3

| File | Change |
|------|--------|
| `services/services/tagging_service.py` | Add `CacheManager` class for db.json read/write |
| `services/models/tagging.py` | Add cache-related config options (`use_cache`, `cache_path`) |
| `services/tagging_service.py` | Add cache lookup before inference, write after |

### Phase 4

| File | Change |
|------|--------|
| `frontend/components/tagging/TagReviewOverlay.tsx` | New — tag confidence review with sliders |
| `frontend/app/dataset/[name]/auto-tag/page.tsx` | Add review step after progress completes |
| `services/tagging_service.py` | Return per-image tag confidences as JSON for review |
| `api/routes/dataset.py` | New endpoint to confirm/save reviewed tags |

---

## Open Questions

- Is the db.json cache worth the complexity for our service-style workflow?
  (Webui runs tagger repeatedly as user tweaks sliders; we run it once-per-config.)
- Z3D-E621 model — is the e621 vocabulary useful for the project's userbase?
  (Furry/anthro tagging differs from anime/Danbooru.)
- Interactive tag review — should it be per-image or batch-level?
  (Batch is simpler; per-image gives more control but more clicks.)
- Custom output filename format — should also support directory structuring?
  (E.g., output to subdirectories per character tag.)
