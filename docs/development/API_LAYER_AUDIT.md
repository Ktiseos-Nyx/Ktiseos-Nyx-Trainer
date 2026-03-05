# API Layer Audit - March 2026

Temporary tracking doc for API layer mismatches found during audit.
**Delete this file when all items are resolved.**

---

## Status Key
- [ ] Not started
- [x] Fixed
- [~] Partial / needs verification

---

## CRITICAL (Breaks functionality)

### 1. ~~`lr_power` TOML key is wrong~~
- **File:** `services/trainers/kohya_toml.py` line ~305
- **Fix:** Renamed `args["lr_power"]` → `args["lr_scheduler_power"]`
- [x] FIXED

### ~~2. HuggingFace upload request model completely mismatched~~
- **Files:** `api/routes/utilities.py`, `services/models/lora.py`, `services/lora_service.py`
- **Fix:** Rewrote all three layers:
  - Route request model now accepts `owner`, `repo_name`, `repo_type`, `selected_files`, `remote_folder`, `commit_message`, `create_pr`
  - Route constructs `repo_id = f"{owner}/{repo_name}"` and passes to service
  - Service model supports multi-file `file_paths`, `repo_type`, `remote_folder`, `create_pr`
  - Service method uploads each file individually via `HfApi.upload_file()`, tracks uploaded/failed
  - Response returns `repo_id`, `uploaded_files`, `failed_files` matching frontend expectations
  - Removed old metadata fields (`model_type`, `base_model`, `trigger_word`, `tags`, `description`) and README generation
- [x] FIXED

### ~~3. HF token validation: body vs query param~~
- **File:** `api/routes/utilities.py`
- **Fix:** Added `ValidateTokenRequest` Pydantic model, updated function to use `request.hf_token`
- [x] FIXED

---

## HIGH SEVERITY (Silent data loss)

### ~~4. `weight_decay` field orphaned~~
- **FALSE POSITIVE** — `weight_decay` IS already in the TOML (line 308 of kohya_toml.py). Audit agent missed it.
- [x] NOT A BUG

### ~~5. `dim_from_weights` field orphaned~~
- **File:** `services/trainers/kohya_toml.py`
- **Fix:** Added `if self.config.dim_from_weights: args["dim_from_weights"] = True` in network config section
- [x] FIXED

### ~~6. LoRA file list extension field name mismatch~~
- **File:** `api/routes/utilities.py`
- **Fix:** Renamed backend field `extension` → `file_extension` to match frontend
- [x] FIXED

### ~~7. Multi-extension glob broken~~
- **File:** `api/routes/utilities.py`
- **Fix:** Split comma-separated extensions and glob each separately, with deduplication
- [x] FIXED

---

## MEDIUM SEVERITY (Default mismatches, missing validation)

### ~~8. `guidance_scale` default mismatch~~
- **Fix:** Python default updated `1.0` → `3.5` (standard Flux dev guidance scale, matches frontend)
- [x] FIXED

### ~~9. `lr_scheduler` default mismatch~~
- **Fix:** Python default updated `COSINE` → `COSINE_WITH_RESTARTS` (matches frontend, standard for LoRA)
- [x] FIXED

### ~~10. `lr_scheduler_number` default mismatch~~
- **Fix:** Python default updated `1` → `3` (matches frontend, sensible for cosine_with_restarts)
- [x] FIXED

### ~~11. `cross_attention` validation gap + missing `mem_eff_attn`~~
- **Fix:** Added `MEM_EFF_ATTN = "mem_eff_attn"` to Python enum, `CrossAttentionSchema` Zod enum in frontend, and `mem_eff_attn` boolean mapping in TOML generation
- [x] FIXED

### ~~12. Unvalidated string fields that should be enums~~
- **Fix:** Added Pydantic enums (`SavePrecision`, `SampleSampler`, `TimestepSampling`, `ModelPredictionType`) in `training.py` and matching Zod schemas in `validation.ts`. All values sourced from Kohya SS argparse `choices`.
- [x] FIXED

### ~~13. Utility endpoint string fields unvalidated~~
- **File:** `api/routes/utilities.py`
- **Fix:** Added `Literal` types for `device`, `save_precision`, `precision`, `model_type` on `LoRAResizeRequest`, `LoRAMergeRequest`, and `CheckpointMergeRequest`
- [x] FIXED

---

## LOW SEVERITY (Code quality, nice-to-have)

### ~~14. `lr_warmup_ratio` accepted but never used~~
- **File:** `services/trainers/kohya_toml.py`
- **Fix:** Kohya's `--lr_warmup_steps` accepts floats < 1 as ratios natively. TOML generation now uses `lr_warmup_ratio` as the value for `lr_warmup_steps` when `lr_warmup_steps == 0`. Removed the phantom `lr_warmup_ratio` TOML key that Kohya silently ignored.
- [x] FIXED

### ~~15. `sdxl_bucket_optimization` field exists but unused~~
- **Fix:** Verified this is NOT a real Kohya argument. Removed from Python model (`training.py`), Zod schema (`validation.ts`), TypeScript interface (`api.ts`), and form defaults (`useTrainingForm.ts`). No UI component rendered it.
- [x] REMOVED

### ~~16. `network_module` always overridden by `lora_type` mapping~~
- **Fix:** This IS correct behavior — `_get_network_config()` maps LoRA types to their proper network modules (e.g., LoCon → `lycoris.kohya` with `algo=locon`). Added documentation comment to the field in `training.py` clarifying it's derived from `lora_type`, not user-configurable.
- [x] DOCUMENTED

### 17. Kohya args we don't expose
Not bugs — just features we could add later:
- `--torch_compile` / `--dynamo_backend` (PyTorch 2.0 compilation)
- `--loss_type` / `--huber_*` (alternative loss functions)
- `--scale_weight_norms` (weight norm scaling for LoRA)
- `--caption_extension` (hardcoded in dataset TOML)
- `--initial_epoch` / `--initial_step` / `--skip_until_initial_step`
- `--base_weights*` (weight merging)
- **Status:** Future feature backlog, not actionable bugs

---

## Suggested Fix Order

**~~Quick wins (one-line fixes, high impact):~~ ALL DONE**
1. ~~#1 — lr_power TOML key~~ FIXED
2. ~~#4 — weight_decay orphaned~~ FALSE POSITIVE
3. ~~#5 — dim_from_weights orphaned~~ FIXED
4. ~~#6 — extension field name~~ FIXED
5. ~~#7 — multi-extension glob~~ FIXED

**~~Small fixes (a few lines each):~~ ALL DONE**
6. ~~#3 — HF token validation body vs query~~ FIXED
7. ~~#8-10 — default alignment~~ FIXED (Python aligned to frontend values)

**~~Medium fixes:~~ ALL DONE**
8. ~~#11 — cross_attention enum + mem_eff_attn~~ FIXED
9. ~~#12 — add validation enums for string fields~~ FIXED
10. ~~#13 — utility endpoint validation~~ FIXED

**~~Large fixes:~~ ALL DONE**
11. ~~#2 — HuggingFace upload rewrite~~ FIXED

**~~Low severity:~~ ALL DONE**
12. ~~#14 — `lr_warmup_ratio` → now properly feeds into `lr_warmup_steps`~~ FIXED
13. ~~#15 — `sdxl_bucket_optimization` phantom field~~ REMOVED
14. ~~#16 — `network_module` override documented~~ DOCUMENTED

**Not actionable (future backlog):**
15. #17 — Kohya args we don't expose yet
