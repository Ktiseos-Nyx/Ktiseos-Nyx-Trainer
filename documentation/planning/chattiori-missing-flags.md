# Chattiori Missing CLI Flags — Pluggable Tuning Knobs

**Status:** Planning  
**Priority:** Low (Phase 2 UI polish)  
**Parent:** `chattiori-integration-design.md`  
**Created:** 2026-07-13  
**Reason:** Phase 1 wired the core merge/bake subprocess but the vendored CLI
supports many more flags than our Pydantic models + service builders + frontend
types expose. This doc inventories every missing flag so we can add them in
priority order later.

---

## 1. Missing Bake Flags (`lora_bake.py` → `BakeRequest` + `_build_bake_command`)

The vendored CLI at `trainer/chattiori/lora_bake.py:2897-2953` accepts these
arguments that our service layer does **not** pass through.

### 1.1 Precision / output format

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--save_bhalf` | Save as bfloat16 | `save_bhalf: bool = False` | High |
| `--save_quarter` | Save as float8 (FP8) | `save_quarter: bool = False` | Medium |

Currently we only have `save_half` (fp16). bfloat16 is important for Flux/Anima
models where the native precision is often bf16.

### 1.2 DARE merge during bake

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--dare` | Apply DARE (Drop-And-REscale) to LoRA weights before baking | `dare: bool = False` | Medium |

Drops a percentage of LoRA weight entries and rescales the remainder — reduces
overfitting artifacts in the baked output.

### 1.3 Multi-LoRA pre-merge before baking

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--merge_loras` | Merge all LoRAs together first, then bake the merged result | `merge_loras: bool = False` | Medium |
| `--merge_rank` | Rank for the merged LoRA (default 64) | `merge_rank: int = 64` | Low |
| `--merge_arch` | Architecture override: `"auto"`, `"lora"`, `"locon"`, `"loha"`, `"lokr"`, `"dylora"` | `merge_arch: str = "auto"` | Low |
| `--merge_norm` | Normalization: `"none"`, `"sqrt"`, `"mean"` | `merge_norm: str = "sqrt"` | Low |
| `--merge_scale` | Scale factor for merged LoRA | `merge_scale: float = 1.0` | Low |
| `--merge_unet_only` | Merge only UNet/DiT, skip text encoder | `merge_unet_only: bool = False` | Low |
| `--merge_clamp_q` | Clamp quantile for weight clipping | `merge_clamp_q: float` | Low |
| `--merge_intermediate_mult` | Intermediate rank multiplier | `merge_intermediate_mult: int = 4` | Low |

This is the "SVDSuperMerge" style workflow — merge N LoRAs into one, then bake
that merged LoRA. Currently users have to do two separate operations.

### 1.4 Baking quality / normalization

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--bake_norm` | Normalization mode: `"none"`, `"sqrt"`, `"mean"` | `bake_norm: str = "sqrt"` | Medium |
| `--bake_rank_cap` | Cap baked LoRA at this effective rank (0 = no cap) | `bake_rank_cap: int = 0` | Low |
| `--bake_clamp_q` | Clamp outlier weights at this quantile (0.0 = no clamp) | `bake_clamp_q: float = 0.0` | Low |
| `--bake_delta_cap` | Cap individual weight deltas (0.0 = no cap) | `bake_delta_cap: float = 0.0` | Low |
| `--bake_fp32` | Compute baking math in FP32 (slower, more precise) | `bake_fp32: bool = False` | Medium |

### 1.5 Guard / safety

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--bake_guard` | Guard mode: `"auto"`, `"cosine"`, `"l2"`, `"none"` | `bake_guard: str = "auto"` | Low |
| `--bake_guard_cap` | Guard cap threshold | `bake_guard_cap: float = 0.05` | Low |
| `--bake_guard_skip` | Fraction of tensors to skip guard on | `bake_guard_skip: float = 0.25` | Low |

Guards detect and cap per-tensor deltas that would cause quality degradation in
the output — useful as a safety net for experimental bakes.

### 1.6 Metadata / debugging

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--bake_budget_report` | Print per-tensor delta budget analysis | `bake_budget_report: bool = False` | Low |
| `--no_metadata` | Strip metadata from output | `no_metadata: bool = False` | Low |

---

## 2. Missing Merge Flags (`merge.py` → `CheckpointAdvancedMergeRequest` + `_build_merge_command`)

### 2.1 Stochastic / range ratios

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--rand_alpha` | Random alpha range, e.g. `"0.3-0.7"` — applies per-tensor random alphas | `rand_alpha: Optional[str] = None` | Medium |
| `--rand_beta` | Random beta range | `rand_beta: Optional[str] = None` | Medium |
| `--sand_alpha` | Sandwich-style alpha: different ratios per depth layer | `sand_alpha: Optional[str] = None` | Low |
| `--sand_beta` | Sandwich-style beta | `sand_beta: Optional[str] = None` | Low |

### 2.2 Difference/comparison modes

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--use_dif_10` | Use model1 - model0 as model B (enables cross-version diff merging) | `use_dif_10: bool = False` | Low |
| `--use_dif_20` | Use model2 - model0 as model B | `use_dif_20: bool = False` | Low |
| `--use_dif_21` | Use model2 - model1 as model B | `use_dif_21: bool = False` | Low |

### 2.3 Custom model names in metadata

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--m0_name` | Custom name for model 0 in output metadata | `m0_name: Optional[str] = None` | Low |
| `--m1_name` | Custom name for model 1 in output metadata | `m1_name: Optional[str] = None` | Low |
| `--m2_name` | Custom name for model 2 in output metadata | `m2_name: Optional[str] = None` | Low |

### 2.4 CFG sensitivity tuning

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--cfg_sens` | CFG sensitivity scalar (default 1.0) | `cfg_sens: float = 1.0` | Low |
| `--cfg_sens_targets` | Which layer types to target: `"kv,out"`, `"all"`, etc. | `cfg_sens_targets: str = "kv,out"` | Low |

Scales CFG-relevant attention layers differently from non-CFG layers during
merge — reduces CFG artifacts in merged checkpoints.

### 2.5 Saturation boost family

| CLI flag | What it does | Pydantic field needed | Priority |
|----------|-------------|----------------------|----------|
| `--sat_boost` | Saturation boost strength (default 1.0) | `sat_boost: float = 1.0` | Low |
| `--sat_boost_side` | Apply boost to `"alpha"`, `"beta"`, or `"both"` | `sat_boost_side: str = "alpha"` | Low |
| `--sat_boost_tags` | Only apply boost to tensors matching these tags | `sat_boost_tags: Optional[str] = None` | Low |
| `--sat_profile` | Saturation profile: `"legacy"` or `"safe_attn2_out"` | `sat_profile: str = "legacy"` | Low |
| `--sat_delta_cap_pct` | Cap delta as percentage of parameter value (0.0 = no cap) | `sat_delta_cap_pct: float = 0.0` | Low |
| `--sat_boost_mix` | Mix ratio between boost and original (1.0 = full boost) | `sat_boost_mix: float = 1.0` | Low |
| `--boost_clamp` | Clamp boosted values: `"auto"`, `"clamp01"`, `"none"` | `boost_clamp: str = "auto"` | Low |
| `--vae_sat` | VAE-specific saturation scalar | `vae_sat: float = 1.0` | Low |
| `--fine_sat` | Fine-tune layer saturation scalar | `fine_sat: float = 1.0` | Low |

All nine flags form the saturation boost subsystem — adjusts how strongly
certain layers receive the merge delta, preventing "overcooking" of attention
and VAE layers.

### 2.6 Deturbo / turbo sub-modes

These are handled at the mode-argument level (e.g. `WS_turbo`, `WS_deturbo` as
mode string variants), not separate flags. Our mode field is a free string, so
passing `"WS_turbo"` already works. No new fields needed.

---

## 3. What Needs to Change (Per-Flag Pattern)

Adding one flag touches four layers:

```
1. services/models/chattiori.py    ← Pydantic model (BakeRequest or
                                      CheckpointAdvancedMergeRequest)
2. services/chattiori_service.py   ← _build_bake_command() or
                                      _build_merge_command()
3. frontend/lib/api.ts             ← TypeScript BakeRequest or
                                      CheckpointAdvancedMergeRequest type
4. frontend/components/merge/*     ← UI toggle/field for the flag
```

The pattern for a new boolean flag in step 2 is:

```python
if request.new_flag:
    command.append("--new_flag")
```

---

## 4. Priority Summary

| Priority | Flags | Impact |
|----------|-------|--------|
| **High** | `save_bhalf` (bake) | Flux/Anima native bf16 — saves ~40% size, no quality loss |
| **Medium** | `dare` (bake), `bake_norm` (bake), `save_quarter` (bake), `bake_fp32` (bake), `merge_loras` (bake), `rand_alpha`/`rand_beta` (merge) | Commonly requested by community |
| **Low** | Everything else | Power-user tuning knobs, useful but niche |

---

## 5. Excluded by Design

These vendored CLI features are intentionally NOT planned for exposure:

- `--checkpoint` positional arg in `lora_bake.py` — we always provide it
  explicitly, no change needed
- `--loras` positional arg — same
- `--mode` / `--model_path` / `--model_0` / `--model_1` in `merge.py` — already exposed
- `--output` — already exposed

---

## 6. Notes

- The `CheckpointAdvancedMergeRequest` Pydantic model already has all of Phase 1's
  core flags; only the sat_boost family, cfg_sens, rand/sand ratios, and diff
  modes are missing from it.
- The `BakeRequest` Pydantic model was patched on 2026-07-13 to add
  `lora_ratios`, `output_dir`, and `text_encoder_path` (three fields the
  frontend type was missing).
- The frontend `BakeRequest` TypeScript type in `api.ts` was fixed on
  2026-07-13 — removed duplicate definition and added the three missing
  fields.
- The `lora_bake.py` `--bake_norm` already defaults to `"sqrt"` in the CLI,
  so baking without passing it uses sqrt normalization. Adding the Pydantic
  field would let users choose `"none"` (no normalization, preserves
  original LoRA magnitudes).
