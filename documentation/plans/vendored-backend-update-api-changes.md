# Vendored Backend Update - API Layer Changes Plan

**Branch**: Create new branch from `fix/vastai-frontend-binding-112` when starting this work
**Status**: Alpha - keeping up with upstream changes
**Priority**: Wire up new model types, then polish

---

## What Changed (Vendored Updates)

### sd-scripts (updated from Ktiseos-Nyx/sd-scripts main)
- **New**: `anima_train.py` / `anima_train_network.py` - Anima model training
- **New**: `hunyuan_image_train_network.py` - HunyuanImage 2.1 LoRA (no finetune script)
- **New**: `anima_minimal_inference.py`, `hunyuan_image_minimal_inference.py`
- **Updated**: `train_util.py` (338 lines changed), `train_network.py`, Lumina support, Flux fixes
- **Updated**: `tag_images_by_wd14_tagger.py` (major rewrite, 619+ lines changed)
- **New strategy files**: `library/strategy_anima.py`, `library/strategy_hunyuan_image.py`
- **New model files**: `library/anima_models.py`, `library/anima_utils.py`, `library/anima_train_utils.py`, `library/hunyuan_image_*`

### LyCORIS (updated from Ktiseos-Nyx/LyCORIS dev branch, v3.1.1 -> v3.2.0)
- **New**: Anima module support, HunYuanVideo/Wan2.1 support
- **New**: `onfly_merge` / `onfly_restore` for inference-time weight merging
- **New**: LoRA-plus learning rate scaling
- **New**: Experimental RamTorch support
- **Updated**: Device/scalar fixes for glora, locon
- **Updated**: `kohya.py` (major refactor), `wrapper.py`, all module files

### Dependencies Bumped
- `accelerate` 0.33.0 -> 1.6.0 (major - required by updated sd-scripts)
- `pytorch_optimizer` 3.1.2 -> 3.9.0
- `prodigy-plus-schedule-free` unpinned -> 1.9.2
- `bitsandbytes` installer shim removed (pip handles it natively now)

---

## API Layer Changes (5-Layer Checklist per Model Type)

The architecture follows a clean, repeatable pattern. Each new model type touches 5 layers:

### Layer 1: Enum + Schema (`services/models/training.py`)
- [ ] Add `ANIMA = "Anima"` to `ModelType` enum
- [ ] Add `HUNYUAN_IMAGE = "HunyuanImage"` to `ModelType` enum
- [ ] Add string mappings to `normalize_model_type` validator
- [ ] Add model-specific optional fields to `TrainingConfig`:

**Anima fields:**
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `qwen3` | `Optional[str]` | None | Path to Qwen3-0.6B text encoder |
| `llm_adapter_path` | `Optional[str]` | None | Path to LLM adapter weights |
| `llm_adapter_lr` | `Optional[float]` | None | LR for LLM adapter (None=base, 0=freeze) |
| `self_attn_lr` | `Optional[float]` | None | LR for self-attention layers |
| `cross_attn_lr` | `Optional[float]` | None | LR for cross-attention layers |
| `mlp_lr` | `Optional[float]` | None | LR for MLP layers |
| `mod_lr` | `Optional[float]` | None | LR for AdaLN modulation layers |
| `t5_tokenizer_path` | `Optional[str]` | None | Path to T5 tokenizer dir |
| `qwen3_max_token_length` | `Optional[int]` | 512 | Max Qwen3 tokens |
| `t5_max_token_length` | `Optional[int]` | 512 | Max T5 tokens |
| `vae_chunk_size` | `Optional[int]` | None | Spatial chunk size for VAE |
| `vae_disable_cache` | `Optional[bool]` | False | Disable VAE caching |
| `attn_mode` | `Optional[str]` | None | torch/xformers/flash/sageattn |
| `split_attn` | `Optional[bool]` | False | Split attention for memory |
| `unsloth_offload_checkpointing` | `Optional[bool]` | False | Offload activations to CPU (LoRA only) |

**HunyuanImage fields:**
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `text_encoder_path` | `Optional[str]` | None | Path to Qwen2.5-VL (bfloat16) |
| `byt5_path` | `Optional[str]` | None | Path to byT5 model (float16) |
| `fp8_scaled` | `Optional[bool]` | False | Scaled fp8 for DiT |
| `fp8_vl` | `Optional[bool]` | False | fp8 for VLM text encoder |
| `text_encoder_cpu` | `Optional[bool]` | False | Run text encoders on CPU |
| `discrete_flow_shift` | `Optional[float]` | 5.0 | Flow shift for Euler scheduler |
| (shares `vae_chunk_size`, `attn_mode`, `split_attn` with Anima) |

**Shared DiT fields** (already partially exist for Flux/SD3/Lumina):
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `discrete_flow_shift` | `Optional[float]` | varies | Anima=1.0, HunYuan=5.0, Lumina=6.0 |
| `timestep_sampling` | `Optional[str]` | varies | sigma/uniform/sigmoid/shift/flux_shift |

### Layer 2: Script Mapping (`services/trainers/kohya.py`)
- [ ] Add to LoRA script map:
  - `ModelType.ANIMA: "anima_train_network.py"`
  - `ModelType.HUNYUAN_IMAGE: "hunyuan_image_train_network.py"`
- [ ] Add to Checkpoint script map:
  - `ModelType.ANIMA: "anima_train.py"`
  - (HunyuanImage has NO finetune script - LoRA only)
- [ ] Fix `ModelType.CHROMA` - uses Flux scripts with `--model_type chroma` arg
- [ ] Fix `ModelType.SD35` - currently falls through to default (wrong)

### Layer 3: TOML Generation (`services/trainers/kohya_toml.py`)
- [ ] Add Anima to resolution-format list (bucket reso steps = 16)
- [ ] Add HunyuanImage to resolution-format list (bucket reso steps = 32)
- [ ] Add Anima arg block in `_get_training_arguments()`:
  - `qwen3`, `t5_tokenizer_path`, token lengths
  - 6-group learning rates (llm_adapter, self_attn, cross_attn, mlp, mod)
  - VAE settings, attention mode
- [ ] Add HunyuanImage arg block:
  - `text_encoder`, `byt5`, fp8 settings
  - `text_encoder_cpu`, `discrete_flow_shift`
  - VAE settings, attention mode

### Layer 4: Validation (`api/routes/training.py`)
- [ ] Add Anima validation block:
  - Warn if `qwen3` path not set
  - Validate attention mode choices
  - Ensure LR fields are non-negative
- [ ] Add HunyuanImage validation block:
  - Require `text_encoder_path` and `byt5_path`
  - Validate fp8 settings
  - Note: LoRA only - reject checkpoint mode

### Layer 5: Frontend
- [ ] `frontend/lib/api.ts` - add `'Anima' | 'HunyuanImage'` to ModelType union
- [ ] `frontend/lib/validation.ts` - add to Zod enum
- [ ] `frontend/components/training/cards/ProjectSetupCard.tsx`:
  - Add dropdown options for Anima and HunyuanImage
  - Add conditional model path fields (Qwen3, byT5, etc.)
- [ ] `frontend/components/training/cards/AdvancedCard.tsx`:
  - Add Anima-specific LR group controls
  - Add HunyuanImage fp8/attention settings
- [ ] `frontend/store/trainingStore.ts` - add default values for new fields

---

## Also Check/Update

- [ ] **README.md** - update supported model list, add Anima/HunyuanImage mentions
- [ ] **CLAUDE.md** - update "Supports SDXL, SD 1.5, and experimental Flux, SD 3.5" line
- [ ] **Frontend docs page** (`frontend/app/docs/page.tsx`) - update if it lists supported models
- [ ] **Diagnose tool** (`diagnose.py`) - may need updates for new deps
- [ ] **WD14 tagger** - `tag_images_by_wd14_tagger.py` had a major rewrite (619+ lines), check if our `custom/tag_images_by_wd14_tagger.py` needs reconciliation

---

## Existing Issues to Fix While We're Here

- [ ] `ModelType.CHROMA` is in enum but has no script mapping (uses Flux scripts + `--model_type chroma`)
- [ ] `ModelType.SD35` is in enum but not in script maps (falls through to wrong default)
- [ ] Lumina is wired in backend but verify frontend exposes it properly

---

## Architecture Note: Adding Non-sd-scripts Trainers

The 5-layer pattern works for ANY training backend, not just Kohya:

1. **Enum + Schema** - model/trainer type
2. **Script/Process mapping** - what to execute and how
3. **Config generation** - TOML, YAML, CLI args, whatever the trainer expects
4. **Validation** - trainer-specific rules
5. **UI** - dropdown + conditional fields

`services/trainers/kohya.py` could become one of many trainer backends.
A future `services/trainers/simpletuner.py` (or whatever) follows the same interface.
The `TrainingService._create_trainer(config)` already has a comment about future trainer selection.

---

## ZLUDA / AMD Note

sd-scripts README says "NO ZLUDA SUPPORT" but this is a support disclaimer, not a technical limitation.
ZLUDA translates CUDA calls to HIP/ROCm at runtime - the training code doesn't know the difference.
In theory it works out of the box for AMD users. We should note this in our README as
"unofficial/community-tested" rather than "unsupported."
