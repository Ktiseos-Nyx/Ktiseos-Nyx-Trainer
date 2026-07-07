# Musubi Tuner / Blissful Tuner Integration Ideas

**Date:** 2026-07-07
**Status:** Draft (exploratory)
**Source:** [67372a/blissful-tuner](https://github.com/67372a/blissful-tuner) (fork of
[KohyaSS/musubi-tuner](https://github.com/kohya-ss/musubi-tuner))

---

## What It Is

Musubi Tuner is Kohya SS's extension for **video model training** — HunyuanVideo,
Wan 2.1/2.2, FramePack, plus newer image models (FLUX.2, Qwen-Image, Z-Image,
Kandinsky 5, HiDream O1, Ideogram 4, Krea 2). Blissful Tuner is Sarania's
feature-packed fork with latent previews, advanced CFG scheduling, fp16 accumulation,
RIFLEx, and more.

Our project currently only supports SD1.5/SDXL/Flux/SD3.5 LoRA training (via the
vendored `sd_scripts`). Musubi Tuner opens the door to a much wider model landscape.

---

## What's Relevant to Us

### 1. Qwen-VL Captioning (`caption_images_by_qwen_vl.py`)

A third captioning backend option alongside JoyCaption and BLIP/GIT.

**How it differs from JoyCaption:**
| Aspect | JoyCaption (GGUF) | Qwen-VL (Musubi) |
|--------|------------------|------------------|
| Runtime | llama.cpp / GGUF | HuggingFace transformers |
| Model | LLaVA-based JoyCaption Beta | Qwen2.5-VL |
| Precision control | n_gpu_layers only | fp8 / bf16 |
| Output | Individual `.txt` files | JSONL or `.txt` |
| Caption types | 10+ built-in styles | Custom prompt only |
| Image sizing | No resize | Resize to buckets (28px factor) |
| GPU mem | Lower (GGUF quant) | Higher (full model) |

**Integration approach:** Same coroutine-job pattern as JoyCaption. The Qwen-VL
runner would load the model via `transformers` + `load_qwen2_5_vl()` from
musubi_tuner, process images sequentially, write caption files.

**Not yet supported by our AGENTS.md:** Qwen Image LoRA training — the vendored
sd-scripts has no training script. But Musubi Tuner *does* have it
(`qwen_image_train_network.py`). This is relevant for SD3.5/other image models
that use Qwen as text encoder.

### 2. Video Model Training Support

Musubi Tuner adds training for these architectures that our system doesn't support:

| Model | Type | Training Scripts |
|-------|------|-----------------|
| HunyuanVideo | T2V video | `hv_train_network.py` |
| Wan 2.1/2.2 | T2V/I2V video | `wan_train_network.py` |
| FramePack | T2V video | `fpack_train_network.py` |
| FLUX.2 dev/klein | Image (next-gen) | `flux_2_train_network.py` |
| Qwen-Image | Image (SD3.5-like) | `qwen_image_train_network.py` |
| Z-Image | Image (novel arch) | `zimage_train_network.py` |
| Kandinsky 5 | Image/Video | `kandinsky5_train_network.py` |
| HiDream O1 | Image | `hidream_o1_train_network.py` |
| Ideogram 4 | Image | `ideogram4_train_network.py` |
| Krea 2 | Image | `krea2_train_network.py` |

### 3. LoRA Utilities

| Script | What It Does |
|--------|-------------|
| `merge_lora.py` | Merge multiple LoRAs into one (similar to our merge page) |
| `convert_lora.py` | Convert LoRA between formats (similar to our resize/convert) |
| `lora_post_hoc_ema.py` | Apply EMA to LoRA weights post-training |

### 4. Blissful Tuner's Extended Features

The Sarania fork adds inference-focused features that could inspire our UI:
- Latent preview during generation (latent2RGB / TAE)
- CFG scheduling (per-step guidance scale)
- Prompt weighting and wildcards
- RIFLEx for longer video generation
- fp16 accumulation for faster inference
- Various sampler/distilled model support

---

## Integration Approaches

### Option A: Captioning-Only (Low Effort)

Treat `caption_images_by_qwen_vl.py` as inspiration for a third captioning
backend, same pattern as JoyCaption. No new training support.

**Estimated effort:** 2-3 days (runner, config model, API, frontend dropdown)

### Option B: Add Video Training Tabs (Medium Effort)

Add a "Video Training" section alongside LoRA / Checkpoint training, starting
with HunyuanVideo and Wan (most popular). Reuse the existing training UI patterns
(model selection, dataset config, hyperparams) but point at musubi_tuner scripts.

**Challenges:**
- Video dataset handling is very different from image datasets (frame counts,
  video_length, bucket configs)
- Pre-caching step (latents + text encoder) is required before training
- Musubi uses toml config files like our system — good overlap
- Different model download sources (HuggingFace, not Civitai)

**Estimated effort:** 1-2 weeks per model

### Option C: Full Training Backend Integration (High Effort)

Vendor musubi_tuner alongside sd_scripts as an alternative training backend,
with model routing based on selected architecture.

**Challenges:**
- ~25KLOC+ of new vendored code
- Python dependency conflicts with sd_scripts (different torch/etc. reqs)
- Would likely need separate venvs or careful dependency management
- UI overhaul to handle model type switching (image vs video workflows)

**Estimated effort:** 4-6 weeks

---

## Dependencies

Qwen-VL captioning needs:
```
torch
transformers>=4.48.3
qwen-vl-utils (if not bundled with transformers)
```

Video training needs much more: `torchvision`, `accelerate`, `sentencepiece`,
`einops`, and architecture-specific packages (e.g. `hunyuan_model` deps).

---

## Open Questions

- Is video training in scope for this project, or do we stay focused on
  image LoRA/checkpoint training?
- Does the `caption_images_by_qwen_vl.py` approach (transformers-based VL)
  overlap enough with JoyCaption to skip it, or is having both useful?
  (JoyCaption: smaller/cheaper GGUF; Qwen-VL: more accurate/bigger model)
- Should we integrate `merge_lora.py` or `convert_lora.py` as alternatives
  to our existing utilities?
- The AGENTS.md says "Qwen Image LoRA — do not add UI for it until upstream
  ships the training script" — Musubi Tuner *is* the upstream having shipped
  it. Should we update that restriction?
