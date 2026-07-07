# Final Batch — Repo Summary

**Date:** 2026-07-07
**Status:** Draft

---

## Tier 1: Directly Training-Relevant

### AI Toolkit (ostris/ai-toolkit, forked to duskfallcrew/ai-toolkit)

The most popular community training toolkit for diffusion models. By Ostris.
Web UI + CLI, YAML-configured. Fairly massive project.

| Aspect | Detail |
|--------|--------|
| Upstream | github.com/ostris/ai-toolkit (1248 commits) |
| Fork | github.com/duskfallcrew/ai-toolkit |
| License | MIT |
| Models | FLUX.1/2, Flex.1/2, Chroma, Lumina2, Qwen-Image, HiDream, OmniGen2, Z-Image, SDXL, SD1.5, ERNIE-Image, Wan 2.1/2.2, LTX 2/2.3, Krea 2, ACE Step audio |
| Methods | LoRA, LoKr, layer-specific training (only_if_contains / ignore_if_contains) |
| UI | Node.js web UI at port 8675 |
| Hardware | Consumer GPU focus (24GB), RunPod + Modal support |
| Windows | Works via AI-Toolkit-Easy-Install companion repo |

**Relevance:** High. This is the most popular alternative to Kohya SS for
consumer training. Its YAML config system, layer targeting, and model support
list overlap significantly with our feature set. Worth evaluating whether parts
of its architecture (extension system, config parsing) could inform our next-gen
training backend. Already forked by you.

### sd-scripts-f2vae (bluvoll)

Fork of kohya-ss/sd-scripts (same codebase we vendor) adding:
- `--vae_type=flux2` — Flux2 VAE training support (latent_channels=32)
- `--flow_model`, `--flow_use_ot` — Rectified Flow training target
- `--flow_timestep_distribution=uniform|logit_normal`
- `--vae_custom_scale`, `--vae_custom_shift` — VAE scaling params
- `--vae_reflection_padding` — Fixes edge artifacts with custom VAEs
- `--contrastive_flow_matching` — Sharper results via CFM loss
- Used to train Mugen (RF + Flux2 VAE), ChenkinRF (RF only)
- 11 stars, 2 forks

**Relevance:** Direct. These are patches to the sd-scripts codebase that could be
applied to our vendored sd-scripts. Mugen and similar RF+Flux2VAE checkpoints
are already on Civitai — if users want to train LoRAs for them, we need these
flags. See AGENTS.md constraint about vendored backend patches.

### diffusion-pipe (tdrussell)

DeepSpeed-based pipeline parallel training. 2k stars, 280 forks.
- Supports: SDXL, Flux, Flux 2, LTX-Video, HunyuanVideo, Wan 2.1/2.2, Qwen-Image,
  Z-Image, Anima, HiDream, Chroma, AuraFlow, Ideogram4, Krea2, and many more
- Pre-caches latents + text embeddings to disk via HF Datasets
- PEFT LoRA + LyCORIS + full-rank + ControlNet
- DeepSpeed pipeline parallelism across GPUs
- Tensorboard logging, eval set metrics, checkpoint resume
- **Requires DeepSpeed = WSL on Windows** (docs say native Windows is difficult)

**Relevance:** Alternative training backend. Much broader model support than
our Kohya-based system. Good reference architecture for pre-caching and
pipeline parallelism. Could complement our system for models we don't support.

### SimpleTuner (bghira)

Pip-installable training system. 2.9k stars, 283 forks.
- Supports even more models: SDXL, SD3, Flux.1, Flux.2, Ideogram 4, Anima,
  Z-Image, Qwen Image, Wan, HiDream, Chroma, LTX Video, Sana, Lumina2, etc.
- Full training web UI with job queue, users, RBAC, quotas, audit logs
- CaptionFlow integration for dataset captioning
- Worker orchestration (distributed GPU workers with SSE dispatch)
- SSO, role-based access, spending limits
- DeepSpeed + FSDP2, S3 training, concept sliders
- `pip install simpletuner`

**Relevance:** The most comprehensive FOSS training platform. The Web UI
architecture (job queue, multi-user, worker orchestration) overlaps with our
goals. The enterprise features (RBAC, quotas, SSO) could inform our future
multi-user support. Very active — 169 releases, latest Jun 2026.

### CaptionFlow (bghira)

Distributed vLLM-powered captioning system. 13 stars.
- Orchestrator + worker architecture over WebSocket
- vLLM backend for fast batched inference
- Supports any vLLM-compatible vision model
- Automatic data serving to remote workers
- Export formats: jsonl, csv, json, txt, parquet, arrow, webshart
- Integrated in SimpleTuner

**Relevance:** Alternative captioning architecture to our in-process approach.
The distributed design (workers on separate GPUs/machines) is useful for
large-scale dataset captioning. The orchestrator pattern differs from our
per-job subprocess model.

### dataset-builder (cagliostrolab)

Gallery-dl + Jupyter notebook for dataset scraping/cleaning/classification.
37 stars. Simple workflow: scrape JSON → clean → classify → order → download.

**Relevance:** Lightweight reference for dataset preparation pipeline. Our
system doesn't have a scraping/download feature for building datasets from
scratch — this shows one approach. Not a priority.

---

## Tier 2: Inference Nodes / ComfyUI (not relevant to training)

| Repo | Stars | What It Does | Why Skip |
|------|-------|-------------|----------|
| ComfyUI-Anima-LLLite | 150 | Anima ControlNet-LLLite inference | Inference node only |
| ControlNet-LLLite-ComfyUI | — | ControlNet LLLite for SD | Inference node only |
| ComfyUI-Custom-Scripts | 3.1k | UI autocomplete, auto-arrange, image feed, etc. | ComfyUI-only |
| ComfyUI-extra-schedulers | — | Custom sampler schedulers | Inference only |
| comfyui-vae-reflection | — | VAE reflection padding fix | Inference only; related to f2vae training |
| ComfyUI-Prompt-Manager | — | Prompt storage/management | ComfyUI-only |
| civitai-comfy-nodes | 41 | Civitai cloud orchestration API nodes | Cloud inference, not local training |
| xl-converter | 1 | JPEG XL/AVIF image converter | We already have Convert to Format |

---

## Tier 3: Adjacent Tools (potentially useful as services)

### withoutbg-python (1.2k stars)

ONNX-based background removal. `pip install withoutbg`.
- Local (free, CPU ONNX) or cloud API (paid, better quality)
- Single inference pass at up to 768px
- Batch processing with progress callbacks
- ~495MB model download on first run, ~2GB RAM

**Relevance:** Could be added as a dataset preprocessing step (remove backgrounds
before tagging/training). Already a mature pip package with local mode.

### backgroundremover (8k stars, 648 forks)

U2Net-based CLI bg removal. `pip install backgroundremover`.
- Image + video with alpha matting
- Model choices: u2net (general), u2net_human_seg (people), u2netp (fast)
- Batch folder processing, pipe support, HTTP API
- Custom background color/image replacement
- Popular open-source standard

**Relevance:** Same as withoutbg — dataset preprocessing. More established but
heavier (requires PyTorch, only GPU optional). withoutbg might be a better
fit for our ONNX-only approach.

### horde-sdk (40 stars)

AI Horde API client. Free community GPU grid for generation.

**Relevance:** Not relevant to training. Interesting for inference distribution.

---

## Already Documented / Known

- **Chattiori-Model-Merger** — Already in memory #42. Pending design doc.
- **AI Toolkit / ComfyUI Manager** — User asked if these were mentioned earlier;
  nothing in this batch matches that name exactly.

---

## Action Items

| Item | Source | Priority |
|------|--------|----------|
| Apply f2vae patches to vendored sd-scripts | sd-scripts-f2vae | Medium |
| Produce design doc for Chattiori-Model-Merger integration | memory #42 | Medium |
| Evaluate SimpleTuner as alternative/companion training system | SimpleTuner | Low (large project) |
| Evaluate CaptionFlow distributed captioning for large datasets | CaptionFlow | Low |
| Add bg removal as dataset preprocessing step | withoutbg / backgroundremover | Low |
| Study diffusion-pipe pre-caching architecture | diffusion-pipe | Informational |
