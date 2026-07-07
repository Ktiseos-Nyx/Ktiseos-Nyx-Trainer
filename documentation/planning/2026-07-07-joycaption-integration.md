# JoyCaption Beta Integration Plan

**Date:** 2026-07-07
**Status:** Draft
**Priority:** High (caption quality upgrade)

## Overview

Replace/supplement the existing BLIP/GIT captioning backends with
[JoyCaption Beta](https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf),
a LLaVA-based vision-language model fine-tuned for image captioning, running
via GGUF/llama.cpp for GPU-accelerated inference.

### Why JoyCaption?

| Aspect | BLIP / GIT | JoyCaption Beta |
|--------|-----------|-----------------|
| Model type | Old encoder-decoder | Modern VLM (LLaVA arch) |
| Caption quality | Generic, often misses details | Rich, context-aware |
| Caption styles | One style | 10+ built-in (descriptive, danbooru, SD prompt, etc.) |
| Extra options | None | 25+ toggles (lighting, angle, NSFW rating, etc.) |
| Inference | PyTorch subprocess | GGUF / llama.cpp (in-process) |
| GPU efficiency | Full model in VRAM | Quantized GGUF, layer-offload control |
| Output length control | min/max_length only | Word count, length adjectives, token cap |

## JoyCaption Version History

There are multiple JoyCaption versions, each using a different model format:

| Version | Model Base | Format | Size | VRAM | Notes |
|---------|-----------|--------|------|------|-------|
| **Beta** (latest) | LLaVA 1.5 arch | GGUF via llama.cpp | ~5-8 GB | Lowest | Our primary target — GGUF quantized, efficient |
| **Alpha Two** | SigLIP + Llama 3.1 8B 4-bit | transformers (bnb) | ~8-12 GB | High | Full transformers, better quality vs Beta? |
| **Alpha One** | SigLIP + Llama 3.1 8B 4-bit | transformers (bnb) | ~8-12 GB | High | Earlier alpha, fewer caption types |
| **Pre-Alpha** | Older arch | transformers | ~6 GB | Medium | Legacy, not recommended |

**Decision:** Focus on GGUF Beta for v1 (lightest, fastest to load). The
transformers-based Alpha versions can be added later if needed — they'd use
a different runner pattern (HuggingFace `transformers` instead of llama.cpp).

## Additional Captioning Models (ComfyUI_CXH_joy_caption)

The [CXH node pack](https://github.com/67372a/Comfyui_CXH_joy_caption) bundles
three captioning approaches, all worth considering as future backends:

### Florence-2 (PromptGen v1.5)

A Microsoft Foundation Model for vision-language tasks. Lighter than JoyCaption,
fast for batch captioning. Two variants:
- `Florence-2-base-PromptGen-v1.5` (base)
- `Florence-2-large-PromptGen-v1.5` (large)

**Pros:** Small, fast, good for simple captions
**Cons:** Less creative/detailed than JoyCaption; no style control

### MiniCPM-v2.6 Prompt Generator

A smaller VLM by OpenBMB, fine-tuned specifically for prompt generation.
Runs via `transformers`, moderate VRAM (~6 GB).

### MiniCPM3-4B Chat

Lightweight chat-based model useful for caption rewrite/translation.

### ToriiGate

A Vision2Seq caption model by Minthy, based on Qwen architecture. Two versions:
v0.3 and v0.4-7B (Qwen2VL-based). Available on HF:
`Minthy/ToriiGate-v0.3`, `Minthy/ToriiGate-v0.4-7B`.

**Unique feature:** Can accept WD14 tags as grounding input — tags are passed
alongside the image prompt for better understanding. This means a two-pass
captioning pipeline: first WD14 tag → then ToriiGate caption with tags as
context.

Three prompt modes: JSON-like structured, long detailed, brief.

Like Qwen-VL and Florence-2, this is a `transformers` model (not GGUF).
Requires torch, transformers, bitsandbytes for quantization.

### Florence-2 / MiniCPM Integration Notes

These would follow the same coroutine-job pattern as the JoyCaption runner.
They use HuggingFace `transformers` (not GGUF), so they share the Qwen-VL
runner design more than the GGUF runner. Not planned for v1 but worth
documenting for future expansion.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 CaptioningService                │
│  (existing orchestrator, extended)              │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ BLIP runner │  │ GIT runner  │  │ JoyCap-  │ │
│  │ (subprocess)│  │ (subprocess)│  │ tion     │ │
│  └─────────────┘  └─────────────┘  │ runner   │ │
│                                     │ (in-proc)│ │
│                                     └──────────┘ │
└─────────────────────────────────────────────────┘
```

The JoyCaption runner is fundamentally different from BLIP/GIT:
- **In-process model loading** — the GGUF model lives inside the Python process
- **No subprocess** — `llama-cpp-python` does the inference directly
- **Model lifecycle** — loaded on first use, kept warm for batch, optionally
  unloaded to free VRAM
- **Batch = sequential** — each image goes through the model one at a time
  (llama.cpp doesn't batch vision APIs the same way)

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `services/models/joycaption.py` | Pydantic config models for JoyCaption |
| `services/captioning/joycaption_runner.py` | In-process JoyCaption runner (model load, generate, unload) |

### Modified Files

| File | Changes |
|------|---------|
| `services/captioning_service.py` | Add `start_joycaption()`, route to runner |
| `services/models/captioning.py` | Add `JOYCAPTION` to `CaptioningModel` enum |
| `api/routes/dataset.py` | Add `POST /api/dataset/caption/joycaption`, status, stop |
| `api/routes/jobs.py` (if separate) | Or route through existing job endpoints |
| `frontend/lib/api.ts` | Add `captioningAPI.startJoyCaption()` etc. |
| `services/__init__.py` | Register new module |

### Config / Infra

| File | Changes |
|------|---------|
| `installer.py` | Add `llama-cpp-python` with CUDA extra index URL |
| `requirements.txt` (optional) | Or just documented install step |

## JoyCaptionConfig Model

```python
class JoyCaptionConfig(BaseModel):
    dataset_dir: str                          # Image directory
    caption_extension: str = ".txt"           # Output caption extension
    caption_type: str = "Descriptive (Casual)" # See CAPTION_TYPE_MAP keys
    caption_length: str | int = "any"         # "any"|"short"|"medium"|"long"|number

    # Model paths (auto-resolved from models/ dir or user-specified)
    gguf_model: str                           # Filename in models/llava_gguf/
    mmproj_file: str                          # Filename (mmproj .gguf or .bin)

    # LLM inference params
    n_gpu_layers: int = -1                    # -1 = all layers on GPU
    n_ctx: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    seed: int = -1

    # Memory management
    keep_model_loaded: bool = True            # Keep in VRAM between jobs
    unload_when_done: bool = False            # Force unload after generation

    # Extra options (mapped from JoyCaption's EXTRA_OPTIONS)
    extra_options: list[str] = Field(default_factory=list)
    character_name: str = ""
```

## Runner Design (`joycaption_runner.py`)

The runner wraps `llama-cpp-python`'s `Llama` + `Llava15ChatHandler` with a
model-cache pattern adapted from the ComfyUI GGUF node:

```python
class JoyCaptionRunner:
    _instance: Optional["JoyCaptionRunner"] = None
    _model_key: Optional[tuple] = None
    _llm: Optional[Llama] = None
    _chat_handler: Optional[Llava15ChatHandler] = None
    _lock: asyncio.Lock = asyncio.Lock()

    async def load_model(self, config: JoyCaptionConfig) -> None
    async def generate(self, image_path: str, config: JoyCaptionConfig) -> str
    async def unload_model(self) -> None
    async def run_batch(self, config: JoyCaptionConfig,
                        progress_callback) -> JoyCaptionStartResponse
```

**Key design decisions:**

1. **Singleton with model-key caching** — if the same `(gguf_model, mmproj_file,
   n_gpu_layers, n_ctx)` is requested, reuse the loaded model
2. **Async lock around generation** — llama.cpp isn't thread-safe for concurrent
   vision calls; serialize access
3. **Progress reporting** — iterate images, call `job_manager.update_progress()`
   after each caption (unlike subprocess where we parse stdout)
4. **Error handling** — failed images log warning + continue; don't fail the
   whole batch

### Batch Loop Pseudocode

```python
async def run_joycaption_batch(config, job_id):
    images = get_image_files(config.dataset_dir)
    await runner.load_model(config)
    try:
        for i, img_path in enumerate(images):
            try:
                caption = await runner.generate(img_path, config)
                write_caption_file(img_path, caption, config.caption_extension)
            except Exception as e:
                logger.warning(f"Failed to caption {img_path}: {e}")
            await update_job_progress(job_id, i+1, len(images))
    finally:
        if config.unload_when_done:
            await runner.unload_model()
```

## API Endpoints

```
POST   /api/dataset/caption/joycaption   → start_joycaptioning(config)
GET    /api/dataset/caption/status/{id}  → get_status(id)
POST   /api/dataset/caption/stop/{id}    → stop_captioning(id)
```

Following the same pattern as existing BLIP/GIT endpoints, except:

- `start_joycaptioning()` runs an **async coroutine job** (not a subprocess)
  via `job_manager.run_coroutine_job()`, same pattern as Batch Crop and
  Convert to Format
- Status/stop reuse existing job infrastructure

### Job Type Addition

Add `JOYCAPTION` to `JobType` enum in `services/models/job.py`:

```python
class JobType(str, Enum):
    TRAINING = "training"
    TAGGING = "tagging"
    DOWNLOAD = "download"
    CONVERSION = "conversion"
    CROP = "crop"
    JOYCAPTION = "joycaption"    # NEW
```

## Frontend

### API Client (`frontend/lib/api.ts`)

```typescript
const captioningAPI = {
  startJoyCaption: (config: JoyCaptionConfig) =>
    api.post("/dataset/caption/joycaption", config),
  getStatus: (jobId: string) =>
    api.get(`/dataset/caption/status/${jobId}`),
  stop: (jobId: string) =>
    api.post(`/dataset/caption/stop/${jobId}`),
}
```

### UI Components

- **Caption type selector** — dropdown from `CAPTION_TYPE_MAP` keys
- **Caption length** — dropdown with "any / very short / short / medium / long /
  very long / 20-260"
- **Extra options** — collapsible section with checkboxes for the 25+ EXTRA_OPTIONS
- **Model selection** — dropdown listing `.gguf` files in `models/llava_gguf/`
  (like a smaller version of the LoRA model picker)
- **MMProj selection** — paired dropdown for the mmproj file

### Page

A new "JoyCaption" tab/option alongside the existing "BLIP Caption" and
"GIT Caption" on the auto-tag page, or a dedicated `joycaption` page under
`frontend/app/dataset/[name]/joycaption/`.

## Model Management

### GGUF Model Directory

```python
MODELS_DIR / "llava_gguf"
```

Models placed here (downloaded manually or via a model downloader):
- `joycaption-beta-*.gguf` (main model)
- `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf` (mmproj)

### Download Sources

| Model | Source |
|-------|--------|
| Main (recommended) | `concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf` |
| Quantized variants | `mradermacher/llama-joycaption-beta-one-hf-llava-GGUF` |
| IQ-quantized | `mradermacher/llama-joycaption-beta-one-hf-llava-i1-GGUF` |

The model download service (`model_service.py`) could be extended to support
these — or manual download is fine for v1.

## Dependency: llama-cpp-python

GPU-accelerated install requires the CUDA extra index URL:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Notes:**
- `cu124` matches the CUDA 12.4 runtime used by the project
- CPU-only fallback is possible (slow but functional) — `pip install llama-cpp-python`
- `installer.py` should detect CUDA and pick the right wheel
- **Do not add** `llama-cpp-python` to `requirements.txt` — the base install
  is CPU-only; GPU path must be explicit

## Migration Path

1. **Phase 1 — Backend runner + API** (this plan)
   - Implement `joycaption_runner.py` and models
   - Add API endpoints
   - Existing BLIP/GIT stay untouched
2. **Phase 2 — Frontend UI**
   - Caption type/length/options selector
   - Model file picker
3. **Phase 3 — Polish**
   - Progress tracking via job logs/WebSocket
   - Model download from HuggingFace via existing downloader
   - Keep-model-warm between pages

## Open Questions

- Should JoyCaption be a *replacement* for BLIP/GIT or a *supplement*?
  (Answer determines if we remove old UI or add alongside)
- How does the extra-options UI translate from 25+ booleans to a usable
  frontend form? (Collapsible groups? Presets?)
- Is the file-list scanning (listing `.gguf` files in `llava_gguf/`) done by
  the backend at endpoint time or by the frontend via the files API?
- Should the coroutine job support cancellation mid-batch? (The current
  `run_coroutine_job` pattern in crop/convert handles this with a cancellation
  flag — replicate that.)
