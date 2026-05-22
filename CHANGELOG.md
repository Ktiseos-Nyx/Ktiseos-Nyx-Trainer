# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — dev branch

### Fixed
- **SDXL NaN loss (root cause)** — `_tokenize_tags` left overflow/padding chunks with all-False CLIP attention masks. `exp(dtype.min)` underflows to 0 in bf16/fp32, producing `0/0 = NaN` in softmax. NaN hidden states propagated through `text_embedding` into the UNet on every run with short prompts. Fixed by setting `ones_mask_chunk` on the three overflow paths in `strategy_base.py`.
- **min_snr_gamma_enabled missing from 4 presets** — Illustrious AdamW8bit, Illustrious CAME Conservative, Pony Prodigy, and pony_template predated the `_enabled` flag convention; their `min_snr_gamma` values were silently ignored after the gate was added.
- **Custom optimizer crash/corruption bugs** — adammini: bare param not wrapped in list; dehaze: NameError on adaptive_muon path; fmarscrop: TypeError on first decoupled stable-weight-decay step; adopt: in-place sqrt corrupted second-moment state; laprop: centered subtraction mutated state through alias.
- **min_snr_gamma injected on every run** — default value of 5.0 caused SNR weighting to be written to every TOML regardless of user intent. Now gated on `min_snr_gamma_enabled`.
- **disable_cross_attn_mask flag not read** — argparse flag existed in the upstream sync but was never connected to the mask guard in the training loop.
- **optimizer_args sent as array instead of string** — caused silent arg-parsing failures for custom optimizers receiving structured kwargs.

### Changed
- README restructured — removed conversational/marketing copy, dev-branch internal notes moved to CHANGELOG, documentation links expanded.

---

## [0.1.x-alpha] — shipped to dev since 2026-04-30 main merge

### Added
- **Tag viewer** — Civitai-inspired chip filter UI on the tag editor page; multi-select tags to filter the image grid; bulk remove/replace from the Actions menu.
- **Stop training button** on the training monitor.
- **WD14 3-way overwrite mode** — overwrite / append / ignore (was boolean append-only).
- **Live training monitor** — step count, loss, learning rate, and ETA parsed from tqdm output.
- **Anima, SDXL, Illustrious, and Pony preset templates** added to `presets/`.
- **NoobAI-XL-v1.1 (EPS)** added to the model browser.
- **Dev-branch provisioning scripts** for VastAI and RunPod.
- **`fetch-restart.sh`** — full pull + npm install + build + service restart in one command.
- **Weight decay auto-injected** into `optimizer_args` for custom optimizers (CAME, Compass, etc.) — top-level `weight_decay` field was silently ignored by the custom optimizer path.

### Fixed
- **CAME NaN (network_train_unet_only)** — defaulting `True` created an empty TE parameter group when `text_encoder_lr > 0`; default changed to `False`.
- **Conservative CAME preset** — was `fp16` + `conv_dim: 32` (both cause NaN with CAME); fixed to `bf16` + `conv_dim: 0`.
- **Log cursor wrap-around** — `deque`-relative index silently stalled after the buffer wrapped; replaced with absolute `total_lines_written` counter. Logs no longer freeze mid-run.
- **Training log polling** — visibility-aware polling with fixed cadence; no more frozen logs on tab switch. `PYTHONUNBUFFERED=1` / `-u` flag ensures stdout streams in real time.
- **Log poller race condition** — logs no longer drop at a mid-run epoch boundary on tab switch.
- **ZIP dataset upload** — request body now piped directly instead of buffered; fixes ECONNRESET on uploads over ~400 MB.
- **Dataset `tags_present` flag** — tag editor page now correctly shows "Has tags" / "No tags" status.
- **LoggingCard Radix fix** — `log_with` SelectItem uses `'none'` sentinel instead of empty string (Radix rejects empty string values).
- **TOML generation path** — config written directly to `config/`; removed intermediate `runtime_store/` copy step.
- **`enable_bucket`** now respects user config (was hardcoded `True`).
- **`network_train_unet_only`** no longer passed in checkpoint training mode.
- **HF upload form** — token pre-filled from saved settings; owner and repo type persist across navigation.
- **LoRA resize** — removed alpha input field that was silently ignored by the backend (SVD auto-calculates it).
- **Merge/resize subprocesses** now have timeouts (resize: 30 min, merges: 1 hr).
- **CUDA availability check** before passing `--device cuda` to merge/resize subprocesses.
- **CheckpointTrainingConfig** TypeScript type now includes SD3.5, Chroma, and Anima.
- **AbortError console noise** on page navigation suppressed in both `pollLogs` paths.
- **Anima presets** — `clip_skip: 1`, `keep_tokens: 1` applied across all Anima configs.

### Changed
- **shadcn CLI** 3.6.1 → 4.7.0.
- **Next.js** 15 → 16 (PR #355, security bump).
- PostCSS CVE-2026-41305 patched.
- fast-uri CVE-2026-6322 patched.

---

## [0.1.0-alpha] — 2026-04-30 initial alpha release

Initial alpha. Core LoRA training pipeline working end-to-end on VastAI and local NVIDIA GPU.

### Features at launch
- Web UI (Next.js 15 + FastAPI) for LoRA training on SDXL, SD1.5, Flux, SD3/SD3.5, Lumina
- Dataset upload, WD14/BLIP/GIT auto-tagging, caption editor
- Training configuration with preset system
- Civitai model browser and HuggingFace upload
- LoRA merge, resize, and checkpoint merge utilities
- VastAI one-click deploy
