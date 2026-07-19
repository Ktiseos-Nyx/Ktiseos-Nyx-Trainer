# Ecosystem

---

| Python | License | Deploy | Discord | Quality |
|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Docker Image](https://img.shields.io/badge/Docker-VastAI%20PyTorch-FF6B6B?style=for-the-badge&logo=docker)](https://hub.docker.com/r/vastai/pytorch/) [![Deploy on RunPod](https://img.shields.io/badge/Deploy-RunPod-673AB7?style=for-the-badge&logo=runpod)](https://console.runpod.io/deploy?template=2kkfdbmlcc&ref=yx1lcptf) | [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![DeepScan grade](https://deepscan.io/api/teams/29397/projects/31347/branches/1015145/badge/grade.svg)](https://deepscan.io/dashboard#view=project&tid=29397&pid=31347&bid=1015145) |

---

**Train, tag, and generate — a full LoRA & checkpoint training suite with ComfyUI built in.**

A web interface built on Kohya SS. Prepare datasets, auto-tag, train across many model types (SDXL, Flux, SD3, Lumina, Anima…), then generate and test results in a bundled ComfyUI workspace.  

---

## Table of Contents

- [Status](#status)
- [Requirements](#requirements)
- [Installation](#installation)
- [Starting the App](#starting-the-app)
- [Updating](#updating)
- [Features](#features)
- [Testing](#testing)
- [Documentation](#documentation)
- [Support](#support)



---

## Requirements

### Hardware

- **GPU:** NVIDIA with CUDA 12.1+ 
- **Disk:** 50 GB+ free space

| GPU | Status |
|-----|--------|
| NVIDIA (CUDA 12.1+) | Fully supported |
| AMD (ROCm) | Community supported — see [AMD ROCm docs](https://github.com/ROCm/rocm) |
| NVIDIA via ZLUDA | Community supported — see [ZLUDA](https://github.com/vosen/ZLUDA) |
| CPU-only | Not recommended for training; tagging/captioning will work |

While baseline VRAM requirements can be high, the community has figured out plenty of workarounds to train on constrained hardware.

### Software

- **Python:** 3.10, 3.11, or 3.12
- **Node.js:** 20.19+ (22.x recommended)
- **Git**

> Full platform requirements are in [documentation/installation/INSTALLATION.md](documentation/installation/INSTALLATION.md).

---

## Installation

> **Windows Users:** Install to a path you own (e.g. `C:\Users\YourName\Projects\`). Restricted system directories (`C:\`, `Program Files`, OneDrive folders) will cause permission errors.

**Windows:**
```bat
git clone https://github.com/UselessToys/Ecosystem_WebUI.git
cd Ecosystem_WebUI
install.bat
```

**Linux:**
```bash
git clone https://github.com/UselessToys/Ecosystem_WebUI.git
cd Ecosystem_WebUI
./install.sh
```

Both scripts will prompt about creating a virtual environment. Pass `--venv` or `--no-venv` to skip the prompt; `--auto` for fully unattended install.

**ComfyUI** (image generation) installs by default — including on cloud. To skip it, pass `--no-comfyui` (e.g. `install.bat --no-comfyui`, `./install.sh --no-comfyui`, or `python installer.py --no-comfyui` on a remote instance).

**Cloud:** Use the VastAI or RunPod deploy buttons. Both auto-configure on launch.

 [![Docker Image](https://img.shields.io/badge/Docker-VastAI%20PyTorch-FF6B6B?style=for-the-badge&logo=docker)](https://hub.docker.com/r/vastai/pytorch/) 
  
 [![Deploy on RunPod](https://img.shields.io/badge/Deploy-RunPod-673AB7?style=for-the-badge&logo=runpod)](https://console.runpod.io/deploy?template=2kkfdbmlcc&ref=yx1lcptf) 

---

## Starting the App

```bash
# Default: frontend on 3000, backend on 8000
start_services_local.bat    # Windows
./start_services_local.sh   # Linux

# Custom ports
start_services_local.bat --port 4000 --backend-port 9000
./start_services_local.sh  --port 4000 --backend-port 9000

# Quick restart without reinstall
restart.bat     # Windows
./restart.sh    # Linux

# Restart with Reinstall + Restart supervisory process (Remote only)

bash fetch-restart.sh
./fetch-restart.sh 

```

**URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Updating

```bash
git pull
install.bat       # Windows
./install.sh      # Linux
```

**If reinstalling doesn't fix the problem:**
```bash
# Preview what will be removed (dry run)
python clean_slate.py --dry-run

# Remove build artifacts, venvs, node_modules, caches (preserves models/datasets/outputs)
python clean_slate.py

# Also remove models, VAEs, datasets, and outputs
python clean_slate.py --nuclear
```

---

## Features

**Dataset preparation:**
- Drag-and-drop upload (individual files, zip archives, URLs)
- WD14, BLIP, and GIT auto-tagging
- Tag editor with bulk add / remove / replace, frequency viewer, and image gallery

**Training:**
- Supported models: SDXL, SD1.5, Flux, SD3/SD3.5, Lumina, Anima, HunyuanImage
- LoRA types: Standard, LoCon, LoHa, LoKr, DoRA
- 40+ community presets (AdamW8bit, CAME, Prodigy, AdaFactor, Compass, Lion8bit, and more)
- Live training monitor: step count, loss, learning rate, ETA
- Preset system with form persistence

**Models:**
- Civitai browser: search, filter, and download models directly into your local folders
- Arc En Ciel browser: browse and download from multiple model sources with unified search
- HuggingFace downloads for training base models
- LoRA Manager for batch downloading from Civitai and HuggingFace via Python or Aria2

**Image generation (ComfyUI):**
- Bundled ComfyUI workspace, installed by default (local and cloud) — generate images to test your trained models without leaving the app
- LoRA Manager for browsing and applying your LoRAs

**Utilities:**
- LoRA merge, resize, and metadata editing
- 24+ checkpoint merge modes via Chattiori (WS, DARE, cosine blending, ReBasin, and more)
- LoRA-to-checkpoint baking (multi-LoRA bake across all architectures)
- HuggingFace upload after training

---

## Testing

Tests validate the API and service layer without requiring a GPU, PyTorch, or a training run. The ML stack is mocked at the subprocess boundary.

```bash
# Install dev dependencies
pip install -r requirements_dev.txt

# Run all tests (~5 seconds, no GPU needed)
pytest tests/test_api_plumbing.py -v

# Skip tests that require real hardware
pytest tests/ -m "not slow"
```

| Area | Coverage |
|------|----------|
| Path validation | Drive-letter casing; out-of-bounds paths rejected |
| Training endpoint | Validation failures return `success=False`; malformed payloads get 422 |
| Subprocess launch | `-u` flag and `PYTHONUNBUFFERED=1` present for log streaming |
| Config paths | TOML config dir anchored to project root |
| Job failure logging | Full traceback written to `app.log` on crash |
| Log buffer | Subprocess stdout lines reach the buffer the UI polls |

New tests go in `tests/`. Tests requiring a real GPU: mark with `@pytest.mark.slow`.

---

## Documentation

- [Installation Guide](documentation/installation/INSTALLATION.md) — full platform setup, manual installation, troubleshooting
- [Beta-Planning](documentation/planning/BETA_PLANNING.md) — Bug Tracking and Beta Planning.
- [Integration Strategy](documentation/planning/INTEGRATION_STRATEGY.md) — Workflow Planning with Comfyui in mind. 
- [Attribution](ATTRIBUTIONS.md) - Full Credits and Licenses.
- [General Guides](documentation/guides/general/README.md) — usage documentation
- [Training: General](documentation/guides/training/general/train_network.md)
- [Training: SDXL](documentation/guides/training/SDXL/sdxl_train_network.md)
- [Training: Flux](documentation/guides/training/flux/flux_train_network.md)
- [Training: SD3](documentation/guides/training/SD3/sd3_train_network.md)
- [Training: Anima](documentation/guides/training/anima/anima_train_network.md)
- [LyCORIS Algorithms](documentation/guides/lycoris/Algo-Details.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

---

## Support

- [GitHub Issues](https://github.com/UselessToys/Ecosystem_WebUI/issues) — bug reports and feature requests
- [Discord](https://discord.gg/HhBSM9gBY) — community support

When reporting a bug, include error messages, logs (`logs/app_YYYYMMDD.log`), and the output of:
```bat
diagnose.bat      # Windows
./diagnose.sh     # Linux
```


---

**License:** MIT — see [LICENSE](LICENSE)
