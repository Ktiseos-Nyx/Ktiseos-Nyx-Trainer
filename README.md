# Ktiseos Nyx LoRA Trainer

Web-based LoRA training interface built on Kohya SS. Runs locally on Windows/Linux or on cloud GPU instances (VastAI, RunPod).

| Python | License | Deploy | Discord | Twitch | Support | Quality |
|---|---|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Deploy on VastAI](https://img.shields.io/badge/Deploy-VastAI-FF6B6B?style=for-the-badge&logo=nvidia)](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-NextJS-Trainer) [![Deploy on RunPod](https://img.shields.io/badge/Deploy-RunPod-673AB7?style=for-the-badge&logo=runpod)](https://console.runpod.io/deploy?template=2kkfdbmlcc&ref=yx1lcptf) | [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) | <a href="https://ko-fi.com/duskfallcrew"><img src="https://img.shields.io/badge/Ko--Fi-Support-FF5E5B?style=for-the-badge&logo=kofi" alt="Ko-fi"></a> | [![DeepScan grade](https://deepscan.io/api/teams/29397/projects/31347/branches/1015145/badge/grade.svg)](https://deepscan.io/dashboard#view=project&tid=29397&pid=31347&bid=1015145) |

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
- [Credits](#credits--acknowledgements)

---

## Status

**Approaching beta — May 2026.** Core training is verified on VastAI, RunPod, and local GPU.

**Confirmed working:** SDXL, Illustrious, Pony, NoobAI, Flux, Anima

**Known gaps:**
- Compass, LPFAdamW, RMSProp optimizers are wired but not yet in the UI dropdown
- Rex and CosineAnnealing schedulers vendored but not exposed in UI
- Preset list needs an organization pass

---

## Requirements

### Hardware

- **GPU:** NVIDIA with CUDA 12.1+ (12 GB VRAM minimum; 24 GB recommended for SDXL/Flux)
- **Disk:** 50 GB+ free space

| GPU | Status |
|-----|--------|
| NVIDIA (CUDA 12.1+) | Fully supported |
| AMD (ROCm) | Community supported — see [PyTorch ROCm docs](https://pytorch.org/get-started/locally/) |
| NVIDIA via ZLUDA | Community supported — see [ZLUDA](https://github.com/vosen/ZLUDA) |
| CPU-only | Not recommended for training; tagging/captioning will work |

### Software

- **Python:** 3.10 or 3.11
- **Node.js:** 20.19+ (22.x recommended)
- **Git**

> Full platform requirements are in [documentation/installation/INSTALLATION.md](documentation/installation/INSTALLATION.md).

---

## Installation

> **Windows:** Install to a path you own (e.g. `C:\Users\YourName\Projects\`). Restricted system directories (`C:\`, `Program Files`, OneDrive folders) will cause permission errors.

**Windows:**
```bat
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
install.bat
```

**Linux:**
```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
./install.sh
```

Both scripts will prompt about creating a virtual environment. Pass `--venv` or `--no-venv` to skip the prompt; `--auto` for fully unattended install.

**Cloud:** Use the VastAI or RunPod deploy buttons above. Both auto-configure on launch.

> New to RunPod? [Sign up here](https://runpod.io?ref=yx1lcptf) for bonus credits.

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
- Civitai browser: search, filter, and download directly into local model folders
- HuggingFace upload after training

**Utilities:**
- LoRA merge, resize, and metadata editing
- Checkpoint merge

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

- [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) — bug reports and feature requests
- [Discord](https://discord.gg/HhBSM9gBY) — community support

When reporting a bug, include error messages, logs (`logs/app_YYYYMMDD.log`), and the output of:
```bat
diagnose.bat      # Windows
./diagnose.sh     # Linux
```

---

## Credits & Acknowledgements

- **[Kohya-ss SD Scripts](https://github.com/kohya-ss/sd-scripts)** — foundational training scripts
- **[Derrian-Distro's Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend)** — core training backend and LyCORIS fork
- **[Jelosus2](https://github.com/Jelosus2/Lora_Easy_Training_Colab)** — original Colab inspiration
- **[HoloStrawberry](https://github.com/holostrawberry)** — training techniques and Colab notebooks
- **[Linaqruf](https://github.com/Linaqruf)** — training methods
- **[LyCORIS Team](https://github.com/67372a/LyCORIS)** — advanced LoRA methods (DoRA, LoKr)
- **[ArcEnCiel](https://arcenciel.io/)** — support, testing, and open source models
- **AndroidXXL, Jelosus2** — accessible LoRA training contributions

---

**License:** MIT — see [LICENSE](LICENSE)
