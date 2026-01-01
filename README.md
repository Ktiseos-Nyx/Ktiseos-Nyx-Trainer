# Ktiseos Nyx LoRA Trainer

LoRA training system built on Kohya SS with a web UI (Next.js + FastAPI). Supports local and cloud deployment on VastAI, RunPod, and similar platforms.

| Python | License | Deploy | Discord | Twitch | Support |
|---|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Deploy on VastAI](https://img.shields.io/badge/Deploy-VastAI-FF6B6B?style=for-the-badge&logo=nvidia)](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-NextJS-Trainer) | [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) | <a href="https://ko-fi.com/duskfallcrew"><img src="https://img.shields.io/badge/Ko--Fi-Support-FF5E5B?style=for-the-badge&logo=kofi" alt="Ko-fi"></a> |

> ⚠️ **ALPHA STAGES - IN ACTIVE DEVELOPMENT**: Features may not work as expected. [Report issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
> ⚠️ ***PLEASE NOTE:***
THE DATASET UPLOADER ON REMOTE CONTAINERS (VAST AI) IS NOT ENTIRELY WORKING - PLEASE USE VASTAI'S INBUILT JUPYTER UNTIL A SOLUTION IS IN.

## Table of Contents

- [Ktiseos Nyx LoRA Trainer](#ktiseos-nyx-lora-trainer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [OS Support Overview](#os-support-overview)
    - [Requirements](#requirements)
    - [Local Installation](#local-installation)
      - [Choose your platform](#choose-your-platform)
    - [VastAI Deployment](#vastai-deployment)
    - [RunPod Deployment](#runpod-deployment)
  - [Overview](#overview)
    - [Architecture](#architecture)
    - [Core Features](#core-features)
    - [Development Status](#development-status)
  - [Troubleshooting \& Support](#troubleshooting--support)
    - [Support Requirements](#support-requirements)
    - [Getting Help](#getting-help)
  - [Credits \& Acknowledgements](#credits--acknowledgements)
  - [Security](#security)
  - [License](#license)
  - [Contributing](#contributing)

## Installation

### OS Support Overview

| Platform        | Training Supported | Installer Script              | Start Script                 | Notes                                  |
|-----------------|--------------------|-------------------------------|------------------------------|----------------------------------------|
| **Windows**     | ✅ (CUDA 11.8)     | `install.bat`                 | `start_services_local.bat`   | Requires NVIDIA GPU; uses `.dll` files |
| **Linux (NVIDIA)** | ✅ (CUDA 12.1)  | `installer_local_linux.py`    | `start_services_local.sh`    | Full GPU training; matches Vast.ai     |
| **macOS**       | ❌ (CPU-only UI)   | Manual `pip install`          | `start_services_local.sh`    | No training (Kohya SS lacks MPS support) |
| **Vast.ai**     | ✅ (CUDA 12.1)     | Auto via `vastai_setup.sh`    | Auto via Supervisor          | Uses `installer_remote.py` internally  |
| **RunPod**      | ⚠️ (Untested)      | Use Linux local scripts       | Use Linux local scripts      | May work; not officially supported yet |

### Requirements

- **GPU**: NVIDIA (CUDA 12.1+) **required for training**
  *(AMD ROCm and Apple Silicon are not currently supported for training due to Kohya SS limitations)*
- **Python**: 3.10 or 3.11
- **Node.js**: 18+ (for web UI)
- **Platform**: Windows, Linux, or macOS
- **VRAM**: 12GB minimum, 24GB recommended for SDXL
- **Disk**: 50GB+ free space
- **RAM**: 8GB+ (16GB recommended for Windows dev environments - see [Development Environments](docs/DEVELOPMENT_ENVIRONMENTS.md))

Install prerequisites if needed:

- **Python**: Download from [python.org](https://www.python.org/downloads/) (3.10.6+ recommended)
- **Git**:
  - Windows: [git-scm.com](https://git-scm.com/download/win)
  - Mac: `xcode-select --install`
  - Linux: `sudo apt install git`
- **Node.js**: Download from [nodejs.org](https://nodejs.org/) (18+)

### Local Installation

```bash
# 1. Clone repository
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
```

#### Choose your platform

- **Windows**:

  ```bat
  install.bat
  start_services_local.bat
  ```

- **Linux (with NVIDIA GPU)**:

  ```bash
  python installer_local_linux.py
  ./start_services_local.sh
  ```

- **macOS or CPU-only Linux**:
  > ⚠️ **Training is not supported** (Kohya SS requires CUDA).
  > You can still run the **web UI + API** for development:

  ```bash
  pip install -r requirements.txt
  ./start_services_local.sh
  ```

> 💡 **Tip**: Use `--verbose` for detailed logs:
>
> ```bash
> python installer_local_linux.py --verbose
> ```

**Access URLs:**

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000>
- API Docs: <http://localhost:8000/docs>

**Quick Restart** (skip reinstall):

```bash
# Linux/Mac:
./restart.sh

# Windows:
restart.bat
```

**Manual Service Startup** (alternative):

```bash
# Terminal 1 - Backend
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run build
npm start
```

See [Deployment Guide](docs/DEPLOYMENT.md) for detailed instructions.

### VastAI Deployment

Use the template via the deploy button in the badge table above.
The system auto-runs `vastai_setup.sh` and configures supervisor for auto-restart.

### RunPod Deployment

RunPod deployment instructions are not yet available. If RunPod doesn't have a portal system like VastAI, the local installation scripts may work, but this is untested.

---

## Overview

### Architecture

- **Frontend**: Next.js 14 with React, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI with lazy-loaded manager system (dependencies load on-demand)
- **Training**: Kohya SS (sd-scripts) with LyCORIS integration
- **File Management**: Built-in browser with upload support (Uppy)

### Core Features

**Dataset Preparation:**

- Upload and extract datasets via web UI
- WD14 auto-tagging (supports v3 taggers with ONNX)
- Caption editor with batch operations
- Image gallery with filtering

**Training Configuration:**

- 132 training parameters across 7 organized tabs
- Support for SDXL, SD1.5, Flux, SD3/SD3.5, Lumina, Chroma
- Multiple LoRA types: Standard, LoCon, LoHa, LoKr, DoRA
- Advanced optimizers: AdamW8bit, Prodigy, Lion, CAME, REX
- Real-time validation with Zod schemas
- Persistent state with Zustand

**Training Execution:**

- TOML-based configuration generation
- Real-time progress monitoring via WebSocket
- Training calculator for automatic step/epoch calculations

**Utilities:**

- LoRA resizing and extraction
- HuggingFace dataset and model upload
- Metadata editing

**Platform Support:**

- Cross-platform: Windows, Linux, macOS
- Cloud-ready: VastAI, RunPod templates

### Development Status

> **ALPHA STAGES**: Active development. Features may not work as expected.
>
> **Experimental Features** (available in Kohya backend, testing status varies):
>
> - Flux training
> - SD3/SD3.5 training
> - Lumina2 training
> - Chroma training (basic support)
> - Some features may not work with the UI once it runs, please check SD-scripts for documentation.
>
> These features use the underlying Kohya scripts but haven't been thoroughly tested in this setup. Report issues on GitHub.

---

## Troubleshooting & Support

See [Troubleshooting Guide](docs/guides/troubleshooting.md) for comprehensive help. You can in turn, also use the frontend WebUI documentation system once you have it up and running.

### Support Requirements

Before requesting help, review the [Support Guidelines](docs/guides/troubleshooting.md#support-guidelines--boundaries). Effective troubleshooting requires running diagnostic commands and providing complete error information.

### Getting Help

- **Official Support**: [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) or [Discord](https://discord.gg/HhBSM9gBY)
- **Documentation**: Check [docs/](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/tree/main/docs) first
- **Not Supported**: Random Discord servers, Reddit DMs, social media comments

For issues with vendored dependencies (Kohya SS, LyCORIS), report them here - we maintain the vendored versions.

## Credits & Acknowledgements

This project builds upon the work of:

- **[Jelosus2's LoRA Easy Training Colab](https://github.com/Jelosus2/Lora_Easy_Training_Colab)** - Original Colab notebook inspiration
- **[Derrian-Distro's LoRA Easy Training Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend)** - Core training backend, forked LyCORIS, CAME/REX optimizers
- **[HoloStrawberry](https://github.com/holostrawberry)** - Training techniques and foundational Colab notebooks
- **[Kohya-ss SD Scripts](https://github.com/kohya-ss/sd-scripts)** - Foundational training scripts
- **[Linaqruf](https://github.com/Linaqruf)** - Influential Colab notebooks and training methods
- **AndroidXXL, Jelosus2** - Colab contributions for accessible LoRA training
- **[ArcEnCiel](https://arcenciel.io/)** - Support, testing, and open source AI models
- **[Civitai](https://civitai.com/)** - Platform for AI content
- **[sd-webui-civbrowser](https://github.com/SignalFlagZ/sd-webui-civbrowser)** - Civitai API integration patterns
- **[LyCORIS Team](https://github.com/67372a/LyCORIS)** - Advanced LoRA methods (DoRA, LoKr, etc.)

---

## Security

Found a security issue? See [Security Policy](SECURITY.md) for responsible disclosure guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! See [Contributing Guide](CONTRIBUTING.md) for details.
