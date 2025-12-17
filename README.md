# Ktiseos Nyx LoRA Trainer

> ⚠️ **BETA - IN ACTIVE DEVELOPMENT**: Features may not work as expected. [Report issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)

LoRA training system built on Kohya SS with a web UI (Next.js + FastAPI). Supports local and cloud deployment on VastAI, RunPod, and similar platforms.

| Python | License | Deploy | Discord | Twitch | Support |
|---|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Deploy on VastAI](https://img.shields.io/badge/Deploy-VastAI-FF6B6B?style=for-the-badge&logo=nvidia)](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-NextJS-Trainer) | [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) | <a href="https://ko-fi.com/duskfallcrew"><img src="https://img.shields.io/badge/Ko--Fi-Support-FF5E5B?style=for-the-badge&logo=kofi" alt="Ko-fi"></a> |

## Table of Contents

- [Installation](#installation)
  - [Local Installation](#local-installation)
  - [VastAI Deployment](#vastai-deployment)
  - [Requirements](#requirements)
- [Overview](#overview)
- [Usage](#usage)
  - [Web UI Workflow](#web-ui-workflow)
  - [VastAI Users](#vastai-users)
- [Troubleshooting](#troubleshooting--support)
- [Credits](#credits--acknowledgements)
- [Security](#security)
- [License](#license)
- [Contributing](#contributing)

## Installation

### Requirements

- **GPU**: Nvidia (CUDA 12.1+) or AMD (ROCm)
- **Python**: 3.10 or 3.11
- **Node.js**: 18+ (for web UI)
- **Platform**: Windows, Linux, or macOS
- **VRAM**: 12GB minimum, 24GB recommended for SDXL
- **Disk**: 50GB+ free space

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

# 2. Run installer (downloads ~10-15GB of dependencies)
python installer.py

# Optional: Verbose output for troubleshooting
python installer.py --verbose

# 3. Start services
# Linux/Mac:
./start_services_local.sh

# Windows:
start_services_local.bat

# 4. Quick restart (skips reinstall - much faster)
./restart.sh  # or restart.bat on Windows
```

**Access URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

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

Click the deploy button in the table above or use this template link:
[Ktiseos Nyx Template](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-NextJS-Trainer)

**Setup:**
- Automated provisioning via `vastai_setup.sh`
- Services auto-start on boot (managed by supervisor)
- Node.js auto-detected and configured

**Access URLs:**
- Frontend: `https://[instance-id].instances.vast.ai:13000`
- Backend: `https://[instance-id].instances.vast.ai:18000`
- Jupyter: `https://[instance-id].instances.vast.ai:18080`

See [VASTAI_TEMPLATE.txt](./VASTAI_TEMPLATE.txt) for manual setup instructions.

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

> **BETA**: Active development. Features may not work as expected.
>
> **Experimental Features** (available in Kohya backend, testing status varies):
> - Flux training
> - SD3/SD3.5 training
> - Lumina2 training
> - Chroma training (basic support)
>
> These features use the underlying Kohya scripts but haven't been thoroughly tested in this setup. Report issues on GitHub.

---

## Usage

### Web UI Workflow

The web interface provides a tab-based workflow:

1. **Files** - Browse and upload datasets
   - File browser for managing training data
   - Drag-and-drop upload with auto-retry and progress tracking
   - Batch uploads (10 files at a time)
   - Directory creation and organization

2. **Dataset** - Prepare images for training
   - WD14 auto-tagging with multiple tagger options
   - Caption editor with batch operations
   - Image gallery with filtering and search

3. **Training** - Configure and start training
   - 7 tabs: Setup, Dataset, LoRA, Learning Rate, Advanced, Saving, Logging
   - 132 training parameters with tooltips and validation
   - Model-specific fields (Flux, SD3, Lumina auto-show based on selection)
   - Real-time config validation

4. **Monitor** - Track training progress *(coming soon)*
   - Live loss graphs and metrics
   - Sample image generation preview
   - Training logs streaming

5. **Models** - Manage trained LoRAs *(coming soon)*
   - Browse output directory
   - Preview LoRA metadata
   - Download and share

6. **Utilities** - Post-training tools *(coming soon)*
   - LoRA resizing (extract/merge)
   - HuggingFace upload
   - Metadata editing

**Service Architecture:**
- Frontend: Port 3000 (Next.js)
- Backend: Port 8000 (FastAPI)
- Lazy-loaded managers: AI dependencies only load when needed
- File browser and config work without AI packages installed

See [Web UI Documentation](docs/WEB_UI_GUIDE.md) for detailed guide *(coming soon)*.

### VastAI Usage

Access the web UI at `http://<instance-ip>:3000` after services auto-start.

See [VastAI Setup Guide](VASTAI_SETUP.md) for detailed instructions.

## Troubleshooting & Support

See [Troubleshooting Guide](docs/guides/troubleshooting.md) for comprehensive help.

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