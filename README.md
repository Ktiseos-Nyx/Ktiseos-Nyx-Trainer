# Ktiseos Nyx LoRA Trainer 🚀

> ⚠️ **BETA - IN ACTIVE DEVELOPMENT**: Features may not work as expected. [Report issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)

A professional LoRA training system built on Kohya SS, offering **two workflows**:
- 🌐 **Modern Web UI** - Next.js frontend with FastAPI backend
- 📓 **Jupyter Notebooks** - Classic widget-based interface

Supports local and cloud deployment on VastAI, RunPod, and similar platforms.

| Python Version | License | Discord | Twitch | Support |
|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Discord](https://img.shields.io/badge/Discord-Join%20Our%20Server-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow%20on%20Twitch-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) |  <a href="https://ko-fi.com/duskfallcrew" target="_blank"><img src="https://img.shields.io/badge/Support%20us%20on-Ko--Fi-FF5E5B?style=for-the-badge&logo=kofi" alt="Support us on Ko-fi"></a> |

## 🌟 Overview & Key Features

**Modern Web UI + Python Backend Architecture:**
- 🌐 **Next.js Frontend** - Modern React-based interface with real-time updates
- ⚡ **FastAPI Backend** - Fast, async Python API with lazy-loaded dependencies
- 🎨 **Tailwind CSS + shadcn/ui** - Beautiful, responsive design system
- 🔄 **Real-time Training Monitor** - Live progress tracking via WebSocket
- 📁 **File Browser** - Manage datasets and models without leaving the UI

**Training Features:**
- 🖼️ **Integrated Dataset Prep** - Upload, tag with WD14, edit captions in-browser
- 🧮 **Training Calculator** - Automatic step/epoch calculations
- 🎯 **132 Training Parameters** - Full control over LoRA training (SDXL, SD1.5, Flux, SD3, Lumina, Chroma)
- 🔧 **Multiple LoRA Variants** - Standard, LoCon, LoHa, LoKr, DoRA support
- 🚀 **Advanced Optimizers** - AdamW8bit, Prodigy, Lion, CAME, and more
- ☁️ **HuggingFace Integration** - Upload datasets and trained LoRAs directly
- 📊 **Async Uploads** - Non-blocking file uploads so your images don't become dreams

**Platform Support:**
- 💻 **Cross-platform** - Windows, Linux, macOS
- 🌩️ **Cloud Ready** - VastAI, RunPod templates available
- 📓 **Legacy Jupyter Support** - Original widget-based notebooks still available

### ⚠️ Note

> **We are STILL in heavy development. New features in theory SHOULD WORK, but are hard to catch.**
>
> This branch includes experimental features that are available in the Kohya backend but may not be fully tested in our setup:
> - 🔬 **FLUX training** - Available in Kohya, integration status unknown
> - 🧬 **SD3/SD3.5 training** - Available in Kohya, integration status unknown
> - 🌟 **Lumina2 training** - Available in Kohya, integration status unknown
> - 🎨 **Chroma training** - Basic support available (may not match [Flow trainer](https://github.com/lodestone-rock/flow) performance)
> - 🔧 **Latest bug fixes** and performance improvements
> - ⚡ **Enhanced upload widgets** (fixed cache issues)
> - ⚡ **Language Cleanup** Cleaned up a lot of marketing speak and started the roadmap to check inconsistencies on missing content.
>
> **Note**: These experimental features exist in the underlying Kohya scripts but haven't been thoroughly tested with our widget system. Use at your own risk and expect possible issues. If they look like they're exposed in our widget setup, there is no saying if they current work due to our unified setup. We're working on fast trying to get functionality quickly. If you have any issues please report them to the issues area.
>
> **Cross-Trainer Compatibility Goal**: We're researching support for additional memory-efficient optimizers and training techniques (inspired by [Flow](https://github.com/lodestone-rock/flow)) to enable training on a wider range of GPU configurations. This is future work - contributions and testing welcome!


## 🚀 Quick Start

### Choose Your Workflow

#### 🌐 Web UI (Recommended for Most Users)
Modern, responsive web interface with real-time monitoring:

**VastAI Deployment** (Easiest):
- Launch the Ktiseos Nyx template on VastAI
- Access web UI at `http://<instance-ip>:3000`
- See [VastAI Setup Guide](VASTAI_SETUP.md)

**Local Installation**:
```bash
# 1. Clone repository
git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

# 2. Install backend
python installer.py

# 3. Start backend
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Start frontend (in another terminal)
cd frontend
npm install
npm run build
npm run start
```

Access at: `http://localhost:3000`

**See**: [Deployment Guide](docs/DEPLOYMENT.md) for detailed instructions

#### 📓 Jupyter Notebooks (Legacy - For Transition)
Widget-based interface with step-by-step control:

> **Note**: Jupyter notebooks are kept in a separate `jupyter-notebooks/` directory for users transitioning from the old workflow. These may be deprecated in future releases as we focus on the modern Web UI. If you prefer the notebook workflow, you can continue using them, or even move them to the project root if desired.

```bash
# 1. Clone and setup
git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
python installer.py

# 2. Start Jupyter
jupyter lab

# 3. Open notebooks
# Navigate to jupyter-notebooks/ directory in Jupyter's file browser
# Or optionally move notebooks to project root:
# cp jupyter-notebooks/*.ipynb .
```

**See**: [Jupyter Notebooks Guide](jupyter-notebooks/README.md)

---

### Requirements

- **GPU**: Nvidia (CUDA 12.1+) or AMD (ROCm)
- **Python**: 3.10 or 3.11
- **Platform**: Windows, Linux, or macOS
- **VRAM**: 12GB minimum, 24GB recommended for SDXL
- **Disk**: 50GB+ free space

More details: [Quick Start Guide](docs/quickstart.md) | [Installation Guide](docs/guides/installation.md)

You will need Git and Python 3.10+.
If you don't have python, you can install Python 3.10+ from Python's [main website here](https://www.python.org/downloads/). Our set up prefers 3.10.6 at a minimum.

**Install Git if needed:**
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **Mac**: `xcode-select --install` in Terminal
- **Linux**: `sudo apt install git` (Ubuntu/Debian)

**Main Installation Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Lora_Easy_Training_Jupyter

# 2. Run the installer (downloads ~10-15GB)
python ./installer.py

# For detailed installation output (recommended for troubleshooting):
python ./installer.py --verbose
# or: python ./installer.py -v

# Alternative for Mac/Linux:
chmod +x ./jupyter.sh && ./jupyter.sh
```

## 📖 Usage Guide

### Web UI Workflow

The web interface provides a streamlined, tab-based workflow:

1. **📁 Files Page** - Browse and upload datasets
   - File browser for managing training data
   - Drag-and-drop upload support
   - Directory creation and organization

2. **🖼️ Dataset Page** - Prepare images for training
   - WD14 auto-tagging with multiple tagger options
   - Caption editor with batch operations
   - Image gallery with filtering and search

3. **🎓 Training Page** - Configure and start training
   - 7 organized tabs: Setup, Dataset, LoRA, Learning Rate, Advanced, Saving, Logging
   - 132 training parameters with tooltips and validation
   - Model-specific settings (Flux, SD3, Lumina fields auto-show)
   - Real-time config validation

4. **📊 Monitor** - Track training progress (coming soon)
   - Live loss graphs and metrics
   - Sample image generation preview
   - Training logs streaming

5. **🎨 Models Page** - Manage trained LoRAs (coming soon)
   - Browse output directory
   - Preview LoRA metadata
   - Quick download and share

6. **🔧 Utilities Page** - Post-training tools (coming soon)
   - LoRA resizing (extract/merge)
   - HuggingFace upload
   - Metadata editing

**Architecture**:
- **Frontend** runs on port 3000 (Next.js)
- **Backend** runs on port 8000 (FastAPI)
- Backend uses **lazy-loaded managers** - AI dependencies only load when needed
- **File browser and config** work without any AI packages installed

**Full Guide**: [Web UI Documentation](docs/WEB_UI_GUIDE.md) (coming soon)

### Jupyter Notebook Workflow

Three specialized notebooks available in `jupyter-notebooks/`:

- **`Unified_LoRA_Trainer.ipynb`** - Complete training pipeline (all-in-one)
- **`Dataset_Preparation.ipynb`** - Dataset management only
- **`Utilities_Notebook.ipynb`** - Post-training tools

**Full Guide**: [Jupyter Notebooks Guide](jupyter-notebooks/README.md)

### VastAI Users

Both workflows work on VastAI:
- **Jupyter**: Access at `http://<instance-ip>:8888` (auto-started)
- **Web UI**: Run `/start_services.sh` via SSH, access at `http://<instance-ip>:3000`

**Full Guide**: [VastAI Jupyter Guide](jupyter-notebooks/VASTAI_JUPYTER.md)

## 🛠️ Troubleshooting & Support

For more help and support please check [Troubleshooting](docs/guides/troubleshooting.md) this has more comprehensive information. If you're a developer, we're working on our testing notebook, there is one in the wings of the /tests folder, but it has older code and may not match what is current running.

### 📋 **Support Requirements**
Before asking for help, please review our [Support Guidelines](docs/guides/troubleshooting.md#support-guidelines--boundaries). We're happy to assist, but effective troubleshooting requires your participation - this means running the basic diagnostic commands and providing complete error information. Cherry-picking troubleshooting steps won't lead to solutions!

**Windows Users:** If you encounter Rust compilation errors during safetensors installation, this is not related to our notebook setup. It's a common Python packaging issue on Windows. Feel free to reach out on our [Discord](https://discord.gg/HhBSM9gBY) for assistance - we're happy to help guide you through the solution!

**Getting Help**:
    - ✅ **Official Support**: [GitHub Issues](https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter/issues) or [Our Discord](https://discord.gg/HhBSM9gBY)
    - ❌ **No Support**: Random discords, Reddit DMs, social media comments, etc.
    - 📚 **Self-Help**: Check our comprehensive [docs/](https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter/tree/main/docs) folder first
    - 🎯 **Submodule Issues**: Feel free to blame us on the original repos (kohya-ss, LyCORIS, etc.)!

## 🙏 Credits & Acknowledgements

- **Built on the Shoulders of Giants**
This project builds upon and integrates the excellent work of:
- **[Jelosus2's LoRA Easy Training Colab](https://github.com/Jelosus2/Lora_Easy_Training_Colab)** - Original Colab notebook that inspired this adaptation
- **[Derrian-Distro's LoRA Easy Training Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend)** - Core training backend and scripts as well as the forked Lycoris Repository and CAME/REX optimization strategies.
- **[HoloStrawberry's Training Methods](https://github.com/holostrawberry)** - Community wisdom and proven training techniques as well as foundational Google Colab notebooks.
- **[Kohya-ss SD Scripts](https://github.com/kohya-ss/sd-scripts)** - Foundational training scripts and infrastructure
- **[Linaqruf](https://github.com/Linaqruf)** - Pioneer in accessible LoRA training, creator of influential Colab notebooks and training methods that inspired much of this work
- **AndroidXXL, Jelosus2** - Additional Colab notebook contributions that made LoRA training accessible
- **[ArcEnCiel](https://arcenciel.io/)** - Ongoing support and testing as well as Open Source AI Generative Models.
- **[Civitai](https://civitai.com/)** - Platform for Open Source AI Content
- **[LyCORIS Team](https://github.com/67372a/LyCORIS)** - Advanced LoRA methods (DoRA, LoKr, etc.)

Special thanks to these creators for making LoRA training accessible to everyone!

---

## 🔒 Security

Found a security issue? Check our [Security Policy](SECURITY.md) for responsible disclosure guidelines.

## 📄 License

MIT License - Feel free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved. Feel free to open issues or submit pull requests on GitHub.

---

Made with ❤️ by the community, for the community.