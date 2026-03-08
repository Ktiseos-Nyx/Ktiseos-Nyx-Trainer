# Ktiseos Nyx LoRA Trainer

LoRA training system built on Kohya SS with a modern web UI (Next.js + FastAPI). Train LoRA models locally or on cloud GPUs with an intuitive interface.

| Python | License | Deploy | Discord | Twitch | Support |
|---|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Deploy on VastAI](https://img.shields.io/badge/Deploy-VastAI-FF6B6B?style=for-the-badge&logo=nvidia)](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-NextJS-Trainer) [![Deploy on RunPod](https://img.shields.io/badge/Deploy-RunPod-673AB7?style=for-the-badge&logo=runpod)](https://console.runpod.io/deploy?template=2kkfdbmlcc&ref=yx1lcptf) | [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) | <a href="https://ko-fi.com/duskfallcrew"><img src="https://img.shields.io/badge/Ko--Fi-Support-FF5E5B?style=for-the-badge&logo=kofi" alt="Ko-fi"></a> |

---

## 🚧 **ALPHA - March 2026**

**Active development.** The web UI (Next.js + FastAPI) is functional but expect rough edges.

Currently dealing with minor broken features until Claude resets in the next ocuple hours, hoping code rabbit can help me!

**Current status:**
- ✅ **Web UI** - Dataset management, tagging, captioning, file browser all working
- ✅ **Training config** - 132+ parameters across 7 tabs, preset system, form persistence
- ✅ **Training execution** - End-to-end training verified on VastAI and RunPod (CAME optimizer, SDXL LoRA)
- ✅ **Custom optimizers** - CAME working, Compass/standard optimizers (AdamW8bit, Prodigy, etc.) supported
- ✅ **HuggingFace upload** - Direct upload from the web UI after training
- 🐛 **Report issues** - [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) or [Discord](https://discord.gg/HhBSM9gBY)

> ⚠️ **ALPHA STAGE**: Core training works! Still polishing edges and testing more configurations. [Report issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) • [Development Status](STATUS.md)

## Quick Start

### Requirements

- **GPU**: NVIDIA with CUDA 12.1+ (12GB VRAM minimum, 24GB for SDXL)
- **Python**: 3.10 or 3.11
- **Node.js**: 18+
- **Disk**: 50GB+ free space

📖 [Full Requirements & Installation Guide](documentation/INSTALLATION.md)

#### 🚨 WHERE YOU INSTALL MATTERS — READ THIS

> **DO NOT install this project in any of these locations:**
> - `C:\` (root of any drive)
> - `C:\Program Files\` or `C:\Program Files (x86)\`
> - `C:\Windows\` or any system directory
> - Network drives, OneDrive, Dropbox, or Google Drive folders
>
> **These locations have restricted permissions on modern Windows and WILL cause "Access Denied" errors**, even when running as Administrator.
>
> **Install to a folder YOU own**, like:
> ```
> C:\Users\YourName\Projects\Ktiseos-Nyx-Trainer
> C:\Users\YourName\Desktop\Ktiseos-Nyx-Trainer
> D:\Projects\Ktiseos-Nyx-Trainer
> ```
>
> If you're thinking "but I always install stuff to `C:\`" — consider that this project manages files, runs scripts, and creates virtual environments. You really don't want a rogue process with write access to your system root. Your user folder exists for a reason!

#### 🛡️ STRONGLY RECOMMENDED: Use a Virtual Environment

**Quick venv setup:**

**Windows:**
```bat
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

REM Create virtual environment
python -m venv venv

REM Activate it (you'll do this EVERY TIME you open a new terminal)
venv\Scripts\activate

REM Now install - packages stay in the venv
install.bat
```

**Linux:**
```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

# Create virtual environment
python3 -m venv venv

# Activate it (you'll do this EVERY TIME you open a new terminal)
source venv/bin/activate

# Now install - packages stay in the venv
python installer_local_linux.py
```

**Starting the app with venv:**
```bash
# Activate venv first
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux

# Then start services
start_services_local.bat  # Windows
./start_services_local.sh # Linux
```

> 💡 **Pro tip:** Your terminal prompt will show `(venv)` when the virtual environment is active. If you don't see it, activate again!

---

### Installation (Without venv - not recommended)

**Windows:**
```bat
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
install.bat
start_services_local.bat
```

**Linux:**
```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
python installer_local_linux.py
./start_services_local.sh
```

**Access URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Cloud Deployment:** Use the VastAI or RunPod deploy buttons above - auto-configures on launch. Training has been verified end-to-end on both platforms.

> 💰 **New to RunPod?** [Sign up with this link](https://runpod.io?ref=yx1lcptf) for bonus credits to get started.

### Updating to Latest Version

**DO NOT delete the folder and re-clone.** This creates orphaned cache files and Python environment shims that cause installation issues.

**Proper update process:**

```bash
# Navigate to your installation directory
cd Ktiseos-Nyx-Trainer

# Get latest changes
git pull

# If you have a virtual environment, reactivate it
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux

# Update dependencies
install.bat             # Windows
./install.sh            # Linux
```

**Why not delete and re-clone?**
- Leaves behind hidden cache directories (`__pycache__`, `.venv`)
- Creates orphaned Python shims in system directories
- Forces re-download of all dependencies
- May cause version conflicts with cached packages

**Having issues?** Just re-run the installer - it's safe to run multiple times and will preserve your datasets/models.

### Clean Reinstall

If things are truly broken and you need a fresh start, use the clean slate script instead of deleting and re-cloning:

```bash
# Preview what will be deleted (nothing is removed)
python clean_slate.py --dry-run

# Delete build artifacts, venvs, node_modules, caches
python clean_slate.py

# Skip confirmation prompt
python clean_slate.py --yes

# Nuclear option: also delete models, VAEs, datasets, outputs
python clean_slate.py --nuclear
```

This removes build artifacts and dependencies while **preserving your models, datasets, and outputs** by default. After running it, just re-run the installer:

```bat
REM Windows
install.bat

REM Linux
./install.sh
```

**When to use this vs. `git pull`:**
- `git pull` + re-run installer = normal updates (do this first)
- `clean_slate.py` = something is broken and reinstalling didn't fix it
- `clean_slate.py --nuclear` = scorched earth, start completely over

## What It Does

**Dataset Preparation:**
- Upload datasets via drag-and-drop interface
- Auto-tag images with WD14 models
- Edit captions with batch operations
- Image gallery with filtering

**Training Configuration:**
- 132+ parameters across 7 organized tabs
- Support for SDXL, SD1.5, Flux, SD3/SD3.5, Lumina
- Multiple LoRA types: Standard, LoCon, LoHa, LoKr, DoRA
- Real-time validation and progress monitoring

**Utilities:**
- LoRA resizing and metadata editing
- HuggingFace uploads
- Model downloads from Civitai/HuggingFace

📖 [Full Feature List](documentation/FEATURES.md) • [In-App Documentation](http://localhost:3000/docs) (when running)

## Documentation

- 📘 [Installation Guide](documentation/INSTALLATION.md) - Detailed setup for all platforms
- 🚀 [Deployment Guide](documentation/DEPLOYMENT.md) - VastAI and RunPod deployment
- 🔧 [Troubleshooting](documentation/guides/troubleshooting.md) - Common issues and solutions
- 💻 [Development Setup](documentation/DEVELOPMENT_ENVIRONMENTS.md) - Contributing and local dev
- ✨ [Features](documentation/FEATURES.md) - Complete feature documentation
- 📊 [Development Status](STATUS.md) - Current progress and roadmap

## Support

**Official Channels:**
- [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) - Bug reports and feature requests
- [Discord Server](https://discord.gg/HhBSM9gBY) - Community support and discussion

**Before Requesting Help:**
- Check [Troubleshooting Guide](documentation/guides/troubleshooting.md)
- Review [Support Guidelines](documentation/guides/troubleshooting.md#support-guidelines--boundaries)
- Include error messages, logs, and system info

**Installation Issues:**

If installation fails, run the diagnostic tool to collect system information:

```bat
# Windows
diagnose.bat

# Linux/WSL
./diagnose.sh
```

Attach the generated `diagnostics_*.txt` file to your issue report.

## Credits & Acknowledgements

Built on the shoulders of giants:

- **[Kohya-ss SD Scripts](https://github.com/kohya-ss/sd-scripts)** - Foundational training scripts
- **[Derrian-Distro's Backend](https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend)** - Core training backend, LyCORIS fork
- **[Jelosus2's LoRA Easy Training Colab](https://github.com/Jelosus2/Lora_Easy_Training_Colab)** - Original Colab inspiration
- **[HoloStrawberry](https://github.com/holostrawberry)** - Training techniques and Colab notebooks
- **[Linaqruf](https://github.com/Linaqruf)** - Influential training methods
- **[LyCORIS Team](https://github.com/67372a/LyCORIS)** - Advanced LoRA methods (DoRA, LoKr)
- **[ArcEnCiel](https://arcenciel.io/)** - Support, testing, and open source models
- **AndroidXXL, Jelosus2** - Accessible LoRA training contributions

## License & Contributing

- **License**: MIT - See [LICENSE](LICENSE)
- **Contributing**: See [Contributing Guide](CONTRIBUTING.md)
- **Security**: See [Security Policy](SECURITY.md)

---

**Made with ❤️ by the Ktiseos-Nyx Team**
