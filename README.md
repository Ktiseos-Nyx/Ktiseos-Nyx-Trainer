# Ktiseos Nyx LoRA Trainer

LoRA training system built on Kohya SS with a modern web UI (Next.js + FastAPI). Train LoRA models locally or on cloud GPUs with an intuitive interface.

| Python | License | Deploy | Discord | Twitch | Support | Quality |
|---|---|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Deploy on VastAI](https://img.shields.io/badge/Deploy-VastAI-FF6B6B?style=for-the-badge&logo=nvidia)](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-NextJS-Trainer) [![Deploy on RunPod](https://img.shields.io/badge/Deploy-RunPod-673AB7?style=for-the-badge&logo=runpod)](https://console.runpod.io/deploy?template=2kkfdbmlcc&ref=yx1lcptf) | [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) | <a href="https://ko-fi.com/duskfallcrew"><img src="https://img.shields.io/badge/Ko--Fi-Support-FF5E5B?style=for-the-badge&logo=kofi" alt="Ko-fi"></a> | [![DeepScan grade](https://deepscan.io/api/teams/29397/projects/31347/branches/1015145/badge/grade.svg)](https://deepscan.io/dashboard#view=project&tid=29397&pid=31347&bid=1015145) |

---

## 🚧 ALPHA — April 2026

Active development. Core training is working and verified across multiple LoRA types and model families. Expect rough edges, UI quirks, and the occasional surprise — that's what alpha is for.

### Current State

- ✅ **End-to-end training** — Verified on VastAI, RunPod, and local 4090. SDXL, Illustrious, Pony, Flux, NoobAI all confirmed working
- ✅ **40+ community presets** — Real training configs from the community across AdamW8bit, CAME, Prodigy, AdaFactor, Compass, Lion8bit, DoRA, LoHa, LoCon, LyCORIS
- ✅ **Dataset management** — Upload (individual, zip, URL), tag editor, bulk caption operations, WD14/BLIP/GIT auto-tagging
- ✅ **Training config** — 132+ parameters across 7 tabs, preset system, form persistence across sessions
- ✅ **Model browser** — Search and download from Civitai directly in the UI
- ✅ **HuggingFace upload** — Direct upload after training
- ✅ **Custom optimizers** — CAME, Compass, LPFAdamW, RMSProp via vendored LoraEasyCustomOptimizer
- ✅ **Security** — Path traversal prevention, input validation across all endpoints
- ✅ **Cross-platform** — Linux primary, Windows supported (WSL2 recommended for best results)

### Known Rough Edges

- 🔧 Training progress display shows 0/0 — log parsing in progress
- 🔧 Preset list ordering is rough — organization pass coming in beta
- 🔧 Compass, LPFAdamW, RMSProp optimizers are wired but have no UI dropdown yet
- 🔧 Rex and CosineAnnealing schedulers vendored but not exposed in UI

> 🐛 Hit something? [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) or [Discord](https://discord.gg/HhBSM9gBY)

---

## 🗺️ Roadmap

### Beta (next)
- Training progress display that actually works
- Job queue for sequential training runs
- Preset UI reorganization — filter by model family and use case
- Expose missing optimizers and schedulers in UI
- Upload progress indicator for large datasets
- Caption sanitization pre-flight check

### Post-Beta / Pre-Stable
- ComfyUI inference integration — generate images directly from the trainer UI without a separate WebUI
- Embedding merge tool — create and combine TI embeddings via vector arithmetic (port of klimaleksus/embedding-merge concept)
- VAE fine-tuning support
- Advanced model merging (QuantumMerge and similar techniques)
- SD 2.1 model type support

> This roadmap lives here in the README. No separate STATUS.md to fall out of date.

---

## Quick Start

### Requirements

- **GPU**: NVIDIA with CUDA 12.1+ (12GB VRAM minimum, 24GB recommended for SDXL)
- **Python**: 3.10 or 3.11
- **Node.js**: 20.19+ (22.x recommended — the installer will handle this automatically if missing)
- **Disk**: 50GB+ free space

#### GPU Compatibility

| GPU | Status | Notes |
|-----|--------|-------|
| **NVIDIA (CUDA 12.1+)** | Fully supported | Primary development and testing target |
| **AMD (ROCm)** | Should work, community supported | Kohya SS supports ROCm. Install PyTorch with ROCm index: `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2`. See [PyTorch ROCm docs](https://pytorch.org/get-started/locally/) for your version. |
| **NVIDIA via ZLUDA** | Should work, community supported | For running CUDA workloads on non-CUDA hardware. See [ZLUDA project](https://github.com/vosen/ZLUDA) for setup instructions. |
| **CPU-only** | Not recommended | Training will be extremely slow. Tagging/captioning will work but slowly. |

> **Note on ROCm and ZLUDA:** These should work since the underlying Kohya SS backend supports them, but we develop and test exclusively on NVIDIA CUDA. We can't provide hands-on debugging for ROCm/ZLUDA issues, but we welcome community contributions, bug reports, and documentation from users running these setups. If you get it working, let us know so we can share your setup notes!

#### 🚨 WHERE YOU INSTALL MATTERS — READ THIS (Windows)

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

---

### Installation

The installers will ask if you want a virtual environment. **Say yes** — it keeps this project's packages isolated and makes troubleshooting much easier. The venv is created as `.venv/` in the project folder.

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

> 💡 Both scripts will interactively prompt about virtual environments. You can also pass `--venv` or `--no-venv` to skip the prompt, and `--auto` to run fully unattended.

**Cloud Deployment:** Use the VastAI or RunPod deploy buttons above — auto-configures on launch. Training has been verified end-to-end on both platforms.

> 💰 **New to RunPod?** [Sign up with this link](https://runpod.io?ref=yx1lcptf) for bonus credits to get started.

---

### Starting the App

```bash
# Default ports: frontend on 3000, backend on 8000
start_services_local.bat   # Windows
./start_services_local.sh  # Linux

# Use custom ports if defaults are taken
start_services_local.bat --port 4000 --backend-port 9000
./start_services_local.sh --port 4000 --backend-port 9000

# Restart without a full reinstall
restart.bat   # Windows
./restart.sh  # Linux
```

**Access URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

> 💡 If you used a venv, activate it first (`source .venv/bin/activate` on Linux, `.venv\Scripts\activate` on Windows) before starting — or just re-run the start script, which handles it.

---

### Updating

**DO NOT delete the folder and re-clone.** This creates orphaned cache files and Python environment shims that cause installation issues.

**Normal update:**
```bash
cd Ktiseos-Nyx-Trainer
git pull

# Re-run the installer to pick up dependency changes
install.bat            # Windows
./install.sh           # Linux
```

**When things are broken and reinstalling didn't fix it:**
```bash
# Preview what will be deleted (nothing is removed)
python clean_slate.py --dry-run

# Delete build artifacts, venvs, node_modules, caches (keeps your models/datasets/outputs)
python clean_slate.py

# Skip confirmation prompt
python clean_slate.py --yes

# Nuclear option: also delete models, VAEs, datasets, outputs
python clean_slate.py --nuclear
```

After `clean_slate.py`, just re-run the installer:
```bat
REM Windows
install.bat

REM Linux
./install.sh
```

**When to use what:**
- `git pull` + re-run installer = normal updates (do this first)
- `clean_slate.py` = something is broken and reinstalling didn't fix it
- `clean_slate.py --nuclear` = scorched earth, start completely over

---

## What It Does

**Dataset Preparation:**
- Upload datasets via drag-and-drop interface
- Auto-tag images with WD14 models
- Edit captions with batch operations (bulk add/remove/replace)
- Image gallery with filtering

**Training Configuration:**
- 132+ parameters across 7 organized tabs
- Support for SDXL, SD1.5, Flux, SD3/SD3.5, Lumina
- Multiple LoRA types: Standard, LoCon, LoHa, LoKr, DoRA
- Real-time validation and progress monitoring
- Preset system with form persistence

**Model Browser:**
- Browse Civitai directly from the UI — search by keyword, tag, or creator
- Filter by model type, base model, sort order, and time period
- Download models directly into your local model folders

**Utilities:**
- LoRA resizing and metadata editing
- HuggingFace uploads direct from the web UI
- Model downloads from Civitai and HuggingFace

---

## Documentation

> 📝 Several guides are still being written. What exists is listed below — links that aren't here don't exist yet.

- 📘 [Installation Guide](documentation/installation/INSTALLATION.md) - Detailed setup for all platforms
- 📖 [General Guides](documentation/guides/general/README.md) - General usage documentation
- 🔧 Contributing and security: [CONTRIBUTING.md](CONTRIBUTING.md) • [SECURITY.md](SECURITY.md)

Deployment guides, troubleshooting, and feature documentation are in progress.

---

## Support

**Official Channels:**
- [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) - Bug reports and feature requests
- [Discord Server](https://discord.gg/HhBSM9gBY) - Community support and discussion

**Before Requesting Help:**
- Check [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) for known problems
- Include error messages, logs (`logs/app_YYYYMMDD.log`), and system info

**Installation Issues:**

Run the diagnostic tool to collect system information:
```bat
REM Windows
diagnose.bat

REM Linux/WSL
./diagnose.sh
```

Attach the generated `diagnostics_*.txt` file to your issue report.

---

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
