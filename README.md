# Ktiseos Nyx LoRA Trainer

LoRA training system built on Kohya SS with a modern web UI (Next.js + FastAPI). Train LoRA models locally or on cloud GPUs with an intuitive interface.

| Python | License | Deploy | Discord | Twitch | Support |
|---|---|---|---|---|---|
| ![Python](https://img.shields.io/badge/python-3.10+-blue.svg) | ![License](https://img.shields.io/badge/license-MIT-green.svg) | [![Deploy on VastAI](https://img.shields.io/badge/Deploy-VastAI-FF6B6B?style=for-the-badge&logo=nvidia)](https://cloud.vast.ai/?ref_id=70354&creator_id=70354&name=Ktiseos-Nyx-NextJS-Trainer) | [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord)](https://discord.gg/HhBSM9gBY) | [![Twitch](https://img.shields.io/badge/Twitch-Follow-9146FF?logo=twitch&style=for-the-badge)](https://twitch.tv/duskfallcrew) | <a href="https://ko-fi.com/duskfallcrew"><img src="https://img.shields.io/badge/Ko--Fi-Support-FF5E5B?style=for-the-badge&logo=kofi" alt="Ko-fi"></a> |

> ⚠️ **ALPHA STAGE**: Active development. [Report issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) • [Development Status](STATUS.md)

## Quick Start

### Requirements

- **GPU**: NVIDIA with CUDA 12.1+ (12GB VRAM minimum, 24GB for SDXL)
- **Python**: 3.10 or 3.11
- **Node.js**: 18+
- **Disk**: 50GB+ free space

📖 [Full Requirements & Installation Guide](docs/INSTALLATION.md)

#### 🛡️ STRONGLY RECOMMENDED: Use a Virtual Environment

**Quick venv setup:**

**Windows:**
```bat
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

REM The installer will ask if you want a venv (say yes!)
install.bat
```

**Linux:**
```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

# The installer will ask if you want a venv (say yes!)
./install.sh
```

**Starting the app:**
```bash
# The start scripts automatically activate your venv if it exists!
start_services_local.bat  # Windows
./start_services_local.sh # Linux
```

> 💡 **Pro tip:** The start scripts now handle venv activation automatically. Just run them and go!

---

### Installation (Without venv - not recommended)

**Windows:**
```bat
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
install.bat --no-venv
start_services_local.bat
```

**Linux:**
```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
./install.sh --no-venv
./start_services_local.sh
```

**Access URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**VastAI Deployment:** Use the deploy button above - auto-configures on launch.

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

📖 [Full Feature List](docs/FEATURES.md) • [In-App Documentation](http://localhost:3000/docs) (when running)

## Documentation

- 📘 [Installation Guide](docs/INSTALLATION.md) - Detailed setup for all platforms
- 🚀 [Deployment Guide](docs/DEPLOYMENT.md) - VastAI and RunPod deployment
- 🔧 [Troubleshooting](docs/guides/troubleshooting.md) - Common issues and solutions
- 💻 [Development Setup](docs/DEVELOPMENT_ENVIRONMENTS.md) - Contributing and local dev
- ✨ [Features](docs/FEATURES.md) - Complete feature documentation
- 📊 [Development Status](STATUS.md) - Current progress and roadmap

## Support

**Official Channels:**
- [GitHub Issues](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) - Bug reports and feature requests
- [Discord Server](https://discord.gg/HhBSM9gBY) - Community support and discussion

**Before Requesting Help:**
- Check [Troubleshooting Guide](docs/guides/troubleshooting.md)
- Review [Support Guidelines](docs/guides/troubleshooting.md#support-guidelines--boundaries)
- Include error messages, logs, and system info

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
