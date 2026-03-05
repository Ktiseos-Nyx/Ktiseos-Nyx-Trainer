# Installation Guide

Comprehensive installation instructions for all supported platforms.

## Table of Contents

- [Platform Support Overview](#platform-support-overview)
- [Prerequisites](#prerequisites)
- [Local Installation](#local-installation)
  - [Windows](#windows)
  - [Linux (NVIDIA GPU)](#linux-nvidia-gpu)
  - [macOS (UI Development Only)](#macos-ui-development-only)
- [Cloud Deployment](#cloud-deployment)
  - [VastAI](#vastai)
  - [RunPod](#runpod)
- [Manual Installation](#manual-installation)
- [Troubleshooting](#troubleshooting)

## Platform Support Overview

| Platform        | Training Supported | Installer Script              | Start Script                 | Notes                                  |
|-----------------|--------------------|-------------------------------|------------------------------|----------------------------------------|
| **Windows**     | ✅ (CUDA 12.1)     | `install.bat`                 | `start_services_local.bat`   | Requires NVIDIA GPU; uses `.dll` files |
| **Linux (NVIDIA)** | ✅ (CUDA 12.1)  | `installer_local_linux.py`    | `start_services_local.sh`    | Full GPU training; matches VastAI      |
| **macOS**       | ❌ (CPU-only UI)   | Manual `pip install`          | `start_services_local.sh`    | No training (Kohya SS lacks MPS support) |
| **VastAI**      | ✅ (CUDA 12.1)     | Auto via `vastai_setup.sh`    | Auto via Supervisor          | Uses `installer_remote.py` internally  |
| **RunPod**      | ⚠️ (Untested)      | Use Linux local scripts       | Use Linux local scripts      | May work; not officially supported yet |

## Prerequisites

### System Requirements

**For Training:**
- **GPU**: NVIDIA GPU with CUDA 12.1+ support
  - Minimum: 12GB VRAM (SD1.5, small batches)
  - Recommended: 24GB VRAM (SDXL, Flux)
  - Not Supported: AMD ROCm, Apple Silicon (MPS)
- **RAM**: 16GB+ recommended (8GB minimum)
- **Disk**: 50GB+ free space
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS (UI only)

**For UI Development (no training):**
- **CPU**: Any modern processor
- **RAM**: 8GB+
- **Disk**: 20GB+

### Software Requirements

**Python:**
- Version: 3.10 or 3.11 (3.12 may work but untested)
- Download: [python.org](https://www.python.org/downloads/)
- Recommended: Python 3.10.6

**Node.js:**
- Version: 18+ (20+ recommended)
- Download: [nodejs.org](https://nodejs.org/)

**Git:**
- Windows: [git-scm.com](https://git-scm.com/download/win)
- macOS: `xcode-select --install`
- Linux: `sudo apt install git`

## Local Installation

### Windows

1. **Clone Repository:**
   ```bat
   git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   cd Ktiseos-Nyx-Trainer
   ```

2. **Run Installer:**
   ```bat
   install.bat
   ```

   The installer will:
   - Detect your Python installation
   - Create a virtual environment
   - Install PyTorch with CUDA 12.1
   - Install all dependencies (Kohya SS, LyCORIS, ONNX)
   - Set up the frontend (npm install + build)

3. **Start Services:**
   ```bat
   start_services_local.bat
   ```

4. **Access Web UI:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

**Quick Restart** (skip reinstall):
```bat
restart.bat
```

**Verbose Installation** (for debugging):
```bat
install.bat --verbose
```

### Linux (NVIDIA GPU)

1. **Clone Repository:**
   ```bash
   git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   cd Ktiseos-Nyx-Trainer
   ```

2. **Run Installer:**
   ```bash
   python installer_local_linux.py
   ```

   Or with verbose output:
   ```bash
   python installer_local_linux.py --verbose
   ```

3. **Start Services:**
   ```bash
   chmod +x start_services_local.sh
   ./start_services_local.sh
   ```

4. **Access Web UI:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

**Quick Restart:**
```bash
./restart.sh
```

### macOS (UI Development Only)

> ⚠️ **Training is NOT supported on macOS** - Kohya SS requires CUDA.
> You can run the web UI and API for development/testing, but training will fail.

1. **Clone Repository:**
   ```bash
   git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   cd Ktiseos-Nyx-Trainer
   ```

2. **Manual Setup:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install Python dependencies
   pip install -r requirements.txt

   # Install frontend dependencies
   cd frontend
   npm install
   npm run build
   cd ..
   ```

3. **Start Services:**
   ```bash
   ./start_services_local.sh
   ```

4. **Access Web UI:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## Cloud Deployment

### VastAI

**One-Click Deployment:**

Use the VastAI template linked in the main README badge. The system will:
1. Auto-run `vastai_setup.sh` on first launch
2. Install all dependencies via `installer_remote.py`
3. Configure supervisor for auto-restart
4. Expose ports via VastAI portal

**Manual VastAI Setup:**

If you want to manually configure VastAI:

1. **Create Instance:**
   - GPU: NVIDIA RTX 3090/4090 or better
   - VRAM: 24GB+ recommended
   - Disk: 100GB+
   - OS: Ubuntu 22.04 with CUDA 12.1

2. **SSH into Instance:**
   ```bash
   ssh root@<vastai-instance-ip>
   ```

3. **Run Setup Script:**
   ```bash
   git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   cd Ktiseos-Nyx-Trainer
   chmod +x vastai_setup.sh
   ./vastai_setup.sh
   ```

4. **Access via VastAI Portal:**
   - Frontend: Port 13000 (mapped to 3000)
   - Backend: Port 18000 (mapped to 8000)

### RunPod

> ⚠️ **RunPod support is experimental and untested.**

RunPod deployment should work using the Linux local installation scripts, but this hasn't been validated. If you try it:

1. Use a RunPod GPU instance with CUDA 12.1+
2. Follow the [Linux installation steps](#linux-nvidia-gpu)
3. Expose ports 3000 and 8000 via RunPod's port forwarding
4. Report results on GitHub!

## Manual Installation

For advanced users or custom setups:

### Backend Only

```bash
# Terminal 1 - Backend API
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Only

```bash
# Terminal 2 - Frontend
cd frontend
npm install
npm run dev  # Development mode
# OR
npm run build && npm start  # Production mode
```

### Production Deployment

```bash
# Build frontend
cd frontend
npm run build
cd ..

# Start services
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
cd frontend && npm start
```

## Troubleshooting

### Installation Issues

**"Python not found":**
- Ensure Python 3.10/3.11 is installed and in PATH
- Windows: Reinstall Python with "Add to PATH" checked
- Linux: `sudo apt install python3.10 python3-pip`

**"CUDA not available":**
- Verify NVIDIA drivers: `nvidia-smi`
- Install CUDA Toolkit 12.1+
- Restart after driver installation

**"npm install fails":**
- Update Node.js to 18+
- Clear npm cache: `npm cache clean --force`
- Try with `--legacy-peer-deps` flag

**"Permission denied":**
- Linux/Mac: Add execute permissions: `chmod +x *.sh`
- Windows: Run as Administrator

### Runtime Issues

**"Port already in use":**
```bash
# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :3000  # Windows (note PID, then taskkill /PID <pid>)
```

**"Cannot connect to backend":**
- Verify backend is running: http://localhost:8000/docs
- Check firewall settings
- Ensure ports 3000 and 8000 aren't blocked

**"Training fails immediately":**
- Check GPU availability: `nvidia-smi`
- Verify VRAM isn't full
- Check logs in `logs/` directory
- See [Troubleshooting Guide](guides/troubleshooting.md)

For detailed troubleshooting, see [docs/guides/troubleshooting.md](guides/troubleshooting.md).

## Next Steps

After installation:
1. 📖 Read the [Quick Start Guide](quickstart.md)
2. 🎯 Check the [Features Documentation](FEATURES.md)
3. 🚀 Join the [Discord](https://discord.gg/HhBSM9gBY) for support

---

**Having issues?** Check [Troubleshooting](guides/troubleshooting.md) or open a [GitHub Issue](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues).
