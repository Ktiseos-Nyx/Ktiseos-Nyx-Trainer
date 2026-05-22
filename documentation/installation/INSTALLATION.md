# Installation Guide

Technical reference for all supported platforms.

---

## Table of Contents

- [Platform Support](#platform-support)
- [Requirements](#requirements)
- [Installation — Windows](#installation--windows)
- [Installation — Linux](#installation--linux)
- [Installation — macOS](#installation--macos-ui-only)
- [Cloud Deployment — VastAI](#cloud-deployment--vastai)
- [Cloud Deployment — RunPod](#cloud-deployment--runpod)
- [Manual / Advanced Setup](#manual--advanced-setup)
- [Post-Install: Starting the App](#post-install-starting-the-app)
- [Updating](#updating)
- [Maintenance Scripts](#maintenance-scripts)
- [Requirements Files Reference](#requirements-files-reference)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Platform Support

| Platform | Training | Installer | Start script |
|----------|----------|-----------|--------------|
| Windows 10/11 (NVIDIA) | ✅ | `install.bat` | `start_services_local.bat` |
| Linux (NVIDIA) | ✅ | `install.sh` | `start_services_local.sh` |
| VastAI | ✅ | `vastai_setup.sh` (auto) | Supervisor (auto) |
| RunPod | ✅ | `provision_runpod.sh` (auto) | `start_services_runpod.sh` |
| macOS | ❌ training / ✅ UI only | manual | `start_services_local.sh` |
| AMD (ROCm) | community | manual PyTorch ROCm | `start_services_local.sh` |
| CPU-only | not recommended | — | — |

AMD ROCm and ZLUDA are not tested by the core team. They should work given Kohya SS supports them; see [PyTorch ROCm docs](https://pytorch.org/get-started/locally/) and [ZLUDA](https://github.com/vosen/ZLUDA) for setup.

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 12 GB VRAM, CUDA 12.1+ | NVIDIA 24 GB VRAM |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB free | 100 GB free |

VRAM guidance by model type:

| Model | Minimum VRAM | Notes |
|-------|-------------|-------|
| SD 1.5 | 8 GB | Small batches |
| SDXL / Pony / Illustrious / NoobAI | 16 GB | 24 GB for batch > 1 |
| Flux.1 dev/schnell | 24 GB | fp8 base helps on 16 GB |
| Anima | 24 GB | 40 GB for large batches |

### Software

| Dependency | Required version | Notes |
|------------|-----------------|-------|
| Python | 3.10 or 3.11 | 3.12 untested; 3.13 not supported |
| Node.js | 20.19+ | 22.x recommended |
| Git | any recent | needed for clone and updates |
| CUDA Toolkit | 12.1+ | must match PyTorch build |
| NVIDIA drivers | 525.x+ | for CUDA 12.1 |

Python downloads: [python.org](https://www.python.org/downloads/)  
Node.js downloads: [nodejs.org](https://nodejs.org/)  
Git (Windows): [git-scm.com](https://git-scm.com/download/win)

---

## Installation — Windows

**Supported:** Windows 10 21H2+ and Windows 11, NVIDIA GPU with CUDA 12.1+.

**Install location:** Use a path under your user directory (e.g. `C:\Users\YourName\Projects\`). Paths under `C:\`, `Program Files`, `Program Files (x86)`, `Windows`, network drives, OneDrive, Dropbox, and Google Drive will cause permission errors.

```bat
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
install.bat
```

The installer:
1. Checks for Python 3.10/3.11 and Node.js 20+
2. Creates `.venv/` virtual environment (unless `--no-venv` is passed)
3. Installs PyTorch with CUDA 12.1 index
4. Installs Python dependencies from `requirements_windows.txt`
5. Runs `npm install` in `frontend/`
6. Runs `npm run build` in `frontend/`

**Installer flags:**

| Flag | Effect |
|------|--------|
| `--venv` | Create virtual environment without prompting |
| `--no-venv` | Skip virtual environment creation |
| `--auto` | Non-interactive (implies `--venv`) |
| `--verbose` | Show full pip/npm output |

---

## Installation — Linux

**Supported:** Ubuntu 20.04+, Debian 11+, and most systemd distros with NVIDIA GPU.

```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
./install.sh
```

Same flags as Windows: `--venv`, `--no-venv`, `--auto`, `--verbose`.

The installer uses `requirements_linux.txt` instead of `requirements_windows.txt`. The CUDA index and PyTorch version are the same.

---

## Installation — macOS (UI only)

Training is not supported on macOS — Kohya SS requires CUDA. The web UI and API will run for development purposes.

```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # CPU-only, no PyTorch CUDA

cd frontend
npm install --legacy-peer-deps
npm run build
cd ..
```

Start:
```bash
./start_services_local.sh
```

---

## Cloud Deployment — VastAI

**One-click:** Use the deploy button in the README. The template runs `vastai_setup.sh` automatically on first boot.

**Manual setup on a custom VastAI instance:**

Recommended instance spec: NVIDIA RTX 3090 or better, 24 GB+ VRAM, 100 GB disk, Ubuntu 22.04 + CUDA 12.1 image.

```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
chmod +x vastai_setup.sh
./vastai_setup.sh
```

`vastai_setup.sh` differs from the local Linux installer in two ways:
- Deletes `package-lock.json` before `npm install` to avoid lockfile platform mismatch (Windows-generated lockfiles break on Linux hosts)
- Installs from `requirements_cloud.txt` which omits packages that conflict with the VastAI base image

Services are managed by Supervisor and restart automatically after instance reboots.

**Dev branch on VastAI:**
```bash
./vastai_setup_dev.sh
```

---

## Cloud Deployment — RunPod

**One-click:** Use the deploy button in the README.

**Manual setup:**
```bash
git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer
chmod +x provision_runpod.sh
./provision_runpod.sh
```

Start services on RunPod:
```bash
./start_services_runpod.sh
```

**Dev branch on RunPod:**
```bash
./provision_runpod_dev.sh
```

---

## Manual / Advanced Setup

For custom environments, CI, or development without the installer scripts.

### Backend

```bash
# Activate venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Start FastAPI backend
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Add `--reload` for development (restarts on code changes). Do not use `--reload` in production.

### Frontend

The frontend uses a custom `server.js` that proxies `/api/*` requests to the FastAPI backend and handles WebSocket upgrades. Use `npm start` (not `next start` directly) to ensure the proxy is active.

```bash
cd frontend

# Development (hot reload, no proxy — use direct FastAPI URLs)
npm run dev

# Production build
npm run build

# Production server (includes API proxy via server.js)
npm start
```

Production access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Custom ports

```bash
# Windows
start_services_local.bat --port 4000 --backend-port 9000

# Linux
./start_services_local.sh --port 4000 --backend-port 9000
```

---

## Post-Install: Starting the App

```bash
# Windows
start_services_local.bat

# Linux / macOS
./start_services_local.sh
```

**Quick restart** (no reinstall, no rebuild):
```bash
restart.bat     # Windows
./restart.sh    # Linux
```

**Full pull + rebuild + restart** (after pulling updates that change dependencies):
```bash
./fetch-restart.sh   # Linux / VastAI / RunPod
```

---

## Updating

```bash
git pull

# Re-run installer to pick up dependency changes
install.bat       # Windows
./install.sh      # Linux
```

The installer is idempotent — re-running it on an existing install only updates what has changed.

---

## Maintenance Scripts

### clean_slate.py

Removes generated artifacts without deleting user data. Useful when the install is in a broken state that reinstalling hasn't resolved.

```bash
# Preview what will be removed (nothing is deleted)
python clean_slate.py --dry-run

# Remove build artifacts, .venv, node_modules, pip caches
python clean_slate.py

# Skip confirmation prompt
python clean_slate.py --yes

# Also remove models, VAEs, datasets, and outputs
python clean_slate.py --nuclear
```

What `clean_slate.py` removes (default):
- `.venv/`
- `frontend/node_modules/`
- `frontend/.next/`
- `__pycache__/` directories
- Pip cache entries for this project
- Generated TOML configs in `config/`

What it preserves (default):
- `pretrained_model/` (downloaded models)
- `vae/`
- `datasets/`
- `output/` (trained LoRAs)
- `presets/`
- `logs/`

After running `clean_slate.py`, re-run the installer.

### diagnose.bat / diagnose.sh

Collects system information for bug reports.

```bash
diagnose.bat    # Windows
./diagnose.sh   # Linux
```

Output: `diagnostics_YYYYMMDD_HHMMSS.txt` in the project root. Include this file when opening a GitHub issue.

The diagnostic collects: OS version, Python version, Node.js version, CUDA version, `nvidia-smi` output, installed pip packages, installed npm packages, and recent log entries.

---

## Requirements Files Reference

| File | Used by | Contains |
|------|---------|----------|
| `requirements_base.txt` | All platforms | Core ML deps: torch, diffusers, transformers, accelerate, Kohya SS deps |
| `requirements_windows.txt` | `install.bat` | Base + Windows-specific packages (bitsandbytes Windows build, etc.) |
| `requirements_linux.txt` | `install.sh` | Base + Linux-specific packages |
| `requirements_cloud.txt` | `vastai_setup.sh`, `provision_runpod.sh` | Base + cloud-specific; omits packages pre-installed in cloud images |
| `requirements.txt` | Manual / macOS | Minimal set without PyTorch CUDA (CPU-only fallback) |
| `requirements_dev.txt` | Development / CI | Adds pytest, pytest-asyncio, httpx for running tests |

Do not mix `requirements_windows.txt` and `requirements_linux.txt` — some packages have platform-specific binary builds that will fail on the wrong OS.

---

## Environment Variables

These are set automatically by the installer and start scripts. They are documented here for reference when running manually or in custom environments.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYTHONIOENCODING` | `utf-8` | Prevents cp1252 encoding errors on Windows |
| `PYTHONUTF8` | `1` | Forces UTF-8 mode in Python 3.7+ |
| `PYTHONUNBUFFERED` | `1` | Disables stdout/stderr buffering so training logs stream in real time |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000/api` | Backend API base URL used by the frontend |
| `PORT` | `3000` | Frontend port (set by start scripts when `--port` is passed) |
| `BACKEND_PORT` | `8000` | FastAPI port (set by start scripts when `--backend-port` is passed) |

---

## Troubleshooting

### Python not found

```text
'python' is not recognized as an internal or external command
```

- Windows: reinstall Python with "Add to PATH" checked, or add manually via System Properties → Environment Variables
- Linux: `sudo apt install python3.10 python3.10-venv python3-pip`
- Verify: `python --version` or `python3 --version`

### CUDA not available

```python
torch.cuda.is_available()  # returns False
```

1. Verify driver: `nvidia-smi` — if this fails, drivers aren't installed
2. Check CUDA version in `nvidia-smi` output — must be 12.1+
3. Verify PyTorch CUDA build: `python -c "import torch; print(torch.version.cuda)"`
4. If PyTorch reports CPU-only, reinstall with the correct CUDA index:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### npm install fails

```text
npm ERR! ERESOLVE unable to resolve dependency tree
```

This error should not occur on Next.js 16 / React 19. If it does, clear the npm cache and retry:
```bash
npm cache clean --force
npm install
```

### Permission denied (Windows)

- Do not install in `C:\`, `Program Files`, or any system-owned directory
- Do not install in OneDrive, Dropbox, or Google Drive synced folders
- Install in a path you own: `C:\Users\YourName\Projects\`

### Permission denied (Linux)

```bash
chmod +x install.sh start_services_local.sh restart.sh
```

### Port already in use

```bash
# Linux/macOS — find and kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Windows — find PID, then kill it
netstat -ano | findstr :3000
taskkill /PID <pid> /F
```

Or start on different ports:
```bash
start_services_local.bat --port 4000 --backend-port 9000
```

### Training fails immediately

1. Check GPU availability: `nvidia-smi`
2. Check VRAM: `nvidia-smi --query-gpu=memory.free,memory.total --format=csv`
3. Check logs in `logs/app_YYYYMMDD.log`
4. Run the diagnostic: `diagnose.bat` / `./diagnose.sh`
5. Open a [GitHub issue](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) with the `diagnostics_*.txt` output

### VastAI: services not starting after reboot

Supervisor manages the services. Check status:
```bash
supervisorctl status
supervisorctl restart all
```

### node_modules in project root (Windows)

If you see a `node_modules/` directory in the project root (not in `frontend/`), it was created by a shadcn CLI command running in the wrong directory. It is safe to delete.

---

*For questions not covered here, open a [GitHub issue](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues) or ask in [Discord](https://discord.gg/HhBSM9gBY).*
