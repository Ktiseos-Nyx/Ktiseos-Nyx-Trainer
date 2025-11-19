# VastAI Template Setup Summary

> ⚠️ **BETA - IN ACTIVE DEVELOPMENT**
> This project is currently in beta. Features may not work as expected, and breaking changes may occur.
> Please report issues at: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues

## 📦 What We Created

Complete VastAI template infrastructure for deploying Ktiseos Nyx LoRA Trainer on cloud GPU instances.

### Files Created

1. **`template.json`** - VastAI template configuration
   - Metadata, GPU/resource requirements
   - Port mappings, environment variables
   - Launch mode configuration (Jupyter+SSH default)
   - Base image: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`

2. **`Dockerfile`** - Container image definition
   - CUDA 12.1 + cuDNN 8 base
   - Python 3.11 + Node.js 20.x
   - Pre-installs PyTorch 2.5.1 with CUDA support
   - All training dependencies pre-installed

3. **`docker/startup.sh`** - Full initialization script
   - Git clone with submodules on first run
   - Dependency installation
   - Service startup (backend, frontend, optional Jupyter/TensorBoard)
   - Process monitoring

4. **`docker/start_services.sh`** - Quick service starter
   - Helper script for Jupyter+SSH mode
   - Checks if services already running
   - Starts backend + frontend only
   - Shows service URLs and status

5. **`docker/README_VASTAI.md`** - VastAI-specific documentation
   - Launch mode explanations
   - Troubleshooting guide
   - Service management commands

6. **`docs/DEPLOYMENT.md`** - Complete deployment guide
   - VastAI, Docker, and local setup instructions
   - Performance tuning by GPU tier
   - Security considerations

7. **`docker-compose.yml`** - Local testing setup
   - Multi-service orchestration
   - GPU passthrough configuration
   - Optional dev profile for Jupyter/TensorBoard

## 🚀 Launch Modes Explained

### Option 1: Jupyter + SSH (Recommended ⭐)

**What VastAI does:**
- Starts Jupyter Lab automatically on port 8888
- Provides SSH access
- Does NOT run the Docker ENTRYPOINT (our startup.sh won't auto-run)

**What you do:**
```bash
# SSH into instance
ssh root@<instance-ip> -p <ssh-port>

# Quick start services
/start_services.sh

# OR full setup (first time)
/startup.sh
```

**Why use this:**
- ✅ Full debugging access if something breaks
- ✅ Can manually restart services
- ✅ Can inspect logs, check processes
- ✅ Jupyter available for interactive work

### Option 2: SSH Only

**What VastAI does:**
- Only provides SSH access
- Nothing auto-starts

**What you do:**
```bash
# SSH in and manually start everything
/startup.sh
```

**Why use this:**
- Maximum control
- For advanced users

### Option 3: Docker ENTRYPOINT

**What VastAI does:**
- Runs `/startup.sh` automatically on container start
- All services auto-start

**What you do:**
- Nothing! Just wait for services to start
- Access web UI at `http://<instance-ip>:3000`

**Why use this:**
- ✅ Hands-off deployment
- ❌ No debugging access if it breaks
- ❌ Can't manually restart services

## 🎯 Recommended Workflow

1. **First Deployment**: Use **Jupyter+SSH** mode
   - SSH in after instance starts
   - Run `/start_services.sh` to start backend/frontend
   - Test everything works
   - Debug any issues

2. **After Testing**: Switch to **Docker ENTRYPOINT** if desired
   - Fully automated startup
   - Use for production training runs

3. **Development**: Use **Jupyter+SSH** always
   - Full access for debugging
   - Can restart services as needed

## 🔧 Service Management

### Quick Commands (Jupyter+SSH mode)

```bash
# Start all services
/start_services.sh

# Check what's running
ps aux | grep -E 'uvicorn|npm'

# View logs
tail -f /workspace/logs/backend.log
tail -f /workspace/logs/frontend.log

# Restart backend
pkill -f uvicorn
cd /workspace/Ktiseos-Nyx-Trainer
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Restart frontend
pkill -f "npm run start"
cd /workspace/Ktiseos-Nyx-Trainer/frontend
npm run start &
```

## 📊 Resource Requirements

### Minimum Specs
- **GPU**: 12GB VRAM (RTX 3060, RTX 4060 Ti)
- **RAM**: 16GB system memory
- **Disk**: 50GB (100GB+ recommended)
- **CUDA**: 12.1 or higher

### Recommended Specs
- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A5000)
- **RAM**: 32GB system memory
- **Disk**: 200GB with persistent volumes

## 💰 Cost Estimates (VastAI)

Approximate hourly rates (as of 2024):

| GPU | VRAM | Typical Cost | Training Speed |
|-----|------|--------------|----------------|
| RTX 3060 | 12GB | $0.10-0.15/hr | Slow but cheap |
| RTX 4060 Ti | 16GB | $0.15-0.20/hr | Good balance |
| RTX 3090 | 24GB | $0.20-0.35/hr | Fast, affordable |
| RTX 4090 | 24GB | $0.35-0.50/hr | Fastest consumer |
| A100 40GB | 40GB | $0.80-1.50/hr | Professional |

**Example**: Train an SDXL LoRA with 1000 images, 10 epochs
- RTX 3060: ~7.5 hours = $0.75-1.13
- RTX 3090: ~2.5 hours = $0.50-0.88
- A100: ~1.5 hours = $1.20-2.25

**Pro tip**: Use interruptible instances (50-70% cheaper) if you don't mind potential restarts.

## 📁 Persistent Storage Setup

Map these VastAI volumes to persist data across instance restarts:

| Container Path | Purpose | Size |
|----------------|---------|------|
| `/workspace/datasets` | Training images | 10-50GB |
| `/workspace/output` | Trained LoRAs | 5-20GB |
| `/workspace/pretrained_model` | Base models (SDXL, Flux, SD3) | 20-50GB |
| `/workspace/vae` | VAE models | 1-5GB |
| `/workspace/training_logs` | TensorBoard logs | 1-5GB |

**Total**: 50-200GB depending on usage

## 🐛 Common Issues & Fixes

### "Repository not found"
```bash
cd /workspace
git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
/startup.sh
```

### "Cannot connect to frontend"
```bash
# Check if services running
ps aux | grep -E 'uvicorn|npm'

# If not, start them
/start_services.sh

# Check logs
tail -f /workspace/logs/frontend.log
```

### "CUDA out of memory"
- Reduce `train_batch_size` to 1 or 2
- Enable `gradient_checkpointing`
- Enable `cache_latents`
- Try FP8 training

### "Port already in use"
```bash
# Find what's using the port
lsof -i :3000
lsof -i :8000

# Kill the process
pkill -f uvicorn
pkill -f "npm run start"

# Restart
/start_services.sh
```

## 🔒 Security Notes

⚠️ **IMPORTANT**: VastAI instances are publicly accessible!

- No authentication on frontend/backend by default
- Jupyter Lab has no password (for convenience)
- Anyone with the IP can access your services

**For production use**, add authentication:
- FastAPI middleware for backend
- NextAuth.js for frontend
- Jupyter password protection

See `docs/DEPLOYMENT.md` for detailed security setup.

## 📋 Submission Checklist

Before submitting to VastAI marketplace:

- [ ] Update repository URL in `template.json`
- [ ] Update repository URL in `docker/startup.sh` (line 19)
- [ ] Update repository URLs in documentation
- [ ] Push all changes to GitHub
- [ ] Test Docker build locally
- [ ] Test on VastAI instance
- [ ] Submit template via VastAI dashboard

## 🔗 Resources

- **VastAI Templates Guide**: https://docs.vast.ai/documentation/templates/creating-templates
- **VastAI Pricing**: https://vast.ai/
- **Docker Documentation**: https://docs.docker.com/
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Training Guide**: `docs/TRAINING_GUIDE.md`

---

**Questions or issues?** Open an issue at: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues
