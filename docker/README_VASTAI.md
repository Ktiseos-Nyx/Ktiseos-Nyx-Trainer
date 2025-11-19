# VastAI Deployment Guide

> ⚠️ **BETA - IN ACTIVE DEVELOPMENT**
>
> This project is in beta. Features may not work as expected. Please report issues at:
> https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues

## Launch Modes

VastAI offers three launch modes for this template:

### 1. **Jupyter + SSH** (Recommended ⭐)
- **What it does**: Starts Jupyter Lab automatically + SSH access
- **Why use it**: Full debugging access if services crash
- **How to start services**: SSH in and run `/startup.sh` manually
- **Best for**: Most users, development, testing

### 2. **SSH Only**
- **What it does**: Only SSH access, no auto-start services
- **Why use it**: Maximum control over what runs
- **How to start services**: SSH in and run `/startup.sh` manually
- **Best for**: Advanced users who want manual control

### 3. **Docker ENTRYPOINT**
- **What it does**: Runs exactly as designed (auto-starts all services)
- **Why use it**: Hands-off deployment
- **How it works**: Executes `/startup.sh` automatically on container start
- **Best for**: Production deployments where you trust the code

## Quick Start After SSH Login

When using **Jupyter+SSH** or **SSH only** modes:

```bash
# Option 1: Auto-start everything
/startup.sh

# Option 2: Start services manually
cd /workspace/Ktiseos-Nyx-Trainer

# Start backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start frontend (after building)
cd frontend
npm install
npm run build
npm run start &

# Optional: Start TensorBoard
tensorboard --logdir=/workspace/training_logs --host=0.0.0.0 --port=6006 &
```

## Service URLs

After starting services (via startup.sh or manually):

| Service | Port | URL | Notes |
|---------|------|-----|-------|
| **Frontend** | 3000 | `http://<instance-ip>:3000` | Main Web UI |
| **Backend API** | 8000 | `http://<instance-ip>:8000` | FastAPI |
| **API Docs** | 8000 | `http://<instance-ip>:8000/docs` | Swagger UI |
| **Jupyter Lab** | 8888 | `http://<instance-ip>:8888` | VastAI auto-starts |
| **TensorBoard** | 6006 | `http://<instance-ip>:6006` | Optional |

## Troubleshooting

### Services won't start automatically?
**Cause**: You're in Jupyter+SSH or SSH-only mode (VastAI doesn't run ENTRYPOINT)

**Solution**:
```bash
# SSH into your instance
ssh root@<instance-ip> -p <ssh-port>

# Run startup script
/startup.sh
```

### Need to manually restart a service?

```bash
# Check what's running
ps aux | grep -E 'uvicorn|node|npm'

# Kill a service
pkill -f uvicorn  # Backend
pkill -f "npm run start"  # Frontend

# Restart
cd /workspace/Ktiseos-Nyx-Trainer
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
cd frontend && npm run start &
```

### Check service logs

```bash
# Backend logs
tail -f /workspace/logs/backend.log

# Frontend logs
tail -f /workspace/logs/frontend.log

# Training logs (during active training)
tail -f /workspace/training_logs/<project_name>/train.log
```

### Repository not found?

First-time setup requires internet to clone the repository:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

# Run installer
python installer.py
```

## Environment Variables

Set these when launching the VastAI instance:

| Variable | Default | Description |
|----------|---------|-------------|
| `START_JUPYTER` | `false` | Auto-start Jupyter Lab on port 8888 |
| `START_TENSORBOARD` | `false` | Auto-start TensorBoard on port 6006 |

**Note**: In Jupyter+SSH mode, VastAI starts Jupyter for you, so `START_JUPYTER=true` is redundant.

## Data Persistence

Map these directories to VastAI volumes for persistence across instance restarts:

| Container Path | Purpose | Size Needed |
|----------------|---------|-------------|
| `/workspace/datasets` | Training images | 10-50 GB |
| `/workspace/output` | Trained LoRA files | 5-20 GB |
| `/workspace/pretrained_model` | Base models | 20-50 GB |
| `/workspace/vae` | VAE models | 1-5 GB |
| `/workspace/training_logs` | Logs & TensorBoard | 1-5 GB |

**Total**: 50-200 GB depending on how many models you store

## Cost Optimization Tips

1. **Use Interruptible Instances**: 50-70% cheaper, fine for most training
2. **Stop When Idle**: VastAI charges by the hour - stop after downloading your LoRAs
3. **Pre-download Models**: Store base models in persistent volumes
4. **Delete Old Logs**: TensorBoard logs can grow large over time

## Security Notes

⚠️ **VastAI instances are publicly accessible by default!**

- Jupyter Lab has no password in this template (for ease of use)
- Frontend/Backend have no authentication
- Only use for personal projects or add authentication
- Don't store sensitive data in instances

### Adding Password Protection

For Jupyter Lab:
```bash
# Generate password hash
jupyter lab password

# Edit startup.sh and remove --NotebookApp.token='' --NotebookApp.password=''
```

For Frontend/Backend:
- Add authentication middleware (see DEPLOYMENT.md)

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
- **VastAI Support**: help@vast.ai
- **Training Guide**: See `docs/TRAINING_GUIDE.md`
