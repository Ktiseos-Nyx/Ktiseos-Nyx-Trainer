# Ktiseos Nyx LoRA Trainer - VastAI Template

Professional LoRA training system with Next.js web UI optimized for VastAI deployment. Train custom LoRA models with a modern, bug-free interface instead of Jupyter widgets.

## 🚀 Quick Start

1. **Rent Instance**: Select this template and choose your GPU
2. **Wait 5-10 minutes**: Instance provisions and installs automatically  
3. **Access UI**: Click "Port 3000" link on your instance page
4. **Start Training**: Upload dataset, configure settings, train your LoRA

## 🌐 Accessing Services

After your instance starts, you'll see these portal links:
- **Port 3000** ← Click this! (NextJS Frontend - Main UI)
- Port 8000 (FastAPI Backend - API docs)
- Port 8888 (Jupyter - Diagnostics only)  
- Port 6006 (TensorBoard - Training logs)

## 📋 Training Workflow

### 1. Dataset Preparation
- Navigate to **Dataset Page**
- Upload training images via drag & drop
- Use **Auto-Tag** with WD14 tagger for automatic captions
- Add your trigger word to all captions

### 2. Training Configuration  
- Go to **Training Page**
- Load a template or create new configuration
- Set dataset path and model settings
- Configure LoRA parameters (dimension, learning rates, etc.)

### 3. Start Training
- Click **Start Training** 
- Monitor progress in real-time
- View logs and loss curves via WebSocket updates

### 4. Download Results
- Training completes automatically
- Go to **Files Page** to browse `/output` directory
- Download trained LoRA models

## 💰 Cost Optimization

**Recommended GPUs:**
- **RTX 3090/4090/A100**: Best for SDXL training (~$0.40-0.80/hr)
- **RTX 3060/4060 Ti**: Budget option for SD 1.5 (~$0.20-0.40/hr)

**Cost Saving Tips:**
- Stop instances when not training
- Use interruptible instances for non-critical training
- Download models once - keep in persistent storage

## 🔧 Troubleshooting

**UI won't load?**
- Wait 5-10 minutes for full provisioning
- Check instance logs via SSH
- Ensure your trigger word is consistent

**Out of VRAM?**
- Reduce batch size to 1
- Enable gradient checkpointing  
- Use smaller network dimensions

**Port 3000 shows error?**
- Services may still be starting
- SSH in and check: `supervisorctl status`
- Check logs: `tail -f /var/log/portal/ktiseos-nyx.log`

## 🛠️ Template Configuration (For Recreation)

If this template ever needs to be recreated, here are the settings:

**Docker Options:**
```
-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -p 3000:3000 -p 8000:8000 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard|localhost:3000:3000:/:Frontend|localhost:8000:8000:/:API"
```

**Environment Variables:**
- `PROVISIONING_SCRIPT`: `https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh`
- `OPEN_BUTTON_PORT`: `1111`  
- `OPEN_BUTTON_TOKEN`: `1`
- `JUPYTER_DIR`: `/`
- `DATA_DIRECTORY`: `/workspace/`

**Launch Mode:** Jupyter + SSH (for diagnostics)

**Repository:** `https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer`

## 🆘 Support

- **GitHub Issues**: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues
- **Discord**: [Your Discord link]

Enjoy training! 🚀