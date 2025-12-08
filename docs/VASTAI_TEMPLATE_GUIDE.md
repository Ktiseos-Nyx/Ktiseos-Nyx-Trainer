# VastAI Template Configuration Guide

## Overview

This guide shows you how to create a VastAI template for Ktiseos-Nyx-Trainer so users can rent a GPU and start training with one click.

## Prerequisites

- VastAI account: https://cloud.vast.ai
- This repository on GitHub (public)
- Scripts pushed to your main branch:
  - `vastai_setup.sh`
  - `start_services.sh`

## Template Configuration

### Step 1: Go to VastAI Templates

1. Visit: https://cloud.vast.ai/templates/
2. Click **"+ New"** button (or edit existing template)

### Step 2: Basic Identification

**Template Name:**
```
Ktiseos-Nyx LoRA Trainer
```

**Template Description:**
```
Professional LoRA training system with Next.js web UI.
Includes dataset preparation, training management, and file browser.
No installation required - just rent GPU and access via browser.
```

**Tags:** (optional)
```
stable-diffusion, lora, training, machine-learning
```

### Step 3: Docker Repository and Environment

**Image Path:Tag:**
```
vastai/base-image:latest
```
*(Uses VastAI's official base image - has Python, Node.js, CUDA pre-installed)*

**Container Disk:**
```
50 GB
```
*(Minimum for models + datasets. Users can increase when renting)*

### Step 4: Environment Variables

Click **"+ Add Variable"** for each:

| Key | Value |
|-----|-------|
| `PROVISIONING_SCRIPT` | `https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh` |
| `POST_START_SCRIPT` | `/workspace/Ktiseos-Nyx-Trainer/start_services.sh` |

**Important Notes:**
- Replace `Ktiseos-Nyx` with your GitHub username if you forked
- Replace `main` with your branch name if different
- Keep these URLs public (required for VastAI to download)

⚠️ **DO NOT add sensitive tokens here** (HuggingFace tokens, API keys, etc.)
Users should add those in the app UI after launching.

### Step 5: Ports

Click **"+ Add Port"** for each:

| Port | Protocol | Description |
|------|----------|-------------|
| `3000` | `http` | **Next.js Web UI (MAIN)** |
| `8000` | `http` | FastAPI Backend API |
| `8888` | `http` | Jupyter Lab (diagnostics only) |
| `22` | `tcp` | SSH (automatically configured) |

**Expose Direct:**
- ✅ Check this for port 3000 (main UI)
- ⚠️ Can leave unchecked for 8000/8888 (optional)

### Step 6: Launch Mode

**Select:**
```
☑️ Run Jupyter in background (optional for debugging)
☑️ Run SSH service
```

**OR** if using dropdown:
```
Jupyter + SSH
```

This enables:
- SSH access for debugging and diagnostics
- Your startup script runs automatically
- Jupyter Lab on port 8888 (for debugging only, not primary workflow)

### Step 7: Docker Options (Advanced)

**Docker Run Command:**
Leave as default or use:
```
-e "SHELL=/bin/bash"
```

**On-Start Script:**
```
/workspace/Ktiseos-Nyx-Trainer/start_services.sh
```

### Step 8: Resource Requirements (Optional)

**Minimum Requirements:**
- GPU: Any CUDA-capable GPU
- VRAM: 8GB+ recommended
- Disk: 50GB minimum

**Recommended:**
- GPU: RTX 3090 / RTX 4090 / A100
- VRAM: 16GB+
- Disk: 100GB+

*(Users choose when renting - these are just suggestions in description)*

### Step 9: README Tab (Optional but Recommended)

Switch to **"ReadMe"** tab and add user instructions:

````markdown
# Ktiseos-Nyx LoRA Trainer

A professional web-based LoRA training system with modern UI.

## 🚀 Quick Start

1. **Rent this instance** on VastAI
2. **Wait 5-10 minutes** for setup to complete
3. **Click the Port 3000 link** to open the web UI
4. **Start training!**

## 🌐 Accessing the Interface

After the instance starts, you'll see port links like:
- **Port 3000** ← 🎯 **CLICK THIS!** (Main UI)
- Port 8000 (API docs)
- Port 8888 (Jupyter Lab - diagnostics only)

## 📁 File System

All your work is in `/workspace/Ktiseos-Nyx-Trainer/`:
- `datasets/` - Your training images
- `output/` - Trained LoRA models
- `configs/` - Training configurations

## 🔧 Features

- ✅ Drag & drop dataset upload
- ✅ Built-in file manager
- ✅ Training progress monitoring
- ✅ HuggingFace integration
- ✅ Config templates included

## 🐛 Troubleshooting

**UI won't load?**
- Wait 5-10 minutes after instance starts (first-time setup)
- Check instance logs: SSH in and run `tail -f /var/log/vast.log`

**Need to restart services?**
SSH in and run:
```bash
cd /workspace/Ktiseos-Nyx-Trainer
./start_services.sh
```

**Need help?**
- GitHub Issues: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues
- Discord: [Your Discord Link]

## 💾 Saving Your Work

**Before destroying the instance:**
1. Download trained models from UI
2. OR: Upload to HuggingFace (built-in)
3. OR: Use SSH to copy files: `scp -r root@instance:/workspace/output ./`

Enjoy training! 🚀
````

### Step 10: Save Template

Click one of:
- **"Create"** - Saves to "My Templates" for later
- **"Create & Use"** - Saves and immediately search for GPUs

## Template Settings Summary

Here's a complete config you can copy:

```yaml
Template Name: Ktiseos-Nyx LoRA Trainer
Image: vastai/base-image:latest
Disk: 50GB

Environment Variables:
  PROVISIONING_SCRIPT: https://raw.githubusercontent.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/main/vastai_setup.sh
  POST_START_SCRIPT: /workspace/Ktiseos-Nyx-Trainer/start_services.sh

Ports:
  3000: http (Main UI - expose direct)
  8000: http (API)
  8888: http (Jupyter - optional)
  22: tcp (SSH)

Launch Mode: Jupyter + SSH (for diagnostics)
```

## For Users: How to Use Your Template

### Renting an Instance

1. Go to: https://cloud.vast.ai/
2. Click **"Search"** → Find your template
3. Click **"Rent"**
4. Choose GPU based on needs:
   - **Training SDXL:** 16GB+ VRAM (RTX 3090, 4090, A100)
   - **Training SD 1.5:** 8GB+ VRAM (RTX 3060, 3070)
5. Click **"Rent"** button

### First Time Setup

After renting:
1. **Wait 5-10 minutes** - provisioning script runs
2. **Check instance page** - status changes to "running"
3. **Find Port 3000 link** - looks like: `https://ssh4.vast.ai:12345`
4. **Click the link** - opens your training UI!

### Monitoring Setup Progress (Optional)

SSH into instance:
```bash
ssh root@instance-ip

# Watch setup logs
tail -f /var/log/vast.log

# Check if services started
ps aux | grep -E 'uvicorn|node'
```

### Using the Interface

1. **Upload Dataset:**
   - Click "File Manager" tab
   - Drag & drop images
   - Or use "Dataset Upload" widget

2. **Configure Training:**
   - Click "Training" tab
   - Load config template
   - Adjust parameters

3. **Start Training:**
   - Click "Start Training"
   - Monitor progress in real-time
   - Training runs on GPU automatically

4. **Download Model:**
   - Training completes → model in `/output`
   - Download via UI
   - OR upload to HuggingFace

### Stopping Instance

**IMPORTANT:** VastAI charges by the hour!

1. Download your trained models first!
2. Go to VastAI dashboard
3. Click "Destroy" on your instance
4. Confirm destruction

## Testing Your Template

Before publishing publicly:

1. **Rent the cheapest GPU** (< $0.20/hr)
2. **Wait for setup** (5-10 min)
3. **Test all features:**
   - UI loads on port 3000
   - File upload works
   - Training starts successfully
4. **Check logs** for errors
5. **Destroy instance**

If issues:
- Check `vastai_setup.sh` script
- Verify GitHub repo is public
- Check VastAI instance logs

## Making Template Public

Once tested:

1. Go to template settings
2. Toggle **"Make Public"**
3. Add to VastAI template marketplace
4. Users can find via search

## Advanced: Custom Docker Image (Optional)

Instead of PROVISIONING_SCRIPT, you can build your own Docker image:

```dockerfile
FROM vastai/base-image:latest
# Pre-install everything
# Faster startup but requires rebuilding for updates
```

Then use your image:
```
Image: yourdockerhub/ktiseos-nyx-trainer:latest
```

See `docs/DOCKER_IMAGE.md` for full guide.

## Troubleshooting

### Setup takes forever (>15 minutes)
- Normal for first-time (downloading models, npm install)
- SSH in and check: `tail -f /var/log/vast.log`

### Port 3000 shows "connection refused"
- Services might not have started
- SSH in: `cd /workspace/Ktiseos-Nyx-Trainer && ./start_services.sh`

### "Repository not found" error
- GitHub repo must be public
- Check URL in PROVISIONING_SCRIPT is correct

### Out of disk space
- Increase "Container Disk" in template settings
- Or: Users select more disk when renting

## Cost Estimates

Typical VastAI rental costs:
- **RTX 3090 (24GB):** $0.20-0.40/hour
- **RTX 4090 (24GB):** $0.40-0.70/hour
- **A100 (40GB):** $0.80-1.50/hour

Example training session:
- SDXL LoRA (20 epochs): ~2-3 hours
- Cost: $0.40-1.20 depending on GPU

## Next Steps

- [ ] Create template on VastAI
- [ ] Test with cheap GPU instance
- [ ] Update README with your specifics
- [ ] Share with community!

---

**Questions?** Open an issue: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues
