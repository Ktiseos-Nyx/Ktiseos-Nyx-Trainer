# Using Jupyter Notebooks on VastAI

> 🚀 **Already on VastAI?** Jupyter is already running! Just open the notebooks and go.

## ⚡ Quick Start (VastAI)

### You're Already Set Up! ✨

When you launch the Ktiseos Nyx template on VastAI with **Jupyter+SSH mode** (recommended), Jupyter Lab is automatically running on port 8888.

### Step 1: Access Jupyter Lab

1. **Find your instance IP** in VastAI dashboard
2. **Open Jupyter**: `http://<instance-ip>:8888`
3. **Navigate to**: `Ktiseos-Nyx-Trainer/jupyter-notebooks/`
4. **Open**: `Unified_LoRA_Trainer.ipynb`

**That's it!** Everything is pre-installed and ready to go.

---

## 📁 Where Are My Files?

On VastAI instances, your directory structure is:

```
/workspace/
├── Ktiseos-Nyx-Trainer/          # Main repository
│   ├── jupyter-notebooks/        # Notebooks are here!
│   ├── widgets/                  # Widget code
│   ├── core/                     # Backend managers
│   └── trainer/                  # Training backend
├── datasets/                     # Your training images
├── output/                       # Trained LoRAs
├── pretrained_model/             # Base models (SDXL, Flux, etc.)
├── vae/                          # VAE models
└── logs/                         # Service logs
```

**Pro Tip**: Map these directories to VastAI volumes for persistence!

---

## 🎯 Training Workflow on VastAI

### Option 1: Use Jupyter Notebooks (Classic)

1. Open Jupyter Lab at `http://<instance-ip>:8888`
2. Navigate to `Ktiseos-Nyx-Trainer/jupyter-notebooks/`
3. Open `Unified_LoRA_Trainer.ipynb`
4. Run cells in order
5. Download your trained LoRA from `/workspace/output/`

### Option 2: Use Web UI (Modern)

1. SSH into your instance (if services aren't auto-started)
2. Run `/start_services.sh`
3. Open Web UI at `http://<instance-ip>:3000`
4. Use the modern web interface

**Both use the same backend** - pick whichever you prefer!

---

## 🔄 Switching Between Interfaces

You can use both Jupyter and Web UI on the same instance:

| Interface | Port | URL | Best For |
|-----------|------|-----|----------|
| **Jupyter Lab** | 8888 | `http://<ip>:8888` | Notebook-style workflow |
| **Web UI** | 3000 | `http://<ip>:3000` | Modern web interface |
| **API** | 8000 | `http://<ip>:8000` | Programmatic access |

All three share the same:
- Datasets in `/workspace/datasets/`
- Models in `/workspace/pretrained_model/`
- Outputs in `/workspace/output/`

---

## 📝 Available Notebooks

### 1. Unified_LoRA_Trainer.ipynb
Complete training pipeline:
- Model downloads
- Dataset upload and tagging
- Training configuration
- Real-time monitoring
- Post-training utilities

### 2. Dataset_Preparation.ipynb
Dataset management only:
- Upload images
- WD14 tagging
- Caption editing
- Gallery-DL scraping
- HuggingFace uploads

### 3. Utilities_Notebook.ipynb
Post-training tools:
- LoRA resizing
- Metadata editing
- Model inspection
- HuggingFace uploads

---

## 💾 Persistence on VastAI

### Important: Map Volumes!

When launching your VastAI instance, map these container paths to volumes:

| Container Path | What It Stores | Recommended Size |
|----------------|----------------|------------------|
| `/workspace/datasets` | Training images | 10-50 GB |
| `/workspace/output` | Trained LoRAs | 5-20 GB |
| `/workspace/pretrained_model` | Base models | 20-50 GB |
| `/workspace/vae` | VAE models | 1-5 GB |

**Why?** Without volumes, your data disappears when the instance stops!

### Downloading Your LoRAs

After training completes:

**Option 1: Jupyter File Browser**
1. Navigate to `/workspace/output/<project_name>/`
2. Right-click your LoRA file
3. Select "Download"

**Option 2: SSH/SCP**
```bash
scp -P <ssh-port> root@<instance-ip>:/workspace/output/<project>/*.safetensors ./
```

**Option 3: Web UI**
1. Open `http://<instance-ip>:3000`
2. Go to "Files" tab
3. Browse to output directory
4. Download files

---

## 🎨 First Training on VastAI

### Quick Training Test (20 minutes)

1. **Access Jupyter**: `http://<instance-ip>:8888`
2. **Open**: `Unified_LoRA_Trainer.ipynb`
3. **Run Setup Cell** (Cell 1)
4. **Download SDXL Model** (Cell 2 - takes 5-10 min)
5. **Upload 20 images** (Cell 3 - use Jupyter file upload)
6. **Tag with WD14** (Cell 4 - automated)
7. **Configure Training** (Cell 5 - use defaults)
8. **Start Training** (Cell 6 - monitor progress)
9. **Download LoRA** (from `/workspace/output/`)

**Total time**: ~20-40 minutes on RTX 3090

---

## 🐛 VastAI-Specific Issues

### "Cannot connect to Jupyter"

**Cause**: Port 8888 not forwarded

**Solution**:
- Check VastAI port forwarding settings
- Ensure firewall allows port 8888
- Try accessing via VastAI's proxy URL

### "Notebooks not found"

**Cause**: Repository not cloned

**Solution**:
```bash
# SSH into instance
ssh root@<instance-ip> -p <ssh-port>

# Run startup script
/startup.sh
```

### "Backend not set up"

**Cause**: First-time setup not complete

**Solution**:
In Jupyter notebook, run:
```python
!python /workspace/Ktiseos-Nyx-Trainer/installer.py
```

### "Out of disk space"

**Cause**: Instance disk full

**Solution**:
- Delete old training outputs
- Clear cached models
- Use VastAI volumes for storage
- Upgrade to larger disk instance

---

## 💰 Cost Optimization

### Tips for VastAI Training

1. **Download models once**: Store in persistent volume, reuse across sessions
2. **Use interruptible instances**: 50-70% cheaper
3. **Stop when idle**: VastAI charges by the hour
4. **Delete old outputs**: Keep disk usage low
5. **Monitor training**: Don't leave it running unnecessarily

### Training Cost Examples

20 images, 10 epochs, SDXL:

| GPU | Training Time | Cost (On-Demand) | Cost (Interruptible) |
|-----|--------------|------------------|----------------------|
| RTX 3060 12GB | ~60 min | $0.10-0.15 | $0.05-0.08 |
| RTX 3090 24GB | ~20 min | $0.07-0.12 | $0.03-0.06 |
| RTX 4090 24GB | ~15 min | $0.09-0.13 | $0.05-0.07 |

**Pro Tip**: Start with interruptible for testing!

---

## 🔧 Advanced: Using Both Workflows

### Scenario: Prepare Dataset in Jupyter, Train in Web UI

1. **In Jupyter**: Run `Dataset_Preparation.ipynb`
   - Upload images
   - Tag with WD14
   - Edit captions
   - Save dataset to `/workspace/datasets/my_dataset/`

2. **In Web UI**:
   - Open `http://<instance-ip>:3000`
   - Go to "Training" tab
   - Select dataset from file browser
   - Configure and train

**Why?** Jupyter is great for interactive dataset prep, Web UI is cleaner for training!

### Scenario: Train in Jupyter, Download via Web UI

1. **In Jupyter**: Run full training in `Unified_LoRA_Trainer.ipynb`
2. **In Web UI**: Use file manager to browse and download trained LoRAs

**Why?** Web UI file manager has better download handling than Jupyter!

---

## 📊 Monitoring Training

### From Jupyter Notebook

The training widget shows real-time progress:
- Epoch/step counters
- Loss values
- ETA calculation
- Sample image generation

### From TensorBoard (Optional)

If you set `START_TENSORBOARD=true`:
1. Open `http://<instance-ip>:6006`
2. View detailed training metrics
3. Compare multiple training runs

### From Logs

```bash
# SSH into instance
tail -f /workspace/Ktiseos-Nyx-Trainer/trainer/derrian_backend/logs/train.log
```

---

## 🎓 Learning Path

1. **First Training**: Use defaults, 20 images, 10 epochs
2. **Experiment**: Try different learning rates, network dims
3. **Advanced**: Explore LyCORIS variants (LoCon, LoKr)
4. **Optimize**: Use block-wise learning rates
5. **Scale**: Train on larger datasets (100+ images)

---

## 📚 Resources

- **Main README**: `../README.md`
- **Deployment Guide**: `../docs/DEPLOYMENT.md`
- **Training Guide**: `../docs/TRAINING_GUIDE.md`
- **VastAI Setup**: `../VASTAI_SETUP.md`

---

## 🆘 Support

- **GitHub Issues**: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues
- **VastAI Support**: help@vast.ai

---

**Happy Training!** 🚀
