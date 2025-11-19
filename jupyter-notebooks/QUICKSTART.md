# Jupyter Notebooks - Quick Start Guide

> Get up and running with the Jupyter notebook workflow in 5 minutes!

## ⚡ 5-Minute Setup

### Step 1: Clone and Setup (2 minutes)

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
cd Ktiseos-Nyx-Trainer

# Run installer (sets up training backend)
python installer.py
```

### Step 2: Install Jupyter (1 minute)

```bash
# Install Jupyter Lab
pip install jupyterlab ipywidgets

# Enable widgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Step 3: Start Jupyter (30 seconds)

```bash
# Start Jupyter Lab
jupyter lab

# Or use classic Jupyter
jupyter notebook
```

### Step 4: Open Notebook (30 seconds)

1. Navigate to `jupyter-notebooks/` folder
2. Open `Unified_LoRA_Trainer.ipynb`
3. Run the first cell to verify setup

### Step 5: Start Training (1 minute)

1. Run cells in order (they're numbered)
2. Download a base model (SDXL recommended)
3. Upload your dataset
4. Configure training with the widget
5. Click "Start Training"

**Done!** 🎉

---

## 🎯 Which Notebook Should I Use?

### Just Want to Train a LoRA?
👉 **`Unified_LoRA_Trainer.ipynb`** - Everything you need in one notebook

### Already Have a Dataset?
👉 **`Unified_LoRA_Trainer.ipynb`** - Skip the dataset cells

### Need to Prepare Multiple Datasets?
👉 **`Dataset_Preparation.ipynb`** - Focus on dataset management

### Already Trained, Need Utilities?
👉 **`Utilities_Notebook.ipynb`** - Resize, upload, inspect LoRAs

---

## 📋 Minimal Training Checklist

- [ ] Run `python installer.py`
- [ ] Download a base model (SDXL, Flux, or SD 1.5)
- [ ] Upload training images (10+ images minimum)
- [ ] Tag images with WD14 tagger
- [ ] Configure training (defaults work fine!)
- [ ] Start training
- [ ] Download your LoRA from `output/` folder

---

## 🎨 First Time Training?

Use these settings for your first SDXL LoRA:

| Parameter | Value | Why |
|-----------|-------|-----|
| **Base Model** | SDXL 1.0 | Most popular |
| **Dataset Size** | 20-50 images | Good starting point |
| **Resolution** | 1024 | SDXL native |
| **Epochs** | 10-20 | Prevents overfitting |
| **Batch Size** | 2-4 | Depends on GPU |
| **Network Dim** | 16 or 32 | Standard LoRA size |
| **Learning Rate** | 1e-4 | Safe default |

**Training time**: ~20-60 minutes on RTX 3090 (24GB)

---

## 🔧 Common Issues (Quick Fixes)

### Issue: "Backend not found"
```bash
python installer.py
```

### Issue: "CUDA out of memory"
In training widget:
- Set `Batch Size` to 1
- Enable `Gradient Checkpointing`
- Enable `Cache Latents`

### Issue: "Widgets not showing"
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Issue: "Submodule empty"
```bash
git submodule update --init --recursive
```

---

## 💡 Pro Tips

1. **Start Small**: Train on 20 images first to test your setup
2. **Use Presets**: The training widget has preset configurations
3. **Monitor GPU**: Use `nvidia-smi` to check VRAM usage
4. **Save Configs**: Export training configs to reuse later
5. **Check Samples**: Enable sample generation to preview results during training

---

## 🚀 Next Steps

After your first successful training:

1. **Experiment with Parameters**: Try different learning rates, network dimensions
2. **Try Different Models**: SDXL, Flux, SD 1.5 each have unique characteristics
3. **Explore LyCORIS**: Try LoCon, LoKr for different use cases
4. **Use the Web UI**: For a modern interface, run `cd frontend && npm run dev`
5. **Read the Guides**: Check `../docs/TRAINING_GUIDE.md` for advanced techniques

---

## 📚 Learning Resources

- **Kohya Training Guide**: https://github.com/kohya-ss/sd-scripts
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **SDXL Guide**: See our `docs/TRAINING_GUIDE.md`
- **LyCORIS Documentation**: https://github.com/KohakuBlueleaf/LyCORIS

---

## ❓ Questions?

- **GitHub Issues**: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues
- **Discussions**: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/discussions

---

**Happy Training!** 🎉
