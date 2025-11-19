# Jupyter Notebook Workflow

> 📓 **Classic Jupyter Interface**
>
> Prefer the original Jupyter notebook interface? You're in the right place! This directory contains the classic widget-based notebooks for users who want the traditional Jupyter experience.

## 📚 Available Notebooks

### 1. **Unified_LoRA_Trainer.ipynb** - Complete Training Workflow
The all-in-one notebook for the full LoRA training pipeline:
- Environment setup and dependency installation
- Model downloads (SDXL, Flux, SD3, SD 1.5)
- Dataset preparation and management
- Training configuration and execution
- Post-training utilities

**Best for**: Users who want everything in one place

### 2. **Dataset_Preparation.ipynb** - Dataset Management
Focused notebook for dataset operations:
- Image upload and organization
- WD14 tagging with multiple tagger models
- Caption editing and trigger word management
- Gallery-DL scraping integration
- HuggingFace dataset uploads

**Best for**: Users who want to prepare datasets separately

### 3. **Utilities_Notebook.ipynb** - Post-Training Tools
Helper utilities for working with trained LoRAs:
- LoRA resizing (change network dimensions)
- Metadata editing
- HuggingFace model uploads
- Model inspection and analysis

**Best for**: Users who need utilities after training

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU (12GB+ VRAM recommended for SDXL)
- Git with LFS support
- Jupyter Lab or Jupyter Notebook

### Installation

1. **Clone Repository with Submodules**:
   ```bash
   git clone --recurse-submodules https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   cd Ktiseos-Nyx-Trainer
   ```

2. **Run Installer** (sets up backend):
   ```bash
   python installer.py
   ```

   This will:
   - Download and set up `trainer/derrian_backend` (Kohya sd-scripts)
   - Initialize git submodules
   - Create virtual environment
   - Install training dependencies

3. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter jupyterlab ipywidgets
   ```

4. **Start Jupyter**:
   ```bash
   # Option 1: Jupyter Lab (modern interface)
   jupyter lab

   # Option 2: Classic Jupyter Notebook
   jupyter notebook
   ```

5. **Open a Notebook**:
   - Navigate to `jupyter-notebooks/` directory
   - Open `Unified_LoRA_Trainer.ipynb` to get started

---

## 📖 Notebook Workflows

### Complete Training Workflow (Unified_LoRA_Trainer.ipynb)

1. **Run Setup Cell** - Installs backend if needed
2. **Download Models** - Get base models (SDXL, Flux, etc.) and VAEs
3. **Upload Dataset** - Upload your training images
4. **Tag Images** - Auto-generate captions with WD14 tagger
5. **Configure Training** - Set up LoRA parameters using interactive widgets
6. **Start Training** - Run training with real-time progress monitoring
7. **Download Results** - Get your trained LoRA files

### Dataset-Only Workflow (Dataset_Preparation.ipynb)

Perfect for preparing datasets that you'll train later (either in the web UI or another notebook):

1. **Upload Images** - Drag and drop or file browser
2. **Organize Folders** - Create concept folders
3. **Tag with WD14** - Multiple tagger model options
4. **Edit Captions** - Add trigger words, remove tags
5. **Export** - Save to disk or upload to HuggingFace

### Utilities Workflow (Utilities_Notebook.ipynb)

For working with already-trained LoRAs:

1. **Resize LoRA** - Change network dimensions (e.g., 32 → 16)
2. **Edit Metadata** - Update LoRA description and tags
3. **Upload to HF** - Share your LoRAs on HuggingFace
4. **Inspect Model** - View LoRA structure and parameters

---

## 🎨 Widget-Based Interface

The Jupyter notebooks use IPyWidgets for an interactive GUI experience:

- **Dropdown menus** for model selection
- **Sliders** for learning rates and training parameters
- **File browsers** for dataset selection
- **Progress bars** for real-time training monitoring
- **Tabbed interfaces** for organized parameter groups

### Example Widget Features:

**Training Configuration Widget**:
- 7 tabs: Project, Dataset, Basic, Learning Rates, LoRA Structure, Advanced, Sample Generation
- ~100 configurable parameters
- Real-time validation warnings
- Preset templates for common configurations

**Dataset Widget**:
- Upload via drag-and-drop or file browser
- WD14 tagger with 10+ model options
- Batch caption editing
- Gallery-DL integration for web scraping

**Monitor Widget**:
- Real-time training logs
- Progress bars for epochs and steps
- Loss graph visualization
- Sample image generation during training

---

## 🔧 Configuration

### Backend Setup

The notebooks use the same backend as the web UI (`trainer/derrian_backend`):

```
trainer/
└── derrian_backend/
    ├── sd_scripts/         # Kohya training scripts (submodule)
    ├── lycoris/           # LyCORIS support (submodule)
    └── install.sh         # Backend installer
```

**Important**: Run `python installer.py` before using notebooks for the first time!

### Directory Structure

```
Ktiseos-Nyx-Trainer/
├── jupyter-notebooks/          # Jupyter notebook workflow (you are here)
│   ├── Unified_LoRA_Trainer.ipynb
│   ├── Dataset_Preparation.ipynb
│   └── Utilities_Notebook.ipynb
├── widgets/                    # Widget implementations
├── core/                       # Manager classes
├── shared_managers.py         # Shared backend access
├── trainer/                   # Training backend
│   └── derrian_backend/
├── datasets/                  # Your training datasets
├── output/                    # Trained LoRA outputs
├── pretrained_model/          # Base models
└── vae/                       # VAE models
```

---

## 🆚 Jupyter vs Web UI

### When to Use Jupyter Notebooks:

✅ **You prefer Jupyter's interface** - Familiar environment for data scientists
✅ **You want step-by-step control** - Run cells individually
✅ **You're experimenting** - Easy to modify and test code inline
✅ **You need notebooks for documentation** - Markdown + code together
✅ **You're on a local machine** - No need for web server setup

### When to Use the Web UI:

✅ **You prefer modern web interfaces** - Clean, responsive design
✅ **You're deploying on servers** - VastAI, Docker, cloud instances
✅ **Multiple users need access** - Centralized training server
✅ **You want file browsing** - Built-in file manager
✅ **You need API access** - RESTful API for automation

**Both workflows use the same backend**, so you can switch between them anytime!

---

## 🐛 Troubleshooting

### "Backend not found" Error

**Cause**: `trainer/derrian_backend` doesn't exist

**Solution**:
```bash
python installer.py
```

### "Submodule empty" Error

**Cause**: Git submodules not initialized

**Solution**:
```bash
git submodule update --init --recursive
```

### "Widget not displaying" Error

**Cause**: IPyWidgets not enabled in Jupyter

**Solution**:
```bash
# Jupyter Lab
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Classic Jupyter
jupyter nbextension enable --py widgetsnbextension
```

### "CUDA out of memory" Error

**Cause**: Training batch size too large for your GPU

**Solution**: In training configuration widget:
- Reduce `train_batch_size` to 1 or 2
- Enable `gradient_checkpointing`
- Enable `cache_latents`
- Try FP8 training

### Upload Widget Stops Working

**Cause**: Race condition bug (fixed but may still occur)

**Solution**:
- Restart the notebook kernel
- Re-run the dataset widget cell
- If persists, use web UI's file manager instead

---

## 📊 Performance Tips

### Jupyter-Specific Optimizations

1. **Clear Output Regularly**: Long training logs can slow down notebooks
   ```python
   # Clear cell output
   from IPython.display import clear_output
   clear_output(wait=True)
   ```

2. **Use Logging to Files**: Redirect logs to files instead of notebook output
   ```python
   # Already handled in our widgets - logs go to /workspace/logs/
   ```

3. **Close Unused Notebooks**: Each notebook uses system memory

4. **Use `%time` Magic**: Benchmark training time
   ```python
   %time training_manager.start_training(config)
   ```

### Training Performance (Same as Web UI)

See main `docs/DEPLOYMENT.md` for GPU-specific optimization guides.

---

## 🔄 Migrating Between Workflows

### From Jupyter to Web UI

Your datasets and models are compatible! Just:
1. Start the web UI: `cd frontend && npm run dev`
2. Your datasets in `/datasets/` will appear in the file manager
3. Trained LoRAs in `/output/` are accessible

### From Web UI to Jupyter

Same compatibility:
1. Start Jupyter: `jupyter lab`
2. Open any notebook in `jupyter-notebooks/`
3. Your web UI datasets/models are accessible

**Shared State**: Both workflows use `shared_managers.py` for coordinated access to the backend.

---

## 📚 Additional Resources

- **Training Guide**: See `../docs/TRAINING_GUIDE.md` for LoRA training best practices
- **Deployment Guide**: See `../docs/DEPLOYMENT.md` for VastAI/Docker setup
- **API Documentation**: See `../api/README.md` for programmatic access
- **Kohya Documentation**: https://github.com/kohya-ss/sd-scripts
- **LyCORIS Documentation**: https://github.com/KohakuBlueleaf/LyCORIS

---

## 🆘 Support

- **GitHub Issues**: [Report bugs](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues)
- **Discussions**: [Community Q&A](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/discussions)
- **Original Colab**: For reference, see git history for the original single-notebook version

---

## ⚠️ Beta Notice

This project is in active development. Jupyter notebook workflow is fully functional but may receive fewer updates than the web UI. Both workflows share the same backend, so core functionality remains consistent.

**Report issues**: https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues
