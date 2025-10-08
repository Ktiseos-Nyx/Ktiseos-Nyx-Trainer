# Installation Guide

This comprehensive guide covers the technical details of installing LoRA Easy Training Jupyter. For a quick setup, see the [Quickstart Guide](../quickstart.md).

## System Requirements

### Hardware Requirements

**Important Note**: These are **practical minimums** for reasonable training experiences. You CAN train on less, but expect significantly longer training times and extensive optimization requirements.

**System Memory:**
- **16GB RAM**: Practical minimum for stable training
- **32GB RAM**: Recommended for comfortable training without system slowdowns

**GPU Requirements - Practical Minimums:**

*For SD 1.5 LoRA Training:*
- **8GB VRAM**: RTX 3070, RTX 4060 Ti 8GB - Good training experience
- **6GB VRAM**: RTX 2060, RTX 3060 - Possible but requires optimization

*For SDXL LoRA Training:*
- **12GB VRAM**: RTX 3060 12GB, RTX 4060 Ti 16GB - Reasonable training times
- **8GB VRAM**: RTX 3070, RTX 4060 Ti 8GB - Requires significant optimization, much slower

*For Flux/SD3 Training:*
- **16GB+ VRAM**: RTX 4060 Ti 16GB minimum - Limited real-world data available

**GPU Platform Support:**
- **NVIDIA**: Best support with CUDA (recommended)
- **AMD**: ROCm (Linux) or DirectML/ZLUDA (Windows) - experimental
- **CPU Training**: Technically possible but impractically slow

**Storage Requirements:**
- **50GB+ Free Space**: For models, datasets, and training outputs
- **SSD Recommended**: Faster model loading and caching
- **Internet Connection**: Required for downloading models and dependencies

**Reality Check**: 
Results vary dramatically based on your settings. Someone training an 11-image LoRA on a 1070 8GB for a month is technically possible, but not a practical training experience. The settings and optimizations required make training painfully slow.

### Software Requirements

**Operating Systems:**
- **Linux**: Ubuntu 20.04+, CentOS 8+, or compatible distributions
- **Windows**: Windows 10/11 (64-bit) with WSL2 recommended for advanced features
- **macOS**: macOS 10.15+ (training limited to CPU, primarily for dataset preparation)

**Python Environment:**
- **Python Version**: 3.10+ required (Kohya and LyCORIS requirement)
- **Package Managers**: pip (included), uv (auto-installed for faster package resolution)
- **Jupyter**: JupyterLab or Jupyter Notebook required

**GPU Drivers and Compute Platforms:**

*NVIDIA GPUs:*
- **NVIDIA Driver**: 470.82.01+ (Linux) or 472.84+ (Windows)
- **CUDA Toolkit**: 11.8 or 12.1 (auto-detected and configured)
- **cuDNN**: 8.6+ (handled automatically with PyTorch installation)

*AMD GPUs:*
- **ROCm** (Linux): 5.4.2+ for native AMD support
- **ZLUDA** (Windows): Experimental CUDA-on-AMD translation layer
- **DirectML** (Windows): Microsoft's cross-platform ML acceleration

## Architecture Overview

### Project Structure

The installation creates this directory structure:

```
Lora_Easy_Training_Jupyter/
├── trainer/                    # Training environment
│   └── derrian_backend/       # Backend with Kohya integration
│       ├── sd_scripts/        # Kohya training scripts (submodule)
│       └── lycoris/           # LyCORIS advanced LoRA types (submodule)
├── pretrained_model/          # Downloaded base models
├── vae/                       # VAE files for improved quality
├── tagger_models/             # WD14/BLIP model cache
├── dataset/                   # Training datasets (created during use)
├── output/                    # Training outputs and LoRA files
├── logs/                      # Installation and training logs
└── docs/                      # Documentation system
```

### Dependency Management

**Core Training Stack:**
- **PyTorch**: GPU-accelerated deep learning framework (auto-configured for your hardware)
- **Transformers**: Hugging Face model library (4.44.0)
- **Diffusers**: Stable Diffusion pipeline library (0.25.0)
- **Safetensors**: Secure tensor format (0.4.4)
- **Accelerate**: Distributed training library (0.33.0)

**Specialized Optimizers:**
- **CAME**: Custom confidence-guided memory-efficient optimizer
- **Prodigy**: Learning rate-free optimization
- **DAdaptation**: Adaptive learning rate methods
- **Lion**: Evolved sign momentum optimizer

**Image Processing and Tagging:**
- **WD14 Taggers**: Anime/art content recognition (v3 models)
- **BLIP**: Natural image captioning
- **ONNX Runtime**: Accelerated inference for tagging (CUDA 12.1)
- **OpenCV**: Image processing (4.8.1.78)
- **FiftyOne**: Visual dataset curation

## Installation Process

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
cd Lora_Easy_Training_Jupyter

# Verify Python version
python --version  # Should be 3.10 or higher
```

### 2. Automated Installation

The installer performs comprehensive environment setup:

```bash
# Run the unified installer
python installer.py

# For verbose logging (useful for debugging)
python installer.py --verbose
```

**What the installer does:**

1. **Environment Detection**: Analyzes your system for GPU type, drivers, and Python environment
2. **Submodule Management**: Clones/updates the Derrian backend with Kohya scripts and LyCORIS
3. **System Dependencies**: Installs aria2c for fast model downloads (via package manager)
4. **Python Packages**: Installs 50+ specialized ML packages using uv (fast) or pip (fallback)
5. **GPU Configuration**: Auto-configures PyTorch for NVIDIA CUDA, AMD ROCm, or CPU-only
6. **ONNX Optimization**: Installs CUDA 12.1 ONNX runtime for 3-5x faster image tagging
7. **Platform Fixes**: Applies Windows-specific fixes for bitsandbytes and safetensors compilation

### 3. GPU-Specific Configuration

**NVIDIA GPU Setup:**
```bash
# Verify CUDA installation
nvidia-smi  # Check driver and CUDA version

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**AMD GPU Setup (Linux):**
```bash
# For ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Verify ROCm
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

**AMD GPU Setup (Windows):**
```bash
# For DirectML support (experimental)
pip install torch-directml

# For ZLUDA support (advanced users)
# Download ZLUDA from https://github.com/vosen/ZLUDA
# Follow ZLUDA installation instructions
```

### 4. Jupyter Environment

```bash
# Install Jupyter if not present
pip install jupyterlab notebook

# Launch Jupyter
jupyter lab
# or
jupyter notebook
```

## Technical Details

### Package Manager Strategy

The installer uses a two-tier package management approach:

1. **uv (Primary)**: Ultra-fast Rust-based package installer
   - 10-100x faster than pip for dependency resolution
   - Better conflict resolution and caching
   - Falls back to pip if uv fails or is unavailable

2. **pip (Fallback)**: Standard Python package installer
   - Universal compatibility
   - Used for complex packages that uv cannot handle
   - Force-installed for ONNX runtime with specific CUDA versions

### Memory and Performance Optimizations

**CUDA Memory Configuration:**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Allows dynamic VRAM allocation
- `CUDA_LAUNCH_BLOCKING=1`: Synchronous CUDA calls for debugging
- Automatic cuDNN benchmark selection for optimal performance

**System-Specific Optimizations:**

*Linux:*
- CUDA library path configuration for optimal GPU utilization
- ROCm environment variables for AMD GPU compatibility
- ONNX CUDA symlink creation for accelerated inference

*Windows:*
- Bitsandbytes DLL fixes for quantized training
- Safetensors Rust toolchain PATH configuration
- UV package manager workarounds for Windows-specific build issues

### Model and Data Management

**Automatic Downloads:**
- **aria2c**: Multi-connection downloads for 3-5x faster model acquisition
- **Hugging Face Hub**: Official model repository integration
- **Civitai API**: Community model discovery and download
- **Gallery-dl**: Dataset acquisition from image platforms

**Storage Optimization:**
- **Symlinked Models**: Shared model storage to prevent duplication
- **Cached Outputs**: Latent and text encoder caching for faster training
- **Progressive Downloads**: Resume interrupted downloads automatically

### Training Backend Architecture

**Kohya Integration:**
- **sd_scripts**: Core training algorithms for SD 1.5, SDXL, Flux, and SD3
- **Network Types**: LoRA, LoCon, LoKR, DoRA, LoHa, and advanced LyCORIS methods
- **TOML Configuration**: Type-safe configuration files for reproducible training

**Custom Extensions:**
- **CAME Optimizer**: Memory-efficient confidence-guided optimization
- **Custom Schedulers**: REX, polynomial, and advanced learning rate schedules
- **Multi-GPU Support**: Distributed training across multiple GPUs

## Verification Steps

After installation, verify your setup:

### 1. Environment Check
```bash
# In the project directory
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')

try:
    import transformers, diffusers, safetensors
    print('✅ Core ML libraries loaded successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

### 2. Notebook Access
```bash
# Launch Jupyter and verify notebook access
jupyter lab --no-browser --port=8888

# Open browser to: http://localhost:8888
# Navigate to: Dataset_Preparation.ipynb, Unified_LoRA_Trainer.ipynb, Utilities_Notebook.ipynb
```

### 3. Widget System Test
Open `Unified_LoRA_Trainer.ipynb` and run Cell 1 (Environment Setup). You should see:
- ✅ Python environment validation
- ✅ GPU detection and compatibility check
- ✅ Dependencies verification
- ✅ Backend initialization

## Installation Logging

The installer creates detailed logs in `logs/installer_[timestamp].log`:

**Log Contents:**
- System environment detection
- Package installation progress
- GPU configuration steps
- Error messages and resolution attempts
- Performance timing information

**Useful for:**
- Debugging installation failures
- Performance optimization
- Technical support requests
- Understanding your system configuration

## Next Steps

After successful installation:

1. **Start Training**: Follow the [Quickstart Guide](../quickstart.md) for your first LoRA
2. **Understand Parameters**: Review the [Parameters Guide](parameters.md) for training settings
3. **Dataset Preparation**: Learn about [Dataset Preparation](../dataset-guides/dataset_preparation.md)
4. **Troubleshooting**: Reference the [Troubleshooting Guide](troubleshooting.md) for common issues

## Platform-Specific Notes

### Windows Users
- **WSL2 Recommended**: For the best Linux-like experience
- **Git for Windows**: Required for proper submodule handling
- **Long Path Support**: Enable in Windows settings for deep directory structures
- **Antivirus Exclusions**: Add project directory to antivirus exclusions for performance

### Linux Users
- **Package Manager Access**: sudo privileges needed for system dependency installation
- **Graphics Drivers**: Ensure latest NVIDIA/AMD drivers are installed
- **Storage Permissions**: Verify write access to installation directory

### macOS Users
- **Homebrew Required**: For system dependency installation
- **GPU Training Limitations**: Training limited to CPU (very slow)
- **Primary Use Case**: Dataset preparation and notebook development