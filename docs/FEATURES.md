# Feature Documentation

Complete guide to all features in Ktiseos Nyx LoRA Trainer.

## Table of Contents

- [Architecture](#architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Training Execution](#training-execution)
- [Post-Training Utilities](#post-training-utilities)
- [Model Management](#model-management)
- [Platform Support](#platform-support)

## Architecture

### Tech Stack

**Frontend:**
- Next.js 15 with React 19
- TypeScript 5+
- Tailwind CSS v4
- shadcn/ui components
- TanStack Query for state management

**Backend:**
- FastAPI with Python 3.10+
- Lazy-loaded manager system (dependencies load on-demand)
- WebSocket support for real-time updates
- RESTful API with automatic OpenAPI docs

**Training Backend:**
- Kohya SS (sd-scripts) - vendored
- LyCORIS integration - vendored
- ONNX runtime for optimized inference

**File Management:**
- Built-in file browser with tree view
- Uppy.js for drag-and-drop uploads
- Multi-file operations

## Dataset Preparation

### Image Upload & Management

**Upload Methods:**
- Drag-and-drop interface
- File browser selection
- Batch upload support (hundreds of images at once)
- Archive extraction (.zip, .tar, .tar.gz, .rar)

**Supported Formats:**
- Images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- Archives: `.zip`, `.tar`, `.tar.gz`, `.rar`

**Features:**
- Automatic dataset organization
- Image preview gallery
- File validation and error reporting
- Progress tracking for large uploads

### Auto-Tagging (WD14)

**Tagger Models:**
- SwinV2 (default) - Balanced accuracy/speed
- ConvNext - High accuracy, slower
- ConvNextV2 - Latest architecture
- ViT - Vision Transformer based
- MOAT - Mobile-optimized

**Configuration Options:**
- **Threshold**: 0.0-1.0 (default: 0.35) - Controls tag sensitivity
- **Character Threshold**: Separate threshold for character tags
- **Batch Size**: 1-32 (adjust based on VRAM)
- **Caption Extension**: `.txt` or `.caption`
- **Remove Underscore**: Clean tag formatting
- **Undesired Tags**: Filter out unwanted tags
- **Recursive**: Process subdirectories

**Advanced Features:**
- ONNX runtime optimization for 2-3x faster tagging
- V3 tagger support with enhanced accuracy
- Real-time progress monitoring via WebSocket
- Batch processing with automatic retry

**Usage:**
1. Navigate to Dataset → Auto-Tag
2. Select dataset from dropdown
3. Choose tagger model and configure thresholds
4. Click "Start Tagging"
5. Monitor progress in real-time
6. Review generated captions in tag editor

### Caption Editing

**Tag Editor Interface:**
- Image gallery with thumbnails
- Side-by-side image and caption view
- Batch editing operations
- Search and filter functionality

**Editing Features:**
- **Add Trigger Words**: Prepend custom tokens to all captions
- **Find & Replace**: Bulk text replacement
- **Tag Sorting**: Alphabetical or by frequency
- **Tag Cleanup**: Remove duplicates, trailing commas
- **Batch Operations**: Apply changes to multiple images

**Workflow:**
1. Navigate to Dataset → Edit Tags
2. Select dataset
3. Browse images in gallery
4. Edit captions inline or in batch
5. Save changes (auto-saved)

### BLIP/GIT Captioning (Alternative to WD14)

**Captioning Models:**
- BLIP - Natural language descriptions
- GIT - Image-to-text generation

**Use Cases:**
- Natural language captions instead of booru tags
- Descriptive training for specific styles
- Caption augmentation

## Training Configuration

### Configuration Interface

**Organization:**
- 132+ parameters organized into 7 tabs
- Real-time validation with Zod schemas
- Persistent state across sessions (Zustand)
- Preset management (save/load configs)

### Tab Overview

#### 1. Project Setup
- Project name
- Model selection (SD1.5, SDXL, Flux, SD3, Lumina)
- VAE selection
- Output directory configuration
- Training precision (fp16, bf16, fp32)

#### 2. Dataset Configuration
- Dataset selection
- Batch size
- Resolution (512, 768, 1024, custom)
- Bucket settings (multi-resolution training)
- Data augmentation options

#### 3. LoRA Structure
- **Network Type**: LoRA, LoCon, LoHa, LoKr, DoRA
- **Network Rank**: 4, 8, 16, 32, 64, 128
- **Network Alpha**: Auto or custom
- **Conv Settings**: Conv rank, alpha, dropout

#### 4. Learning Rate & Optimizer
- **Learning Rate**: Base LR, UNET LR, Text Encoder LR
- **Optimizer**: AdamW, AdamW8bit, Lion, Prodigy, CAME, REX, AdaFactor
- **Scheduler**: Constant, Linear, Cosine, Polynomial
- **Warmup Steps**: Gradual LR increase

#### 5. Performance & Memory
- **Mixed Precision**: fp16, bf16
- **Gradient Checkpointing**: Reduce VRAM usage
- **xFormers**: Enable memory-efficient attention
- **Cache Latents**: Speed up training
- **Gradient Accumulation**: Effective batch size increase

#### 6. Advanced Settings
- **Noise Offset**: 0.0-0.2 (improves dark/light generation)
- **Min SNR Gamma**: Signal-to-noise ratio weighting
- **Dropout**: Prevent overfitting
- **Max Norm**: Gradient clipping
- **LR Schedulers**: Warmup, decay strategies
- **Sample Generation**: Auto-generate samples during training

#### 7. Saving & Checkpoints
- **Save Frequency**: Every N steps/epochs
- **Save Format**: safetensors, ckpt
- **Save Precision**: fp16, bf16, fp32
- **Keep Last N**: Automatic checkpoint cleanup
- **Save State**: Resume training capability

### Supported Model Architectures

| Architecture | Status | Notes |
|--------------|--------|-------|
| **SD 1.5** | ✅ Stable | Fully supported, well-tested |
| **SDXL** | ✅ Stable | 24GB VRAM recommended |
| **Flux.1** | ⚠️ Experimental | Requires latest Kohya backend |
| **SD3/SD3.5** | ⚠️ Experimental | Requires sd3 branch |
| **Lumina** | ⚠️ Experimental | Basic support |
| **Chroma** | ⚠️ Experimental | Minimal testing |

### LoRA Type Comparison

| Type | Rank Options | Parameters | Speed | Quality | Use Case |
|------|--------------|------------|-------|---------|----------|
| **LoRA** | 4-128 | Low | Fast | Good | General purpose |
| **LoCon** | 4-128 | Medium | Medium | Better | Improved details |
| **LoHa** | 4-128 | Medium | Medium | Better | Hadamard product |
| **LoKr** | 4-128 | High | Slow | Best | Kronecker factorization |
| **DoRA** | 4-128 | Highest | Slowest | Best | Magnitude + direction |

### Optimizer Comparison

| Optimizer | VRAM | Speed | Quality | Notes |
|-----------|------|-------|---------|-------|
| **AdamW** | High | Fast | Good | Standard choice |
| **AdamW8bit** | Low | Fast | Good | 8-bit quantization |
| **Lion** | Medium | Fast | Good | Memory efficient |
| **Prodigy** | Medium | Medium | Best | Adaptive LR |
| **CAME** | Low | Medium | Better | Confidence-guided |
| **REX** | Medium | Medium | Better | Regularized |
| **AdaFactor** | Low | Fast | Good | Memory optimized |

## Training Execution

### TOML Configuration Generation

The UI generates two TOML files for Kohya SS:

**`dataset.toml`** - Dataset configuration:
```toml
[[datasets]]
resolution = 1024
batch_size = 1

[[datasets.subsets]]
image_dir = '/workspace/datasets/my_dataset'
num_repeats = 10
```

**`config.toml`** - Training hyperparameters:
```toml
[model_arguments]
pretrained_model_name_or_path = "/workspace/models/model.safetensors"
vae = "/workspace/vae/vae.safetensors"

[network_arguments]
network_module = "lycoris.kohya"
network_dim = 32
network_alpha = 16
```

### Real-Time Monitoring

**WebSocket Progress Updates:**
- Current epoch/step
- Loss values
- ETA (estimated time remaining)
- VRAM usage
- Training speed (it/s)

**Log Streaming:**
- Live console output
- Error detection and highlighting
- Scrollable log viewer
- Download logs for debugging

### Training Calculator

Automatic step/epoch calculations:

**Inputs:**
- Number of images
- Repeats per image
- Batch size
- Desired total steps

**Outputs:**
- Required epochs
- Total steps
- Approximate training time
- VRAM usage estimate

## Post-Training Utilities

### LoRA Resizing

**Resize Options:**
- Target rank: 4, 8, 16, 32, 64, 128
- Precision: fp16, bf16, fp32
- Algorithm: SVD (Singular Value Decomposition)

**Use Cases:**
- Reduce file size for distribution
- Lower VRAM usage during inference
- Trade quality for speed

### Metadata Editing

**Editable Fields:**
- Model name
- Author
- Description
- Tags
- Training parameters
- Trigger words

### HuggingFace Integration

**Upload Features:**
- Dataset upload to HuggingFace Hub
- Model upload with metadata
- Automatic README generation
- Version control

## Model Management

### Model Downloads

**Supported Sources:**
- HuggingFace Hub (direct links)
- Civitai (model IDs or URLs)

**Download Types:**
- Base models (SD1.5, SDXL, etc.)
- VAE files
- LoRA models for reference

**Features:**
- Progress tracking
- Automatic file naming
- Integrity verification
- Resume partial downloads

### Popular Models (Built-in Links)

**SD1.5:**
- RunwayML SD 1.5
- Anything V3

**SDXL:**
- Stable Diffusion XL 1.0
- Pony Diffusion V6 XL

**VAE:**
- SD1.5 VAE
- SDXL VAE

## Platform Support

### Cross-Platform Compatibility

| Feature | Windows | Linux | macOS | VastAI |
|---------|---------|-------|-------|--------|
| **Web UI** | ✅ | ✅ | ✅ | ✅ |
| **Training** | ✅ | ✅ | ❌ | ✅ |
| **Auto-Tag** | ✅ | ✅ | ✅ | ✅ |
| **File Upload** | ✅ | ✅ | ✅ | ⚠️ |

> Note: File upload on VastAI is currently limited - use Jupyter for dataset uploads.

### Cloud Deployment Features

**VastAI:**
- One-click deployment via template
- Automatic setup with `vastai_setup.sh`
- Supervisor-managed auto-restart
- Port mapping (3000→13000, 8000→18000)

**RunPod:**
- Experimental support
- Manual setup required
- Use Linux installation scripts

## Coming Soon

Features in development:

- [ ] Improved file upload for VastAI
- [ ] Stable Flux training support
- [ ] SD3.5 UI integration
- [ ] Advanced sampling during training
- [ ] Multi-LoRA merging
- [ ] Training resume from UI
- [ ] Dataset preprocessing pipeline
- [ ] Automatic hyperparameter tuning

---

**Want a feature?** [Open a feature request](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer/issues/new?labels=enhancement)
