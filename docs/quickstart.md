# Getting Started: Your First LoRA

Welcome to Ktiseos Nyx LoRA Trainer! This guide will walk you through the entire process of training your first LoRA using our modern Next.js web interface. We'll go from installation to a finished, usable LoRA in just a few steps.

## Prerequisites

**System Requirements:**
- **GPU**: NVIDIA GPU with CUDA 12.1+ support (12GB+ VRAM recommended)
- **OS**: Windows, Linux, or macOS
- **Python**: 3.10 or 3.11
- **Node.js**: 20+ (for local development)

## 1. Installation

First, you need to get the project onto your machine and install the necessary dependencies.

1. **Clone the Repository**: Open a terminal or command prompt and run:
   ```bash
   git clone https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer.git
   ```

2. **Run the Installer**: Navigate into the newly created directory and run the installer script:
   ```bash
   cd Ktiseos-Nyx-Trainer
   python installer.py
   ```
   This step might take a while, as it needs to download several gigabytes of data.

3. **Start Services**: Run the start script to launch both frontend and backend:
   ```bash
   # For local development
   ./start_services_local.sh

   # Or for production (using built frontend)
   ./start.sh
   ```

4. **Access the Web UI**: Open your browser to `http://localhost:3000`

## 2. Web UI Architecture

The project uses a modern Next.js frontend with FastAPI backend, organized into several tabs:

### 📁 **Files Page**
- **Purpose**: File browser and management
- **What it handles**: Upload images, organize datasets, manage models
- **Features**: Drag & drop uploads, directory creation, file operations

### 🖼️ **Dataset Page**
- **Purpose**: Dataset preparation and management
- **What it handles**: Image tagging with WD14, caption editing, dataset validation
- **Features**: Batch tagging, caption review, dataset organization

### 🎓 **Training Page**
- **Purpose**: Configure and run training
- **What it handles**: 132 training parameters, model configuration, start/stop training
- **Features**: Real-time monitoring, config templates, parameter validation

### 🧪 **Utilities Page**
- **Purpose**: Post-training utilities and tools
- **What it handles**: Model conversion, upload to HuggingFace, file management

## 3. Your First LoRA: Step-by-Step

### Step 1: Prepare Your Dataset

1. **Navigate to**: **Dataset Page** in the web UI
2. **Prepare your images**:
   - Gather 15-30 high-quality images of your character/style
   - Ensure images are in common formats (JPG/PNG)
3. **Upload images**:
   - Drag and drop your images directly into the upload area
   - Or click to select files from your computer
   - Images will be organized into a new dataset folder
4. **Auto-tag your images**:
   - Click the **"Auto-Tag"** button
   - Select **"WD14 Tagger"** for anime/illustration or **"BLIP"** for photos
   - Set appropriate threshold (0.35 for characters)
   - Review and edit tags as needed
5. **Add trigger word**:
   - Add a unique trigger word to all captions (e.g., "saria_zelda", "mystyle_art")
   - Use the bulk edit tools to apply to all images

### Pre-Training Checklist

Before moving to training, ensure you have:

**✅ Dataset Structure**
- [ ] Images are uploaded to a dedicated folder
- [ ] All images have corresponding caption files
- [ ] No corrupted or unreadable images
- [ ] Consistent image format (jpg/png)

**✅ Caption Quality**
- [ ] All captions contain your trigger word
- [ ] Tags are accurate and relevant
- [ ] No unwanted or problematic tags
- [ ] Caption length is reasonable (50-200 tokens)

**✅ Content Verification**
- [ ] Images represent what you want to train
- [ ] Sufficient variety in poses/angles
- [ ] Consistent quality across dataset
- [ ] No duplicate or near-duplicate images

### Step 2: Train Your LoRA

1. **Navigate to**: **Training Page** in the web UI
2. **Load a config template**: Use the default "SDXL LoRA" or "SD1.5 LoRA" template
3. **Configure basic settings**:
   - **Model Name**: Give your LoRA a unique name (e.g., "my_first_character")
   - **Base Model**: Choose your model file (auto-detects SD1.5/SDXL/Flux/SD3)
   - **Dataset Path**: Select your prepared dataset folder
   - **Trigger Word**: Same one you used in dataset prep
   - **Network Settings**: Start with 8 dim / 4 alpha
   - **Learning Rate**: 5e-4 for UNet, 1e-4 for Text Encoder
4. **Review advanced settings** (optional):
   - Adjust batch size based on your VRAM
   - Enable gradient checkpointing if memory-constrained
5. **Click "Start Training"**: Monitor progress in real-time
6. **Monitor progress**: Watch the live training logs, loss curves, and progress indicators

### Step 3: Post-Training

1. **Wait for completion**: Training will automatically stop when finished
2. **Find your LoRA**: Navigate to **Files Page** → `/output` folder
3. **Download**: Click the download icon next to your `.safetensors` file
4. **Upload to HuggingFace**: Use the **Utilities Page** → "Upload to HuggingFace" tool

### Step 4: Use Your LoRA

1. **Find your LoRA**: The `.safetensors` file will be in your output folder
2. **Install in your SD UI**: Copy the file to your Stable Diffusion web UI's `models/lora` directory
3. **Test generation**: Use your trigger word in prompts to activate the LoRA
4. **Optional**: See our [Testing LoRAs guide](guides/testing-loras.md) for setting up Automatic1111/Forge

## 4. Quick Tips for Success

### Dataset Quality
- **Image quality matters**: Use high-resolution, clear images
- **Variety is key**: Different poses, expressions, angles
- **Consistent style**: For style LoRAs, maintain visual consistency

### Training Settings
- **Start simple**: Use default settings for your first LoRA
- **Monitor loss**: Watch for steady decrease in training loss
- **Don't overtrain**: Stop if loss plateaus or starts increasing

### Common Issues
- **CUDA out of memory**: Reduce batch size to 1, enable gradient checkpointing
- **Training too slow**: Check your target step count in the calculator
- **Poor results**: Review dataset quality and caption accuracy

## 5. Next Steps

Once you've successfully trained your first LoRA:

1. **Experiment with settings**: Try different optimizers (CAME, Prodigy)
2. **Advanced techniques**: Explore DoRA, LoKr, and other network types
3. **Larger datasets**: Scale up to 50-200 images for style LoRAs
4. **Share your work**: Upload to HuggingFace or Civitai using the utilities tools

**Congratulations on your first LoRA!** 🎉

---

*Need help? Check out our troubleshooting guide or join the Discord community.*