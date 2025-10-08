# Getting Started: Your First LoRA

Welcome to LoRA Easy Training! This guide will walk you through the entire process of training your first LoRA using our widget-based notebooks. We'll go from installation to a finished, usable LoRA in just a few steps.

## Prerequisites

**Important**: This installation assumes you already have Jupyter Lab or Jupyter Notebook running on your system.

## 1. Installation

First, you need to get the project onto your machine and install the necessary dependencies.

1. **Clone the Repository**: Open a terminal or command prompt and run:
   ```bash
   git clone https://github.com/Ktiseos-Nyx/Lora_Easy_Training_Jupyter.git
   ```

2. **Run the Installer**: Navigate into the newly created directory and run the installer script:
   ```bash
   cd Lora_Easy_Training_Jupyter
   python installer.py
   ```
   This step might take a while, as it needs to download several gigabytes of data.

3. **Launch Jupyter**: Start your Jupyter environment:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

## 2. The Three-Notebook System

The project uses three specialized notebooks for different purposes:

### 📁 **Dataset_Preparation.ipynb** 
- **Purpose**: Prepare your training dataset
- **What it handles**: Upload images, visual curation with FiftyOne, auto-tag with WD14/BLIP, edit captions, add trigger words
- **Required**: Must complete this before training
- **Features**: Image deduplication, quality inspection, advanced tagging options

### 🚀 **Unified_LoRA_Trainer.ipynb**
- **Purpose**: Configure and run training
- **What it does**: Environment setup, universal training configuration (auto-detects SD1.5/SDXL/Flux), live monitoring
- **Smart Detection**: Automatically selects the correct Kohya training script based on your model

### 🛠️ **Utilities_Notebook.ipynb**
- **Purpose**: Post-training utilities and tools
- **What it handles**: LoRA resizing, file management, uploading to HuggingFace, testing tools, calculators

## 3. Your First LoRA: Step-by-Step

### Step 1: Prepare Your Dataset

1. **Open**: `Dataset_Preparation.ipynb`
2. **Run Cell 1A** (if needed): Environment setup validation (skip if already done)
3. **Run Cell 3**: Dataset Management Widget
4. **Prepare your images**:
   - Gather 15-30 high-quality images of your character/style
   - Create a ZIP file of your images OR prepare a folder
5. **Upload and curate**:
   - Upload your ZIP file or point to your folder using the widget
   - Use FiftyOne for visual curation (remove duplicates, inspect quality)
   - Clean and organize your dataset
6. **Auto-tag your images**:
   - Choose tagging method (WD14 for anime, BLIP for photos)  
   - Set threshold (0.35 for characters)
   - Run auto-tagging
7. **Add trigger word**:
   - Choose a unique trigger word (e.g., "saria_zelda", "mystyle_art")
   - Add it to all captions using the bulk edit tools
8. **Review and refine**: Check a few captions manually and edit if needed

### Pre-Training Checklist

Before moving to training, ensure you have:

**✅ Dataset Structure**
- [ ] Images are in a single folder
- [ ] All images have corresponding .txt caption files
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

1. **Open**: `Unified_LoRA_Trainer.ipynb`
2. **Run Cell 1**: Environment validation (validates installer.py setup)
3. **Run Cell 2**: Universal training configuration widget
4. **Configure basic settings**:
   - **Model Name**: Give your LoRA a unique name (e.g., "my_first_character")
   - **Base Model**: Choose your model file (system auto-detects SD1.5/SDXL/Flux/SD3)
   - **Dataset Path**: Point to your prepared dataset folder
   - **Trigger Word**: Same one you used in dataset prep
   - **Network Settings**: Start with 8 dim / 4 alpha
   - **Learning Rate**: 5e-4 for UNet, 1e-4 for Text Encoder
5. **Run Cell 3**: Training execution and real-time monitoring
6. **Monitor progress**: Watch the live training logs, loss curves, and progress bars
7. **Smart Detection**: System automatically selects the correct Kohya training script

### Step 3: Post-Training (Optional)

1. **Open**: `Utilities_Notebook.ipynb` for additional tools:
   - **LoRA Resizing**: Reduce file size while maintaining quality
   - **Upload to HuggingFace**: Share your LoRA with the community
   - **File Management**: Organize and backup your training outputs
   - **Training Calculator**: Plan future training sessions

### Step 4: Use Your LoRA

1. **Find your LoRA**: After training completes, your `.safetensors` file will be in the output folder
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
- **CUDA out of memory**: Reduce batch size to 1
- **Training too slow**: Check your target step count with the calculator
- **Poor results**: Review dataset quality and caption accuracy

## 5. Next Steps

Once you've successfully trained your first LoRA:

1. **Experiment with settings**: Try different optimizers (CAME, Prodigy)
2. **Advanced techniques**: Explore DoRA, LoKr, and other network types
3. **Larger datasets**: Scale up to 50-200 images for style LoRAs
4. **Share your work**: Upload to HuggingFace or Civitai using the utilities widget

**Congratulations on your first LoRA!** 🎉

---

*Need help? Check out our troubleshooting guide or join the Discord community.*