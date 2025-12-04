'use client';

import { useState } from 'react';
import {
  Home,
  BookOpen,
  ExternalLink,
  ChevronRight,
  Cpu,
  Zap,
  Database,
  Settings,
  Monitor,
  Upload,
  Bug,
  Wrench,
} from 'lucide-react';
import Breadcrumbs from '@/components/Breadcrumbs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

interface DocSubsection {
  id: string;
  title: string;
  content: string;
  links?: { label: string; url: string }[];
}

interface DocSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  content?: string;
  links?: { label: string; url: string }[];
  subsections?: DocSubsection[];
}

const docSections: DocSection[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    icon: <Zap className="w-4 h-4" />,
    content: `Welcome to Ktiseos-Nyx Trainer! This guide will help you get started with training your first LoRA model.

**Prerequisites:**
- A dataset of images (20-100+ images recommended)
- A base model downloaded (SDXL, SD 1.5, Flux, or SD 3.5)
- (Optional) A VAE file for improved quality

**Quick Start Workflow:**
1. Download a base model from the Models & VAEs page
2. Upload your dataset (images in a folder)
3. Auto-tag your images using WD14 tagger
4. Configure training parameters
5. Start training and monitor progress
6. Download your trained LoRA`,
    links: [
      { label: 'VastAI Setup Guide', url: 'https://vast.ai/docs' },
      { label: 'Kohya Documentation', url: 'https://github.com/kohya-ss/sd-scripts' },
    ],
  },
  {
    id: 'hardware-requirements',
    title: 'Hardware Requirements',
    icon: <Cpu className="w-4 h-4" />,
    content: `Understanding hardware requirements helps you choose the right training mode and optimize your setup.

**GPU Memory Requirements (from Kohya documentation):**

**LoRA Training:**
- SD 1.5: 8GB minimum (10GB recommended)
- SDXL: 10GB minimum (16GB+ recommended)
- Flux: 16GB+ recommended
- SD3: 16GB+ recommended

**Full Checkpoint/Model Training:**
- SD 1.5: 16-24GB minimum
- SDXL: 24GB minimum (often requires multi-GPU)
- Flux: 40GB+ recommended
- SD3: 40GB+ recommended

**Memory Optimization Options:**
- \`gradient_checkpointing\`: Trade compute for memory (slower but uses less VRAM)
- \`cache_latents\`: Cache VAE outputs to save memory during training
- \`mixed_precision\`: Use fp16/bf16 instead of fp32 (BF16 recommended for A100/H100)
- \`gradient_accumulation\`: Simulate larger batch sizes with less memory
- \`blocks_to_swap\`: Swap transformer blocks to CPU (Flux/SD3)
- Lower \`dim\` values: 4-8 for 8GB GPUs, 16-32 for 12GB+

**Storage Requirements:**
- LoRA models: 10-300 MB per save
- Checkpoints: 2-7 GB per save
- Working space: 50-100 GB recommended for checkpoint training

**CPU/RAM:**
- 16GB+ system RAM recommended
- Modern multi-core CPU (dataset preprocessing benefits from multiple cores)

**Recommended GPUs:**
- Budget: RTX 3060 12GB (LoRA only)
- Mid-range: RTX 4070 Ti 12GB, RTX 4080 16GB
- High-end: RTX 4090 24GB, A100 40/80GB, H100 80GB
- Cloud: VastAI, RunPod, Lambda Labs for pay-per-hour access`,
    links: [
      { label: 'Kohya SDXL Training Guide', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_SDXL-en.md' },
      { label: 'VastAI GPU Marketplace', url: 'https://vast.ai/' },
    ],
  },
  {
    id: 'lora-types',
    title: 'Network Algorithms',
    icon: <Wrench className="w-4 h-4" />,
    subsections: [
      {
        id: 'lora-standard',
        title: 'Standard LoRA',
        content: `The conventional LoRA algorithm with convolution support.

**Trigger:** \`algo=lora\` or \`algo=locon\` (alias)

**Description:**
- Most common and widely supported algorithm
- Includes Conv layer implementation from LoCon
- Linear layers: $\\Delta W = B \\cdot A$ with $rank(\\Delta W) \\leq dim$
- Good for characters, concepts, and general fine-tuning

**Recommended Settings:**
- \`dim\`: 4-64 (32 is common default)
- \`alpha\`: 1 to half dimension (or same as dim)
- Works well for most use cases

**File Size:** Small to medium (10-150MB depending on dim)

**Best For:** General purpose, characters, concepts, styles`,
        links: [
          { label: 'LoRA Paper', url: 'https://arxiv.org/abs/2106.09685' },
          { label: 'LyCORIS Algo Details', url: 'https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md' },
        ],
      },
      {
        id: 'lora-loha',
        title: 'LoHa (Hadamard Product)',
        content: `LoRA with Hadamard product for higher effective rank.

**Trigger:** \`algo=loha\`

**Description:**
- Uses Hadamard product: $\\Delta W = B \\odot A$
- Can achieve $rank(\\Delta W) \\leq rank(B) \\times rank(A)$ (rank squared!)
- Conventional LoRA is limited to $rank(\\Delta W) \\leq dim$
- Originally designed for federated learning

**Recommended Settings:**
- \`dim\`: 8-32 (use lower dim than LoRA)
- \`alpha\`: 1 to half dimension
- **WARNING**: High dim may cause unstable loss or NaN - use lower LR!

**Characteristics:**
- **Strongest dampening effect** among LoRA types
- Better generalization ability
- Suitable for easier concepts and multi-concept training
- Custom backward pass to save memory

**File Size:** Small to medium

**Best For:** Multi-concept training, style preservation, generalization`,
        links: [
          { label: 'FedPara Paper', url: 'https://arxiv.org/abs/2108.06098' },
        ],
      },
      {
        id: 'lora-lokr',
        title: 'LoKr (Kronecker Product)',
        content: `Most parameter-efficient LoRA using Kronecker product.

**Trigger:** \`algo=lokr\`

**Description:**
- Uses Kronecker product for extreme parameter efficiency
- Rank is multiplicative: $rank(\\Delta W) \\leq min(a,b) \\times r$
- Can achieve square root parameter reduction
- Two modes: Small LoKr (default) and Large LoKr

**Recommended Settings:**

**Small LoKr (Default):**
- \`factor=-1\`: Automatic factorization (smallest decomposition)
- File size: 900KB - 2.5MB
- Very parameter efficient
- May be harder to transfer between models

**Large LoKr (LoRA-like):**
- \`factor~8\`: Manual factor setting
- \`dim\`: Set very high (like 10000) for full dimension
- Alpha is ignored in full dimension mode
- Similar behavior to standard LoRA

**Characteristics:**
- Most parameter-efficient among all methods
- Can use \`decompose_both=True\` to decompose both matrices
- Results may be model-specific (harder to transfer)

**File Size:** Tiny to medium (under 1MB to 150MB)

**Best For:** Minimal file size, embedded models, experimental training`,
        links: [
          { label: 'KronA Paper', url: 'https://arxiv.org/abs/2212.10650' },
        ],
      },
      {
        id: 'lora-full',
        title: 'Native Fine-Tuning (Full)',
        content: `Full weight matrix training without decomposition.

**Trigger:** \`algo=full\`

**Description:**
- Also known as DreamBooth when used for concept training
- Trains full weight matrices instead of low-rank decomposition
- No rank limitation - highest quality potential
- Most flexible but also most sensitive to hyperparameters
- Can merge and change base models similar to LoRA

**Characteristics:**
- **File Size**: Very large (several GB) - trains full weights
- **Quality**: Highest potential quality (no decomposition losses)
- **Flexibility**: Can capture very complex concepts
- **Sensitivity**: More prone to overtraining, requires careful tuning
- **VRAM**: Requires more memory than LoRA methods

**Recommended Settings:**
- Lower learning rate than LoRA (1e-6 to 5e-6 common)
- Fewer training steps to avoid overtraining
- Monitor closely for overtraining signs
- Best for when LoRA methods aren't capturing enough detail

**Best For:** Maximum quality when LoRA isn't sufficient, complex concepts requiring full expressiveness`,
        links: [
          { label: 'DreamBooth Paper', url: 'https://arxiv.org/abs/2208.12242' },
          { label: 'LyCORIS Algo Details', url: 'https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md' },
        ],
      },
      {
        id: 'lora-ia3',
        title: '(IA)^3',
        content: `Infused Adapter by Inhibiting and Amplifying Inner Activations.

**Trigger:** \`algo=ia3\`

**Description:**
- Extremely parameter-efficient method
- Learns scaling factors for activations instead of weight updates
- Special case of Diag-OFT algorithm
- Different training approach than traditional LoRA

**Characteristics:**
- **File Size**: Extremely tiny (under 1MB!)
- **Learning Rate**: Requires very high LR (5e-3 to 1e-2, much higher than LoRA)
- **Quality**: Good at learning styles
- **Transferability**: Hard to transfer between different base models
- **Network Args**: No dim/alpha needed - uses different parameter structure

**Recommended Settings:**
- \`learning_rate\`: 5e-3 to 1e-2 (10-100x higher than LoRA!)
- Standard LoRA learning rates will not work
- Monitor training carefully due to high learning rate
- Experiment with different LR values

**Trade-offs:**
- Minimal file size vs limited transferability
- Fast training vs base model dependency
- Style learning vs detail capture

**Best For:** Extremely small file sizes, style training, experimental use`,
        links: [
          { label: '(IA)^3 Paper', url: 'https://arxiv.org/abs/2205.05638' },
          { label: 'LyCORIS Algo Details', url: 'https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md' },
        ],
      },
      {
        id: 'lora-dylora',
        title: 'DyLoRA',
        content: `Dynamic LoRA with flexible rank resizing.

**Trigger:** \`algo=dylora\`

**Description:**
- Train once at high rank, resize to any lower rank afterward
- Updates one row/column per training step (dynamic approach)
- Enables rank flexibility without retraining
- Requires more training steps than standard LoRA

**How It Works:**
- Train with high \`dim\` value (e.g., 128)
- During training, updates different subsets of the rank
- After training, can resize down to any target rank (e.g., 64, 32, 16, 8)
- No need to retrain for different rank sizes

**Recommended Settings:**
- \`dim\`: High value like 128 (train at max rank you might need)
- \`alpha\`: dim/4 to dim (experiment in this range)
- \`block_size\`: Update multiple rows/cols per step for faster training
- **More training steps**: Needs more steps than regular LoRA due to dynamic updates

**Workflow:**
1. Train with high dim (128)
2. Complete training
3. Resize to target dim as needed (no retraining required)
4. Test different ranks to find best size/quality trade-off

**Advantages:**
- Flexibility to resize without retraining
- One training run supports multiple rank sizes
- Can find optimal rank after training

**Trade-offs:**
- Slower training (more steps needed)
- Larger initial file size
- Requires resizing step for smaller sizes

**Best For:** When you want flexibility to test different ranks, uncertainty about optimal rank size`,
        links: [
          { label: 'DyLoRA Paper', url: 'https://arxiv.org/abs/2210.07558' },
          { label: 'LyCORIS Algo Details', url: 'https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md' },
        ],
      },
      {
        id: 'lora-diag-oft',
        title: 'Diag-OFT',
        content: `Diagonal Orthogonal Fine-Tuning.

**Trigger:** \`algo=diag-oft\`

**Description:**
- Orthogonal fine-tuning preserving hyperspherical energy
- Different mathematical approach than LoRA (orthogonal matrices)
- According to paper, converges faster than LoRA
- Preserves certain geometric properties during training

**Key Concepts:**
- Uses orthogonal transformations instead of low-rank decomposition
- Preserves norm and angular relationships
- \`dim\` parameter = block size (different meaning than LoRA)
- Mathematical guarantees about training stability

**Recommended Settings:**
- \`dim\`: Block size for orthogonal blocks
- \`constraint\`: Set for COFT (Constrained OFT) variant
- \`rescaled\`: Set for rescaled OFT variant
- Experiment with constraint/rescaled options

**Variants:**
- **Standard Diag-OFT**: Basic diagonal orthogonal fine-tuning
- **COFT**: Constrained variant with additional restrictions
- **Rescaled OFT**: Rescaled variant for different scale handling

**Characteristics:**
- Faster convergence than LoRA (per paper)
- Different training dynamics
- Preserves geometric properties
- Less widely tested than standard LoRA

**Best For:** Experimental use, faster convergence requirements, when geometric preservation matters`,
        links: [
          { label: 'Diag-OFT Paper', url: 'https://arxiv.org/abs/2306.07280' },
          { label: 'LyCORIS Algo Details', url: 'https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md' },
        ],
      },
      {
        id: 'lora-boft',
        title: 'BOFT',
        content: `Butterfly Orthogonal Fine-Tuning.

**Trigger:** \`algo=boft\`

**Description:**
- Advanced variant of Diag-OFT using butterfly operation
- Full orthogonal matrix capability (more expressive than Diag-OFT)
- Sits between Diag-OFT and full OFT in terms of power
- Uses butterfly factorization for efficiency

**How It Differs from Diag-OFT:**
- More expressive than diagonal-only approach
- Butterfly structure allows full orthogonal matrices
- Higher capacity than Diag-OFT
- Still more efficient than full OFT

**Recommended Settings:**
- \`dim\`: Controls butterfly block structure
- \`constraint\`: Enable for constrained BOFT variant
- \`rescaled\`: Enable for rescaled variant
- Supports same variants as Diag-OFT (standard, constrained, rescaled)

**Characteristics:**
- More powerful than Diag-OFT
- More efficient than full OFT
- Butterfly structure provides good expressiveness/efficiency trade-off
- Newer algorithm with less community testing

**Power Hierarchy:**
- Diag-OFT < BOFT < Full OFT
- (Less expressive → More expressive)
- (More efficient → Less efficient)

**Best For:** When Diag-OFT isn't expressive enough but full OFT is too expensive, experimental advanced training`,
        links: [
          { label: 'BOFT Paper', url: 'https://arxiv.org/abs/2311.06243' },
          { label: 'LyCORIS Algo Details', url: 'https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md' },
        ],
      },
    ],
  },
  {
    id: 'datasets',
    title: 'Dataset Preparation',
    icon: <Database className="w-4 h-4" />,
    subsections: [
      {
        id: 'datasets-image-requirements',
        title: 'Image Requirements',
        content: `Technical requirements and options for training images.

**Note:** Our web interface helps organize your dataset automatically through the Dataset page. This documentation covers technical details if you prefer direct control.

**Supported Formats:**
- JPG, JPEG, PNG, WebP, BMP
- Most common image formats are supported
- Transparent PNGs work but alpha channel is ignored during training

**Resolution & Aspect Ratios:**

**Aspect Ratio Bucketing (Recommended):**
- Enable with \`enable_bucket = true\` in TOML config
- **You don't need square images** - portraits, landscapes, and squares all work!
- System automatically groups images by similar aspect ratios into buckets
- Common buckets: 1:1 (square), 3:4 (portrait), 4:3 (landscape), 16:9 (widescreen)
- Configure bucket ranges:
  - \`min_bucket_reso\`: Minimum resolution (default: 256)
  - \`max_bucket_reso\`: Maximum resolution (default: 1024 for SD 1.5, 2048 for SDXL)
  - \`bucket_reso_steps\`: Step size for buckets (default: 64, SDXL minimum: 32)
- No need to crop or resize your images to exact dimensions!

**Model-Specific Base Resolutions:**
- **SD 1.5**: Base ~512px
  - Examples: 512x512, 512x768, 768x512, 640x512
  - Trained on 512x512, works well with similar resolutions
- **SDXL**: Base ~1024px
  - Examples: 1024x1024, 1024x1536, 768x1024, 1152x896
  - Trained on 1024x1024, handles higher resolutions better
- **Flux**: Base ~1024px
  - Similar to SDXL, handles 1024+ resolutions
  - Can train at higher resolutions (1280x1280+) with sufficient VRAM
  - More flexible with aspect ratios than SDXL
- **SD 3 / SD 3.5**: Base ~1024px
  - Similar resolution handling to SDXL
  - Good aspect ratio flexibility

**Image Size Guidelines:**
- Very small images (<256px shortest side): Consider upscaling first
- Very large images (4000px+): May want to downsize for training efficiency
- Bucketing handles mixed sizes automatically within configured ranges

**Random Crop Alternative:**
- Use \`random_crop = true\` in dataset subset config
- Randomly crops images to target resolution instead of bucketing
- Useful for datasets with very inconsistent framing
- Can add variety by showing different parts of images each epoch
- Trade-off: May cut off important parts of images

**Augmentation Options:**
- \`flip_aug\`: Random horizontal flip (good for symmetric subjects)
- \`color_aug\`: Color jittering augmentation
- \`random_crop\`: Crop to resolution instead of resize
- Configure per-subset in TOML config`,
        links: [
          { label: 'Kohya sd-scripts Dataset Config', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md' },
        ],
      },
      {
        id: 'datasets-folder-organization',
        title: 'Folder Organization',
        content: `How to structure your dataset directories.

**Basic Structure:**
- Create a directory for your dataset (e.g., \`my_character_dataset/\`)
- Place all training images in this directory
- Caption files (.txt or .caption) go next to each image with matching filename
- Optionally use subfolders to organize different concepts or image types

**Flat Directory (Simplest):**
All images and captions in one folder:
- \`dataset/\`
  - \`image_001.png\`
  - \`image_001.txt\`
  - \`image_002.jpg\`
  - \`image_002.txt\`
  - etc.

**Subfolder Organization:**
- Use subfolders to organize images by concept, pose, outfit, etc.
- Enable \`--recursive\` flag in captioning scripts to process subfolders
- Example structure:
  - \`dataset/\`
    - \`portraits/\` (images and captions)
    - \`full_body/\` (images and captions)
    - \`action_poses/\` (images and captions)

**Important Notes:**
- Repeat counts are configured in TOML file, NOT in folder names
- Don't use folder naming conventions like \`10_character_name\` (not supported yet)
- Caption files must be in same directory as their corresponding images
- Subfolder names don't affect training (they're just for organization)

**Multiple Datasets:**
You can configure multiple dataset directories in your TOML config:
- Each dataset can have different resolution settings
- Each dataset can have different batch sizes
- Each dataset can have different repeat counts
- Useful for training on multiple resolutions or concepts simultaneously`,
        links: [
          { label: 'Kohya sd-scripts Dataset Config', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md' },
        ],
      },
      {
        id: 'datasets-caption-files',
        title: 'Caption Files',
        content: `How caption files work and formatting options.

**Caption File Basics:**
- Same filename as image with .txt or .caption extension
- Example: \`character_01.png\` → \`character_01.txt\`
- Must be in same directory as the image
- Text encoding: UTF-8 (supports international characters)

**Caption Formats:**

**Tag Format (Booru-style):**
Comma-separated tags, common for anime/illustration training:
- Example: \`1girl, blue eyes, long hair, smile, outdoor, cherry blossoms\`
- Used by WD14 tagger
- Works well with \`shuffle_caption\` to vary tag order
- Use \`keep_tokens\` to lock important tags at the start (like trigger words)

**Natural Language Format:**
Descriptive sentences, more detailed than tags:
- Example: \`A girl with blue eyes and long hair smiling under cherry blossoms\`
- Used by BLIP and GIT captioning
- More context and relationships between elements
- Better for photorealistic or complex scene training

**Multi-line Captions:**
- Multiple captions per image (one per line)
- System randomly selects one caption per epoch
- Adds variety to training
- Example:
  - Line 1: \`1girl, blue eyes, long hair, happy expression\`
  - Line 2: \`1girl, blue eyes, long hair, gentle smile\`
  - Line 3: \`1girl, blue eyes, long hair, cheerful mood\`

**Caption Generation:**
Captions can be generated automatically using:
- **WD14 Tagger**: Tag-based (see WD14 Tagger subsection)
- **BLIP Captioning**: Natural language (see BLIP Captioning subsection)
- **GIT Captioning**: Detailed descriptions (see GIT Captioning subsection)

**Manual Editing:**
- You can (and should) manually edit auto-generated captions
- Add trigger words for your LoRA (e.g., \`ohwx person\`, \`my_character\`)
- Remove incorrect tags or add missing details
- Refine descriptions for better training results

**Caption Processing Options (in TOML config):**
- \`shuffle_caption\`: Randomly shuffle tag order during training (tag format only)
- \`keep_tokens\`: Number of tags to keep at start without shuffling (e.g., trigger word)
- \`caption_extension\`: File extension to look for (default: .txt)
- \`caption_separator\`: Separator for tags (default: ", " for comma-space)

**Fallback Captions:**
- Use \`class_tokens\` in TOML config as fallback if caption file missing
- Example: \`class_tokens = "1girl, my_character"\`
- Useful if some images don't need detailed captions`,
        links: [
          { label: 'Kohya sd-scripts Dataset Config', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md' },
        ],
      },
      {
        id: 'datasets-toml',
        title: 'TOML Configuration',
        content: `Detailed guide to dataset configuration files (from Kohya sd-scripts).

**Note:** Our system can automatically generate TOML configuration for you through the web interface. However, if you prefer more control or need to customize advanced settings, you can create your own TOML files manually using the format below.

**Basic Structure:**

The configuration file has three main sections:
- **[general]** - Settings that apply to all datasets
- **[[datasets]]** - Individual dataset configuration (can have multiple)
- **[[datasets.subsets]]** - Image directories within a dataset

**[general] Section Options:**
- \`shuffle_caption\`: Randomly shuffle tags during training (true/false)
- \`keep_tokens\`: Number of tags to keep at start that won't be shuffled (integer)
- \`caption_extension\`: File extension for caption files (default: .txt)

**[[datasets]] Section Options:**
- \`resolution\`: Training resolution - can be single number (512) or array ([512, 768]) or SDXL (1024)
- \`batch_size\`: Number of images processed per training step
- \`enable_bucket\`: Enable Aspect Ratio Bucketing to handle different image sizes (recommended: true)
- \`min_bucket_reso\`: Minimum bucket resolution (default: 256)
- \`max_bucket_reso\`: Maximum bucket resolution (default: 1024 for SD 1.5, 2048 for SDXL)
- \`bucket_reso_steps\`: Bucket step size (default: 64, SDXL minimum: 32)

**[[datasets.subsets]] Section Options:**
- \`image_dir\`: Path to your image directory (required)
- \`num_repeats\`: How many times to repeat each image during training (default: 1)
- \`class_tokens\`: Fallback caption text if .txt file is missing
- \`flip_aug\`: Enable random horizontal flip augmentation (true/false)
- \`color_aug\`: Enable color augmentation (true/false)
- \`random_crop\`: Use random crop instead of center crop (true/false)
- \`keep_tokens\`: Override general keep_tokens setting for this specific subset

**Example Configuration:**

A basic single-dataset configuration at 512x512 resolution with 4 images per batch, 10 repeats per image, and the dataset located at /path/to/images would use these settings in the [general], [[datasets]], and [[datasets.subsets]] sections.

For multi-resolution training, you can define multiple [[datasets]] blocks - for example, one at 512x512 with batch size 4 and another at 768x768 with batch size 2, each pointing to their respective image directories with different repeat counts.`,
        links: [
          { label: 'Kohya Config Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md' },
        ],
      },
      {
        id: 'datasets-wd14',
        title: 'WD14 Tagger',
        content: `Tag-based image captioning using WD14 tagger models.

WD14 tagger generates comma-separated tags (e.g., "1girl, blue eyes, long hair, smile") commonly used for anime/illustration datasets.

**Note:** Our web interface provides automated WD14 tagging through the Dataset page. However, you can also run the sd-scripts tagger manually if you prefer direct control or need to tag images outside the interface.

**Basic Usage:**
Run the tagger with ONNX runtime (faster) on your image directory:
- Script: tag_images_by_wd14_tagger.py
- Use --onnx flag for ONNX runtime acceleration
- Specify model with --repo_id (default: wd-v1-4-convnext-tagger-v2)
- Set batch size with --batch_size (adjust for VRAM)

**Available Models (SmilingWolf):**
- \`wd-swinv2-tagger-v3\`: Latest v3 model, most accurate (recommended)
- \`wd-vit-tagger-v3\`: V3 ViT architecture variant
- \`wd-convnext-tagger-v3\`: V3 ConvNext architecture variant
- \`wd-v1-4-convnext-tagger-v2\`: V2 model (default if --repo_id omitted)

**Key Options:**
- \`--onnx\`: Use ONNX runtime for faster inference (highly recommended)
- \`--batch_size\`: Process multiple images simultaneously (adjust based on VRAM)
- \`--thresh\`: Global confidence threshold (default: 0.35, lower values = more tags)
- \`--general_threshold\`: Separate threshold for general tags
- \`--character_threshold\`: Separate threshold for character tags
- \`--recursive\`: Process subfolders recursively

**Tag Formatting Options:**
- \`--remove_underscore\`: Remove underscores from tags (tag_name → tag name)
- \`--undesired_tags\`: Comma-separated list of tags to exclude from output
- \`--use_rating_tags\`: Add content rating tag at start (safe/questionable/explicit)
- \`--use_rating_tags_as_last_tag\`: Add rating tag at end instead of start
- \`--character_tags_first\`: Place character tags before general tags
- \`--character_tag_expand\`: Split character tags (miku_(vocaloid) → miku, vocaloid)
- \`--always_first_tags\`: Comma-separated tags to always place first (e.g., "1girl,1boy")
- \`--caption_separator\`: Tag separator string (default: ", ")

**Animagine XL Format Example:**
For Animagine XL style formatting, use: --onnx, --remove_underscore, --use_rating_tags_as_last_tag, --character_tags_first, --character_tag_expand, and --always_first_tags "1girl,1boy"

**Output:**
- Creates .txt caption files next to each image
- Filename matches image: image.png → image.txt
- Tags are comma-separated and can be manually edited after generation`,
        links: [
          { label: 'WD14 Tagger Models', url: 'https://huggingface.co/SmilingWolf' },
          { label: 'Kohya WD14 Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/wd14_tagger_README-en.md' },
        ],
      },
      {
        id: 'datasets-blip',
        title: 'BLIP Captioning',
        content: `Natural language image captioning using BLIP model.

BLIP generates natural language captions (e.g., "A girl with blue eyes and long hair smiling") which can be more descriptive than tag-based methods.

**Note:** Our web interface provides automated BLIP captioning through the Dataset page. However, you can also run the sd-scripts captioning script manually if you prefer direct control or need custom generation settings.

**Basic Usage:**
Run BLIP captioning on your image directory:
- Script: finetune/make_captions.py in sd_scripts
- Requires BLIP model weights (auto-downloaded from URL or local path)
- Specify your training data directory as first argument
- Default model: model_large_caption.pth from Google Cloud Storage

**Key Options:**
- \`train_data_dir\`: Directory containing training images (required, first argument)
- \`--caption_weights\`: Path or URL to BLIP weights (default: model_large_caption.pth from Google)
- \`--caption_extension\`: Output caption file extension (default: .caption)
- \`--batch_size\`: Number of images to process at once (default: 1)
- \`--max_data_loader_n_workers\`: Enable DataLoader with N workers for faster image reading
- \`--recursive\`: Search for images in subfolders recursively

**Generation Settings:**
- \`--beam_search\`: Use beam search instead of nucleus sampling for more deterministic results
- \`--num_beams\`: Number of beams for beam search (default: 1, higher = better quality but slower)
- \`--top_p\`: Top-p value for nucleus sampling (default: 0.9)
- \`--max_length\`: Maximum caption length in tokens (default: 75)
- \`--min_length\`: Minimum caption length in tokens (default: 5)
- \`--seed\`: Random seed for reproducibility (default: 42)

**Sampling Methods:**
**Nucleus Sampling (default):** Randomly samples from top probability tokens. More creative and varied captions. Use --top_p to control randomness (0.9 = fairly conservative).

**Beam Search (--beam_search):** Deterministic search for best caption. More accurate but less varied. Use --num_beams to control quality vs speed trade-off.

**Performance:**
- Use --max_data_loader_n_workers with multiple CPU cores for faster image loading
- Increase --batch_size if you have sufficient VRAM (2-4 for most GPUs)
- BLIP requires more VRAM than WD14 tagger but produces more natural captions

**Output:**
- Creates caption files next to each image
- Default extension: .caption (customize with --caption_extension)
- Filename matches image: image.png → image.caption
- Use --debug flag to print captions to console during generation`,
        links: [
          { label: 'BLIP Model Paper', url: 'https://arxiv.org/abs/2201.12086' },
          { label: 'Kohya sd-scripts Repository', url: 'https://github.com/kohya-ss/sd-scripts' },
        ],
      },
      {
        id: 'datasets-git',
        title: 'GIT Captioning',
        content: `Advanced image captioning using Microsoft GIT (Generative Image-to-text) model.

GIT is designed for generating detailed text descriptions from images, particularly good at reading text within images and generating comprehensive captions.

**Note:** Our web interface provides automated GIT captioning through the Dataset page. However, you can also run the sd-scripts captioning script manually if you prefer direct control or need custom generation settings.

**Basic Usage:**
Run GIT captioning on your image directory:
- Script: finetune/make_captions_by_git.py in sd_scripts
- Requires GIT model from HuggingFace (auto-downloaded)
- Specify your training data directory as first argument
- Default model: microsoft/git-large-textcaps

**Key Options:**
- \`train_data_dir\`: Directory containing training images (required, first argument)
- \`--model_id\`: HuggingFace model ID (default: microsoft/git-large-textcaps)
- \`--caption_extension\`: Output caption file extension (default: .caption)
- \`--batch_size\`: Number of images to process simultaneously (default: 1)
- \`--max_data_loader_n_workers\`: Enable DataLoader with N workers for faster image loading
- \`--recursive\`: Search for images in subfolders recursively

**Generation Settings:**
- \`--max_length\`: Maximum caption length in tokens (default: 50)
- \`--remove_words\`: Remove phrases like "with the words xxx" from captions (helps reduce false text detection)

**Available Models:**
- \`microsoft/git-large-textcaps\`: Large model trained on TextCaps (default, best for text-heavy images)
- \`microsoft/git-large\`: Large model, general purpose
- \`microsoft/git-base\`: Base model, faster but less detailed
- Check HuggingFace for additional GIT model variants

**Remove Words Option:**
GIT models trained on TextCaps can hallucinate text in images. The --remove_words flag uses regex patterns to remove common false detections:
- "has the words/letters/name XXX"
- "with a sign that says XXX"
- "with the number XXX on it"
- And similar patterns

Recommended to use --remove_words unless you specifically need text detection.

**Performance:**
- Use --max_data_loader_n_workers with multiple CPU cores for faster image loading
- Increase --batch_size for better GPU utilization (2-4 for most GPUs)
- GIT models are larger than BLIP but provide more detailed captions
- TextCaps variant is best for images containing text or signage

**Output:**
- Creates caption files next to each image
- Default extension: .caption (customize with --caption_extension)
- Filename matches image: image.png → image.caption
- Use --debug flag to print captions to console and see remove_words effects`,
        links: [
          { label: 'GIT Model Paper', url: 'https://arxiv.org/abs/2205.14100' },
          { label: 'Microsoft GIT Models', url: 'https://huggingface.co/microsoft/git-large-textcaps' },
          { label: 'Kohya sd-scripts Repository', url: 'https://github.com/kohya-ss/sd-scripts' },
        ],
      },
    ],
  },
  {
    id: 'training-config',
    title: 'Training Configuration',
    icon: <Settings className="w-4 h-4" />,
    subsections: [
      {
        id: 'training-config-learning',
        title: 'Learning Parameters',
        content: `Core training parameters that control learning behavior.

**Learning Rate (LR):**
- Default: 1e-4 (0.0001)
- Lower = slower, more stable training
- Higher = faster, but risk of overtraining
- SDXL: 1e-4 to 5e-5
- SD 1.5: 1e-4 to 1e-3

**Network Rank (Dim):**
- Default: 32
- Higher = more detail, larger file size
- Common values: 8, 16, 32, 64, 128
- Recommendation: 32 for characters, 64+ for styles

**Network Alpha:**
- Generally set to same value as dim or half
- Affects learning rate scaling
- Default: 1

**Epochs:**
- Number of times to train on full dataset
- Start with 5-10 epochs
- Monitor loss to avoid overtraining

**Batch Size:**
- Higher = faster training, more VRAM
- Lower = slower, less VRAM
- Recommendation: 1-4 depending on GPU`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'training-config-optimizer',
        title: 'Optimizers',
        content: `Choosing the right optimizer for your training.

**AdamW8bit (Recommended):**
- Memory efficient (requires bitsandbytes)
- Widely used and reliable
- Good default choice

**AdamW:**
- Standard optimizer
- More VRAM usage than 8bit version
- Stable and well-tested

**Lion:**
- Newer optimizer (requires lion-pytorch)
- Can be faster than AdamW
- Less memory usage

**Adafactor:**
- Very memory efficient
- Good for large models (SDXL)
- May require learning rate adjustments
- Use with: \`scale_parameter=False, relative_step=False, warmup_init=False\`

**DAdaptation:**
- Adaptive learning rate
- Requires dadaptation package
- Can automatically adjust LR`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'training-config-scheduler',
        title: 'Learning Rate Schedulers',
        content: `Control how learning rate changes during training.

**constant:**
- Learning rate stays the same
- Simple and predictable
- Good default choice

**cosine:**
- Learning rate follows cosine curve
- Gradually decreases over time
- Popular choice for smooth training

**cosine_with_restarts:**
- Cosine with periodic restarts
- Can help escape local minima
- More complex training dynamics

**linear:**
- Linear decay of learning rate
- Gradual decrease to zero
- Simple fallback option

**constant_with_warmup:**
- Starts low, ramps up, then stays constant
- Helps stabilize early training
- Use with \`--lr_warmup_steps\`

**Warmup Steps:**
- Gradually increase LR at start
- Typical: 100-500 steps
- Helps prevent early instability`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'training-config-model-components',
        title: 'Model Components',
        content: `Configuring text encoder training for LoRA.

**U-Net Only Training (Recommended for SDXL):**
- \`--network_train_unet_only\`: Train only U-Net, freeze text encoders
- Default behavior for most LoRA training
- SDXL has **two** text encoders - training both is unpredictable
- Recommended for SDXL to avoid complications

**Separate Learning Rates:**
- \`--text_encoder_lr=1e-5\`: Set separate LR for text encoder
- Should be **lower** than U-Net learning rate
- Example: U-Net at 1e-4, Text Encoder at 1e-5 or 5e-6
- If not specified, uses same LR as \`--learning_rate\`

**When to Train Text Encoder:**
- Small datasets (under 20 images) may benefit
- Character/concept training with limited data
- Similar to DreamBooth approach
- Not recommended for SDXL (too unpredictable)

**SD 1.5 vs SDXL:**
- SD 1.5: Single text encoder (CLIP)
- SDXL: Two text encoders (CLIP-L and OpenCLIP-G)
- Training both SDXL encoders often produces worse results`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'training-config-memory',
        title: 'Memory Management',
        content: `Reducing VRAM usage for training on consumer GPUs.

**Gradient Checkpointing:**
- \`--gradient_checkpointing\`: Enable gradient checkpointing
- **Significant** VRAM savings (20-40% reduction)
- Trades compute time for memory (slightly slower)
- Essential for 8-12GB GPUs training SDXL
- Minimal impact on final quality

**Mixed Precision Training:**
- \`--mixed_precision="fp16"\`: Use half precision (16-bit)
- \`--mixed_precision="bf16"\`: Use bfloat16 (RTX 30/40 series, A100, H100)
- Reduces VRAM usage by ~30-50%
- Faster training on modern GPUs
- **bf16 recommended** for Ampere/Ada/Hopper GPUs
- fp16 for older GPUs (RTX 20 series, Pascal)

**Caching:**
- \`--cache_latents\`: Cache VAE outputs to disk
  - Saves VRAM during training
  - Faster training (VAE only runs once)
  - Essential for low VRAM setups
- \`--cache_text_encoder_outputs\`: Cache text encoder (SDXL only)
  - SDXL-specific optimization
  - Saves significant VRAM with dual encoders
  - Cannot use with caption shuffling/dropout

**Batch Size vs Gradient Accumulation:**
- Lower \`batch_size\` = less VRAM
- Use \`--gradient_accumulation_steps=N\` to simulate larger batches
- Effective batch = \`batch_size * gradient_accumulation_steps\`
- Example: batch_size=1, gradient_accumulation=4 = effective batch of 4

**8-bit Optimizers:**
- \`--optimizer_type="AdamW8bit"\`: 8-bit Adam optimizer
- Requires bitsandbytes library
- Reduces optimizer VRAM usage by ~50%
- Widely used and stable`,
        links: [
          { label: 'Kohya sd-scripts SDXL Training', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_SDXL-en.md' },
        ],
      },
      {
        id: 'training-config-duration',
        title: 'Training Duration & Checkpoints',
        content: `Managing training duration and checkpoint saving.

**Training Duration:**
- \`--max_train_steps=10000\`: Train for specific number of steps
- \`--max_train_epochs=10\`: Train for specific number of epochs
- **Epochs override steps** if both specified
- Epoch = one complete pass through dataset

**Step Calculation:**
- Total steps = (num_images × num_repeats / batch_size) × epochs
- Example: 50 images, 10 repeats, batch 4, 10 epochs = 1250 steps
- Use calculator to determine optimal epoch count

**Checkpoint Saving:**
- \`--save_every_n_epochs=1\`: Save after every N epochs
- \`--save_every_n_steps=500\`: Save after every N steps
- Both can be used together (saves at both intervals)
- Checkpoints saved as: \`{output_name}-{epoch}.safetensors\`
- Only final model saved if neither specified

**Save Precision:**
- \`--save_precision="fp16"\`: Save as half precision (smaller files)
- \`--save_precision="bf16"\`: Save as bfloat16
- \`--save_precision="float"\`: Save as full precision (default)
- fp16/bf16 reduce file size with minimal quality loss

**Sample Generation During Training:**
- \`--sample_every_n_epochs=1\`: Generate samples every N epochs
- \`--sample_every_n_steps=500\`: Generate samples every N steps
- \`--sample_prompts="path/to/prompts.txt"\`: Prompts for sampling
- Helps monitor training progress visually`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'training-config-loss',
        title: 'Loss Functions & Weighting',
        content: `Loss calculation and weighting strategies.

**Min-SNR Gamma:**
- \`--min_snr_gamma=5\`: Min-SNR Weighting Strategy
- Adjusts loss weights for timesteps with high noise in early training
- Stabilizes learning and can improve convergence
- Recommended value: 5 (from paper)

**Debiased Estimation:**
- \`--debiased_estimation_loss\`: Alternative to Min-SNR
- Calculates loss using Debiased Estimation
- Similar purpose to Min-SNR but different approach

**Huber Loss:**
- \`--loss_type="huber"\` or \`"smooth_l1"\`
- Alternative loss functions to MSE
- Can be more robust to outliers
- Use with \`--huber_schedule="snr"\` (recommended)
- \`--huber_c=0.1\`: Huber loss parameter

**Scale v-prediction Loss:**
- \`--scale_v_pred_loss_like_noise_pred\`: Scale v-pred loss
- Only relevant for v-prediction models
- Normalizes loss scaling
- Generally recommended when using v-pred models`,
        links: [
          { label: 'Kohya sd-scripts Advanced Training', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_advanced.md' },
        ],
      },
      {
        id: 'training-config-noise',
        title: 'Noise Settings',
        content: `Noise-related training configurations.

**Noise Offset:**
- \`--noise_offset=0.1\`: Add offset to noise
- Helps generate better dark/light images
- Improves contrast in outputs
- Typical values: 0.05 to 0.1
- Can help with "washed out" results

**Adaptive Noise Scale:**
- \`--adaptive_noise_scale=N\`: Adjusts noise offset dynamically
- Based on mean absolute value of latents
- Used together with \`--noise_offset\`

**Multi-Resolution Noise:**
- \`--multires_noise_iterations=6\`: Enable multi-res noise
- \`--multires_noise_discount=0.3\`: Discount rate
- Adds noise of different frequency components
- Expected to improve detail reproduction
- Typical: 6-10 iterations, 0.3 discount

**Input Perturbation Noise:**
- \`--ip_noise_gamma=0.1\`: Enable IP noise
- Adds small noise to input (latents) for regularization
- \`--ip_noise_gamma_random_strength\`: Randomize strength

**Zero Terminal SNR:**
- \`--zero_terminal_snr\`: Zero terminal SNR
- Training technique for certain model types
- Can improve quality for some use cases`,
        links: [
          { label: 'Kohya sd-scripts Advanced Training', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_advanced.md' },
        ],
      },
      {
        id: 'training-config-model-specific',
        title: 'Model-Specific Settings',
        content: `Settings specific to model architecture and type.

**CLIP Skip:**
- \`--clip_skip=2\`: Skip last N layers of text encoder
- SD 1.5: Usually 1 (default) or 2
- SDXL: Use 2 (or omit - defaults correctly)
- Check base model card for recommended value
- Some anime/manga models prefer clip_skip=2

**V-Parameterization:**
- \`--v_parameterization\`: Enable v-prediction mode
- Required for v-prediction models:
  - SD 2.x 768px models
  - NoobAI SDXL (and other v-pred SDXL finetunes)
  - Some custom finetunes
- Check model card to confirm if needed
- Don't use for standard SD 1.5 or base SDXL

**SD 2.x Specific:**
- \`--v2\`: Enable SD 2.x mode
- Required for SD 2.0/2.1 base models
- Use with \`--v_parameterization\` for 768px models

**Model Version Detection:**
- Script usually auto-detects from model config
- Manual flags override auto-detection
- Check model card if unsure`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
          { label: 'Kohya sd-scripts SDXL Training', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_SDXL-en.md' },
        ],
      },
    ],
  },
  {
    id: 'monitoring',
    title: 'Training Monitoring',
    icon: <Monitor className="w-4 h-4" />,
    content: `Understanding training progress and when to stop.

**Loss Values:**
- Loss should generally decrease over time
- Too low (<0.05): Possible overtraining
- Fluctuating wildly: Learning rate too high
- Not decreasing: Learning rate too low

**Sample Images:**
- Generate sample images during training
- Check for quality and adherence to concept
- Stop if images degrade (overtraining)

**Signs of Good Training:**
- Steady loss decrease
- Sample images improve quality
- Concept is recognizable
- No artifacts or degradation

**Signs of Overtraining:**
- Loss becomes very low (<0.05)
- Images look "burned" or have artifacts
- Loss of diversity in outputs
- Stop training and use earlier checkpoint`,
  },
  {
    id: 'post-training',
    title: 'Post-Training',
    icon: <Upload className="w-4 h-4" />,
    content: `What to do after training completes.

**LoRA Resizing:**
- Reduce LoRA file size while maintaining quality
- Useful for sharing or reducing VRAM usage
- Common targets: dim 32 → 16 or 8
- Use the Resize tool in Utilities

**Testing Your LoRA:**
- Test in multiple inference UIs (ComfyUI, A1111, Forge)
- Try different strengths (0.5 to 1.0)
- Combine with different base models
- Test with various prompts

**Sharing:**
- Upload to HuggingFace Hub via Utilities page
- Include sample images and trigger words
- Document recommended settings
- Specify compatible base models`,
  },
  {
    id: 'faq',
    title: 'FAQ',
    icon: <BookOpen className="w-4 h-4" />,
    subsections: [
      {
        id: 'faq-image-quality',
        title: 'What Makes Good Training Images?',
        content: `Guidelines for image quality in your dataset.

These are recommendations based on common experience - not strict requirements. Your specific use case may vary.

**Lighting:**
- Well-lit images help the model learn features clearly
- Avoid extremely dark or overexposed images
- Consistent lighting across dataset helps but isn't required

**Sharpness:**
- Sharp, in-focus images generally work best
- Blurry images may teach the model to generate blurry outputs
- Motion blur or intentional artistic blur is fine if that's part of your style

**Resolution:**
- Higher quality source images generally produce better results
- Don't need to be exact training resolution (bucketing handles this)
- Very low resolution images (<256px) may benefit from upscaling first

**Consistency:**
- Try to maintain similar quality levels across your dataset
- Mix of professional and amateur photos is fine
- Avoid heavy JPG compression artifacts if possible`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'faq-variety',
        title: 'How Much Variety Do I Need?',
        content: `Recommendations for dataset variety and diversity.

**Poses & Angles:**
- Different poses help model generalize beyond specific positions
- Variety of angles: front view, side view, 3/4 view, etc.
- Balance is key - don't have 90% of one pose and 10% others

**Expressions & Emotions:**
- If training a character, vary emotions and expressions
- Happy, neutral, serious, surprised, etc.
- Not needed for style training or objects

**Settings & Backgrounds:**
- Various backgrounds and environments
- Different lighting conditions
- Indoor, outdoor, different settings

**Outfits & Appearance:**
- Multiple clothing styles (if applicable to your use case)
- Different accessories or props
- Variation in appearance helps model learn the core concept

**How Much is Enough?**
- You don't need EVERY possible variation
- Enough variety that the model doesn't memorize one specific look
- Quality and relevance matter more than checking every box`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'faq-dataset-size',
        title: 'How Many Images Do I Need?',
        content: `Dataset size guidelines based on training complexity.

These are starting points - you may need more or less depending on your specific case.

**Simple Concepts (15-30 images):**
- Single object or simple style
- Consistent appearance across images
- Limited variation needed
- Examples: specific object, simple logo, basic pattern

**Character Training (30-50 images):**
- Person, character, or creature with consistent features
- Multiple poses and expressions recommended
- Different outfits and settings helpful
- Examples: person's face, anime character, OC

**Complex Styles (50-100+ images):**
- Art style, aesthetic, or complex concept
- More variety needed to capture style essence
- Quality over quantity still applies
- Examples: artist style, rendering technique, aesthetic

**Important Note:**
These are rough guidelines. Some people successfully train characters with 15 images, others need 100. It depends on:
- Consistency of your subject
- How unique/complex the concept is
- Quality of your images
- How much variation you want the model to handle

Experimentation is part of the process!`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'faq-quality-vs-quantity',
        title: 'Quality vs Quantity?',
        content: `When more images doesn't mean better results.

**The Short Answer:**
20 high-quality, varied images > 100 similar or low-quality images

**Quality Matters More:**
- One excellent, well-tagged image teaches more than 10 mediocre duplicates
- Remove images that don't add new information
- If two images are nearly identical, keep the better one

**Curation is Important:**
- Remove duplicates and near-duplicates (they don't add training value)
- Remove off-topic images that don't match your concept
- Remove extremely poor quality images (very blurry, heavily compressed, etc.)
- Every image should contribute something to the training

**Diminishing Returns:**
- Beyond a certain point, more similar images don't help much
- Better to have 30 varied images than 100 images of the same pose
- Focus on filling gaps in variety rather than adding more of what you have

**When Quantity Helps:**
- Training complex styles that appear in many forms
- When you have genuinely different images (not just slight variations)
- Large datasets can work well if properly curated

**Bottom Line:**
Curate your dataset. Quality and variety trump raw numbers.`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
      {
        id: 'faq-repeat-counts',
        title: 'How Many Repeats Should I Use?',
        content: `Guidelines for configuring repeat counts in your TOML config.

**What Are Repeats?**
Repeats control how many times each image is seen during an epoch. Configured with \`num_repeats\` in your dataset TOML file.

**General Guidelines:**

**Small Datasets (15-30 images):**
- Higher repeats: 10-20 per image
- Ensures model sees each image enough times to learn
- More critical when you have limited data

**Medium Datasets (30-50 images):**
- Moderate repeats: 5-10 per image
- Balance between exposure and variety
- Most character training falls here

**Large Datasets (50-100+ images):**
- Lower repeats: 1-5 per image
- Dataset already has enough variety
- Higher repeats risk overtraining

**When to Adjust:**
- Watch your training loss and sample quality
- If loss decreases too slowly: increase repeats or learning rate
- If you see overtraining (loss very low, samples degrading): decrease repeats
- If samples lack detail: might need more repeats or epochs
- If samples look "fried" or artifacted: probably too many repeats

**No Perfect Formula:**
These are starting points. The "right" repeat count depends on your specific dataset, learning rate, and training goals. Experimentation and monitoring results is key.`,
        links: [
          { label: 'Kohya sd-scripts Dataset Config', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md' },
        ],
      },
      {
        id: 'faq-common-pitfalls',
        title: 'Common Dataset Mistakes',
        content: `Pitfalls to avoid when preparing your dataset.

**All Images Same Pose/Angle:**
- Model won't generalize to other poses
- Add variety in poses, angles, and compositions
- If you only have one pose, model will only learn that pose

**Inconsistent Subject Appearance:**
- Mixing very different looks confuses the model
- Make sure your subject is recognizable across images
- Some variation is good, but extreme differences hurt training

**Too Many Watermarked Images:**
- Model may learn to generate watermarks
- Remove or crop out watermarks when possible
- A few watermarked images in a large dataset is usually fine

**Mixing Unrelated Concepts:**
- Training "person A" and "person B" in same dataset
- Model gets confused about what it's supposed to learn
- Separate different concepts into different training runs

**Extremely Imbalanced Dataset:**
- Example: 90% frontal face shots, 10% profile
- Model will be biased toward the overrepresented view
- Try to balance your dataset if you want model to handle all views

**Poor Quality Images:**
- Very blurry, heavily compressed, or corrupted files
- Teaches model to produce poor quality outputs
- Remove or replace problematic images

**Not Curating Your Dataset:**
- Including every image without review
- Quality control is important
- Spend time reviewing and improving your dataset before training

**Ignoring Captions:**
- Auto-generated captions often have errors
- Review and edit captions for accuracy
- Good captions significantly improve training results`,
        links: [
          { label: 'Kohya sd-scripts Training Documentation', url: 'https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network.md' },
        ],
      },
    ],
  },
  {
    id: 'troubleshooting',
    title: 'Troubleshooting',
    icon: <Bug className="w-4 h-4" />,
    content: `Common issues and solutions.

**Training Won't Start:**
- Check that base model is downloaded
- Verify dataset path is correct
- Ensure images are valid format
- Check backend logs for errors

**Out of Memory (OOM):**
- Reduce batch size to 1
- Lower network dimension (rank)
- Enable gradient checkpointing
- Use a smaller base model

**Poor Results:**
- Increase number of epochs
- Adjust learning rate
- Add more varied training images
- Check image quality and tags
- Try different LoRA type

**Loss Not Decreasing:**
- Increase learning rate
- Check that images are properly tagged
- Verify dataset structure
- Increase batch size if possible`,
  },
];

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState<string>('getting-started');
  const [activeSubsection, setActiveSubsection] = useState<string | null>(null);

  // Find current content to display
  const getCurrentContent = () => {
    const section = docSections.find((s) => s.id === activeSection) || docSections[0];

    if (activeSubsection && section.subsections) {
      const subsection = section.subsections.find((sub) => sub.id === activeSubsection);
      if (subsection) {
        return {
          title: subsection.title,
          content: subsection.content,
          links: subsection.links,
          icon: section.icon,
        };
      }
    }

    // If section has subsections but none selected, show first one
    if (section.subsections && section.subsections.length > 0) {
      return {
        title: section.subsections[0].title,
        content: section.subsections[0].content,
        links: section.subsections[0].links,
        icon: section.icon,
      };
    }

    // Regular section without subsections
    return {
      title: section.title,
      content: section.content || '',
      links: section.links,
      icon: section.icon,
    };
  };

  const currentContent = getCurrentContent();

  return (
    <div className="min-h-screen bg-background pt-16">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Breadcrumbs */}
        <div className="mb-8">
          <Breadcrumbs
            items={[
              { label: 'Home', href: '/', icon: <Home className="w-4 h-4" /> },
              { label: 'Documentation', icon: <BookOpen className="w-4 h-4" /> },
            ]}
          />
        </div>

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-green-400 via-emerald-400 to-green-400 bg-clip-text text-transparent">
            Documentation
          </h1>
          <p className="text-xl text-muted-foreground">
            Guides and references for LoRA training
          </p>
        </div>

        {/* Two Column Layout */}
        <div className="flex gap-8 items-start">
          {/* Navigation Bubble/Panel */}
          <aside className="w-72 flex-shrink-0 sticky top-20 space-y-4">
            {/* Header Bubble */}
            <div className="bg-card/50 backdrop-blur-sm border border-border rounded-lg p-4">
              <h2 className="font-semibold text-foreground mb-1">Contents</h2>
              <p className="text-xs text-muted-foreground">
                {docSections.length} guides available
              </p>
            </div>

            {/* Navigation Links Bubble */}
            <nav className="bg-card/50 backdrop-blur-sm border border-border rounded-lg p-4">
              <ul className="space-y-1">
                {docSections.map((section) => {
                  // Section with subsections - use accordion
                  if (section.subsections && section.subsections.length > 0) {
                    return (
                      <li key={section.id}>
                        <Accordion type="single" collapsible>
                          <AccordionItem value={section.id} className="border-none">
                            <AccordionTrigger
                              className={`
                                px-3 py-2 rounded-md text-sm hover:no-underline
                                ${
                                  activeSection === section.id
                                    ? 'text-primary font-medium'
                                    : 'text-muted-foreground hover:text-foreground'
                                }
                              `}
                              onClick={() => {
                                setActiveSection(section.id);
                                setActiveSubsection(section.subsections![0].id);
                              }}
                            >
                              <div className="flex items-center gap-3">
                                <span className="flex-shrink-0">{section.icon}</span>
                                <span>{section.title}</span>
                              </div>
                            </AccordionTrigger>
                            <AccordionContent className="pb-0 pt-1">
                              <ul className="space-y-1 ml-8">
                                {section.subsections.map((subsection) => (
                                  <li key={subsection.id}>
                                    <button
                                      onClick={() => {
                                        setActiveSection(section.id);
                                        setActiveSubsection(subsection.id);
                                      }}
                                      className={`
                                        w-full text-left px-3 py-1.5 rounded-md text-xs transition-colors
                                        ${
                                          activeSubsection === subsection.id && activeSection === section.id
                                            ? 'bg-primary/10 text-primary font-medium'
                                            : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                                        }
                                      `}
                                    >
                                      {subsection.title}
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            </AccordionContent>
                          </AccordionItem>
                        </Accordion>
                      </li>
                    );
                  }

                  // Regular section without subsections
                  return (
                    <li key={section.id}>
                      <button
                        onClick={() => {
                          setActiveSection(section.id);
                          setActiveSubsection(null);
                        }}
                        className={`
                          w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors
                          ${
                            activeSection === section.id
                              ? 'bg-primary/10 text-primary font-medium'
                              : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                          }
                        `}
                      >
                        <span className="flex-shrink-0">{section.icon}</span>
                        <span className="text-left">{section.title}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </nav>
          </aside>

          {/* Main Content Area */}
          <main className="flex-1 min-w-0">
            <Card className="bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-primary/10">{currentContent.icon}</div>
                  <CardTitle className="text-3xl">{currentContent.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="prose prose-invert max-w-none break-words overflow-hidden">
                  {currentContent.content.split('\n\n').map((paragraph, idx) => {
                    if (paragraph.startsWith('**') && paragraph.endsWith(':**')) {
                      // Section header
                      return (
                        <h3 key={idx} className="text-xl font-bold text-cyan-400 mt-6 mb-3">
                          {paragraph.replace(/\*\*/g, '').replace(':', '')}
                        </h3>
                      );
                    } else if (paragraph.startsWith('**') && paragraph.includes('**\n')) {
                      // Bold header with content
                      const [header, ...rest] = paragraph.split('\n');
                      return (
                        <div key={idx} className="mb-4">
                          <h4 className="text-lg font-semibold text-foreground mb-2">
                            {header.replace(/\*\*/g, '')}
                          </h4>
                          <p className="text-muted-foreground leading-relaxed whitespace-pre-line">
                            {rest.join('\n')}
                          </p>
                        </div>
                      );
                    } else if (paragraph.startsWith('- ') || paragraph.includes('\n- ')) {
                      // List items
                      const items = paragraph.split('\n').filter((line) => line.trim());
                      return (
                        <ul key={idx} className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
                          {items.map((item, i) => (
                            <li
                              key={i}
                              className="ml-4"
                              dangerouslySetInnerHTML={{
                                __html: item.replace(/^- /, '').replace(/\*\*(.*?)\*\*/g, '<strong class="text-foreground">$1</strong>')
                              }}
                            />
                          ))}
                        </ul>
                      );
                    } else {
                      // Regular paragraph
                      return (
                        <p
                          key={idx}
                          className="text-muted-foreground leading-relaxed mb-4"
                          dangerouslySetInnerHTML={{
                            __html: paragraph
                              .replace(/\*\*(.*?)\*\*/g, '<strong class="text-foreground">$1</strong>')
                              .replace(/`(.*?)`/g, '<code class="bg-slate-800 px-2 py-1 rounded text-cyan-400 break-all">$1</code>'),
                          }}
                        />
                      );
                    }
                  })}
                </div>

                {/* External Links */}
                {currentContent.links && currentContent.links.length > 0 && (
                  <div className="mt-8 pt-6 border-t border-border">
                    <h4 className="text-lg font-semibold text-foreground mb-3">External Resources</h4>
                    <div className="space-y-2">
                      {currentContent.links.map((link, idx) => (
                        <a
                          key={idx}
                          href={link.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors"
                        >
                          <ExternalLink className="w-4 h-4" />
                          <span>{link.label}</span>
                          <ChevronRight className="w-4 h-4" />
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </main>
        </div>
      </div>
    </div>
  );
}
