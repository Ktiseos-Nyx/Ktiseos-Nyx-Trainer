# Notebook Workflow Guide

## Overview

The LoRA Easy Training system uses three specialized notebooks for different stages of the training process:

- **`Dataset_Maker_Widget.ipynb`** - Prepare images and captions for training
- **`Unified_LoRA_Trainer.ipynb`** - Configure and execute LoRA training  
- **`Utilities_Notebooks.ipynb`** - Calculate training parameters and resize LoRA models

## Data Ingestion Options

### URL/ZIP Download
Download and extract datasets from URLs (Hugging Face, Civitai) or local ZIP files through the widget interface.

### Direct Image Upload
Upload individual images directly into your dataset folder using the file upload widgets.

### Gallery-DL Scraper
Use the integrated `gallery-dl` tool to scrape images and metadata from over 300 supported websites.

## Getting Model and VAE Links

To use custom models or VAEs, you need direct download links. Here's how to find them:

### From Civitai

#### Method 1: Using Model Version ID
1. Navigate to the model or VAE page
2. Check the URL for `?modelVersionId=XXXXXX` 
3. Copy the entire URL if the ID is present
4. If no ID is visible, switch between model versions to make it appear

![How to get a link from Civitai using the version ID](../../assets/model_url_civitai_1.png)

#### Method 2: Copying Download Link
1. Scroll to the "Files" section on the model page
2. Right-click the **Download** button
3. Select "Copy Link Address" from the context menu

![How to get a link from Civitai by copying the download address](../../assets/model_url_civitai_2.png)

### From Hugging Face

#### Method 1: Repository URL
1. Go to the model or VAE repository main page
2. Copy the URL from your browser's address bar

![How to get a link from Hugging Face using the repository URL](../../assets/model_url_hf_1.png)

#### Method 2: Direct File Link
1. Navigate to "Files and versions" tab
2. Find the specific file you want
3. Click the "..." menu next to the file
4. Right-click "Download" and copy the link address

![How to get a link from Hugging Face by copying the direct file address](../../assets/model_url_hf_2.png)

## Advanced Features

### Image Utilities
- **Resizing**: Batch resize images to target resolutions with quality options
- **Quality optimization**: Adjust compression and quality settings

### Tag Curation
- **FiftyOne Integration**: Visual tag editing interface for dataset inspection
- **Batch operations**: Apply changes to multiple images simultaneously
- **Caption management**: Edit and refine training captions

## System Architecture

The unified architecture features:
- **Automatic model detection**: Identifies SDXL vs SD 1.5 models automatically
- **Kohya backend integration**: Uses proven training strategies and scripts
- **Cross-platform compatibility**: Works with conda, venv, and system Python installations
- **Memory optimization**: Automatic VRAM detection and profile selection
- **Environment agnostic**: Supports local, VastAI, and RunPod deployments