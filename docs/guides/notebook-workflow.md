# Web UI Workflow Guide

## Overview

The Ktiseos Nyx LoRA Trainer uses a modern Next.js web interface organized into several dedicated sections:

- **Files Page** - Browse and upload datasets, manage models and outputs
- **Dataset Page** - Prepare images and captions, auto-tag with WD14/BLIP
- **Training Page** - Configure and execute LoRA training with 132 parameters
- **Utilities Page** - Post-training tools, HuggingFace upload, model management

## Data Ingestion Options

### Direct Image Upload
Upload individual images or entire folders directly through the drag-and-drop interface on the Dataset page.

### ZIP Upload
Upload ZIP files containing your training dataset through the Files page.

### Model/VAE Management
Download and manage models directly through the Files page with download integration.

## Getting Model and VAE Links

To use custom models or VAEs, you need direct download links. Here's how to find them:

### From Civitai

#### Method 1: Using Model Version ID
1. Navigate to the model or VAE page
2. Check the URL for `?modelVersionId=XXXXXX`
3. Copy the entire URL if the ID is present
4. If no ID is visible, switch between model versions to make it appear

#### Method 2: Copying Download Link
1. Scroll to the "Files" section on the model page
2. Right-click the **Download** button
3. Select "Copy Link Address" from the context menu

### From Hugging Face

#### Method 1: Repository URL
1. Go to the model or VAE repository main page
2. Copy the URL from your browser's address bar

#### Method 2: Direct File Link
1. Navigate to "Files and versions" tab
2. Find the specific file you want
3. Click the "..." menu next to the file
4. Right-click "Download" and copy the link address

## Advanced Features

### Image Utilities
- **Drag & Drop**: Upload entire folders of images at once
- **Batch Operations**: Select multiple files for operations
- **Directory Management**: Create, rename, and organize folders

### Tag Curation
- **WD14 Auto-Tagging**: Integrated automatic tagging for anime/illustration
- **Batch editing**: Apply changes to multiple images simultaneously
- **Caption management**: Edit and refine training captions in table view
- **Quality filtering**: Filter and curate images based on tags and metadata

### Training Configuration
- **132 Training Parameters**: Full control over LoRA training settings
- **Config Templates**: Save and load training configurations
- **Real-time Monitoring**: Live loss graphs and training progress
- **Parameter Validation**: Automatic validation of training parameters

## System Architecture

The Next.js + FastAPI architecture features:
- **Modern UI**: Responsive, real-time web interface with WebSocket updates
- **Automatic model detection**: Identifies SDXL vs SD 1.5 vs Flux models automatically
- **Kohya backend integration**: Uses proven training strategies and scripts
- **Cross-platform compatibility**: Runs in any modern browser
- **Environment agnostic**: Supports local, VastAI, and cloud deployments
- **No race conditions**: Proper state management instead of Jupyter widget issues
- **Real-time updates**: WebSocket-based progress monitoring (no polling!)