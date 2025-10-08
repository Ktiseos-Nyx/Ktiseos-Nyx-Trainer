# Widget Reference Guide

This reference covers the widgets in the current three-notebook system.

## Dataset_Preparation.ipynb

### Cell 1A: Setup Widget (Simple)
**Widget Type**: `setup_simple`
**Purpose**: Environment validation and setup

**What it does:**
- Validates installer.py setup
- Downloads training backend if needed
- Sets up directory structure
- Validates system resources

### Cell 3: Dataset Widget
**Widget Type**: `dataset`
**Purpose**: Dataset tagging and management

**Key Features:**
- **Dataset Input**: Point to your dataset directory
- **Auto-Tagging**: WD14 v3 taggers or BLIP captioning
- **Caption Editing**: Bulk edit, find/replace, manual tweaks
- **Trigger Words**: Add unique trigger word to all captions
- **Tag Filtering**: Remove unwanted tags with blacklists
- **Quality Tools**: Tag analysis and review

**Tagging Options:**
- **WD14 Models**: For anime/art content
  - wd14-vit-v2
  - wd14-convnext-v2
  - wd14-swinv2-v2
  - wd14-convnext-v3
- **BLIP**: For photos/realistic content

## Unified_LoRA_Trainer.ipynb

### Cell 1: Setup Widget (Simple)
**Widget Type**: `setup_simple`
**Purpose**: Environment validation

**What it does:**
- Validates that installer.py completed successfully
- Checks for required dependencies
- Confirms system readiness

### Cell 2: Training Configuration Widget
**Widget Type**: `training_widget`
**Purpose**: Universal training configuration

**Key Features:**
- **Smart Model Detection**: Auto-detects SD1.5/SDXL/Flux/SD3
- **Dataset Configuration**: Point to prepared dataset
- **Network Settings**: LoRA dimensions and architecture
- **Training Parameters**: Learning rates, batch size, epochs
- **Optimizer Selection**: AdamW, CAME, Prodigy options
- **Advanced Options**: Memory optimization, precision settings

**Auto-Detection:**
- Automatically selects correct Kohya training script
- Adapts settings based on model architecture
- Handles precision requirements per model type

### Cell 3: Training Monitor Widget
**Widget Type**: `training_monitor`
**Purpose**: Training execution and monitoring

**Key Features:**
- **Real-time Progress**: Live training progress bars
- **Loss Curves**: Visual training metrics
- **Log Output**: Detailed training logs
- **Error Handling**: Automatic error detection and reporting
- **Checkpoint Management**: Save and manage training checkpoints

## Utilities_Notebook.ipynb

**Note**: This notebook contains additional utility widgets that are documented separately as they are supplementary tools rather than core training workflow components.

## Widget Integration

### Shared Manager System
All widgets use a shared manager system for:
- Consistent state management
- Cross-notebook data sharing
- Unified configuration handling

### Error Handling
Widgets include built-in error handling for:
- Missing dependencies
- Configuration validation
- Resource availability checks
- User input validation

### Progress Feedback
Most widgets provide:
- Real-time progress indicators
- Status messages
- Error notifications
- Completion confirmations

## Widget Parameters

### Common Parameters Across Widgets

**Dataset Path Settings:**
- Input validation for directory paths
- Automatic path completion
- Error checking for missing directories

**Model Selection:**
- File browser for model selection
- Automatic format detection (.safetensors, .ckpt)
- Model architecture validation

**Training Settings:**
- Learning rate inputs with validation
- Batch size selection with memory checks
- Epoch/step configuration with estimates

**Output Configuration:**
- Output directory selection
- Naming conventions
- File format options

## Usage Notes

### Widget Dependencies
- Some widgets require previous setup completion
- Dataset widget needs prepared image data
- Training widget needs configured dataset
- Monitor widget requires active training configuration

### Performance Considerations
- Widgets adapt to available system resources
- Memory usage monitoring included
- Automatic fallbacks for limited hardware

### Compatibility
- All widgets work with the current Kohya-ss backend
- Support for multiple model architectures
- Cross-platform compatibility (Windows/Linux)