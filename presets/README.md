# Training Configuration Presets

This directory contains reusable training configuration presets.

## Structure

Presets are stored as JSON files with the following format:

```json
{
  "name": "Preset Name",
  "description": "Description of what this preset is optimized for",
  "model_type": "SDXL",
  "config": {
    // Training hyperparameters only (no dataset/model paths)
  }
}
```

## Usage

- **Built-in presets**: Shipped with the application, read-only
- **User presets**: Created by users, stored in this directory
- **Import/Export**: Users can share preset files

## Attribution

Some presets are adapted from:
- [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss) - Credit to bmaltais and contributors
