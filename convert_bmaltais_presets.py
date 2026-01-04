#!/usr/bin/env python3
"""
Convert Bmaltais/Kohya_SS presets to our format with metadata
"""
import json
from pathlib import Path

# Project-specific fields to exclude (same as frontend filter)
PROJECT_FIELDS = {
    'pretrained_model_name_or_path',
    'train_data_dir',
    'output_dir',
    'output_name',
    'ae', 'clip_l', 'clip_g', 't5xxl',  # Model paths
    'vae',
    'logging_dir',
    'log_tracker_name',
    'log_tracker_config',
    'wandb_api_key',
    'wandb_run_name',
    'sample_prompts',  # User-specific
    'huggingface_token',
    'huggingface_repo_id',
    'huggingface_path_in_repo',
    'resume',
    'resume_from_huggingface',
    'network_weights',
    'lora_network_weights',
}

def detect_model_type(filename: str, config: dict) -> str:
    """Detect model type from filename or config"""
    fn_lower = filename.lower()

    # Flux
    if 'flux' in fn_lower or config.get('flux1_checkbox'):
        return 'Flux1'
    # SD3
    if 'sd3' in fn_lower:
        return 'SD3'
    # SDXL
    if 'sdxl' in fn_lower or config.get('sdxl'):
        return 'SDXL'
    # SD 1.5/2.x
    if 'sd15' in fn_lower or 'sd1.5' in fn_lower:
        return 'SD1.5'
    if 'sd2' in fn_lower:
        return 'SD2.x'

    # Default based on LoRA type
    lora_type = config.get('LoRA_type', 'Standard')
    if lora_type == 'Flux1':
        return 'Flux1'

    return 'SD1.5'  # Default

def filter_config(config: dict) -> dict:
    """Remove project-specific fields"""
    return {k: v for k, v in config.items() if k not in PROJECT_FIELDS}

def generate_description(filename: str, config: dict) -> str:
    """Generate a description from the preset"""
    parts = []

    # Model type
    model_type = detect_model_type(filename, config)
    parts.append(f"{model_type} training preset")

    # LoRA type
    lora_type = config.get('LoRA_type', 'Standard')
    if lora_type and lora_type != 'Standard':
        parts.append(f"using {lora_type}")

    # Optimizer
    optimizer = config.get('optimizer', '')
    if optimizer:
        parts.append(f"with {optimizer} optimizer")

    return ' '.join(parts)

def convert_preset(input_path: Path) -> dict:
    """Convert a Bmaltais preset to our format"""
    with open(input_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)

    # Filter out project-specific fields
    filtered_config = filter_config(original_config)

    # Get name from filename (remove .json and category prefix if present)
    name = input_path.stem

    # Detect model type
    model_type = detect_model_type(name, original_config)

    # Generate description
    description = generate_description(name, original_config)

    # Create wrapped preset
    preset = {
        "name": name,
        "description": description,
        "model_type": model_type,
        "config": filtered_config,
        "is_builtin": True,
        "source": "bmaltais/kohya_ss"
    }

    return preset

def main():
    """Convert all Bmaltais presets"""
    presets_dir = Path(__file__).parent / 'presets'

    converted_count = 0

    # Process all subdirectories
    for category_dir in presets_dir.iterdir():
        if not category_dir.is_dir():
            continue

        if category_dir.name in ['.', '..']:
            continue

        print(f"\nProcessing {category_dir.name}/")

        for json_file in category_dir.glob('*.json'):
            try:
                preset = convert_preset(json_file)

                # Save in root presets dir with category prefix
                output_name = f"{category_dir.name}_{json_file.stem}.json"
                output_path = presets_dir / output_name

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(preset, f, indent=2, ensure_ascii=False)

                print(f"  ✓ {json_file.name} → {output_name}")
                converted_count += 1

            except Exception as e:
                print(f"  ✗ {json_file.name}: {e}")

    print(f"\n✅ Converted {converted_count} presets!")

if __name__ == '__main__':
    main()
