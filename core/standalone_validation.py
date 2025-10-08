# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

"""Core validation system for LoRA training configurations"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import logging
logger = logging.getLogger(__name__)

def validate(args: dict) -> tuple[bool, bool, list[str], dict, dict, dict]:
    """
    Main validation function for LoRA training configurations
    
    Args:
        args: Dictionary with 'args' and 'dataset' keys (standard backend format)
        
    Returns:
        tuple: (passed, sdxl, errors, validated_args, validated_dataset_args, tag_data)
    """
    over_errors = []
    
    # Basic structure validation
    if "args" not in args:
        over_errors.append("args is not present")
    if "dataset" not in args:
        over_errors.append("dataset is not present")
    if over_errors:
        return False, False, over_errors, {}, {}, {}
    
    # Validate individual sections
    args_pass, args_errors, args_data = validate_args(args["args"])
    dataset_pass, dataset_errors, dataset_data = validate_dataset_args(args["dataset"])
    
    over_pass = args_pass and dataset_pass
    over_errors = args_errors + dataset_errors
    tag_data = {}
    
    if not over_errors:
        validate_warmup_ratio(args_data, dataset_data)
        validate_restarts(args_data, dataset_data)
        tag_data = validate_save_tags(dataset_data)
        validate_existing_files(args_data)
        validate_optimizer(args_data)
    
    # Detect if this is SDXL based on model or settings
    sdxl = validate_sdxl(args_data)
    
    if not over_pass:
        return False, sdxl, over_errors, args_data, dataset_data, tag_data
    
    return True, sdxl, over_errors, args_data, dataset_data, tag_data


def validate_args(args: dict) -> tuple[bool, list[str], dict]:
    """
    Validate training arguments section
    
    Args:
        args: Training arguments dictionary
        
    Returns:
        tuple: (passed, errors, validated_args)
    """
    passed_validation = True
    errors = []
    output_args = {}

    # Process all provided arguments
    for key, value in args.items():
        if not value and value != 0:  # Allow 0 but not empty strings/None
            # Only error on truly required fields
            if key in ['pretrained_model_name_or_path', 'output_dir']:
                passed_validation = False
                errors.append(f"No data filled in for {key}")
                continue
        
        # Handle nested configurations
        if isinstance(value, dict):
            # Handle network args or other nested configs
            if key == "network_args":
                for arg, val in value.items():
                    output_args[f"network_args.{arg}"] = val
            else:
                # Flatten nested config
                for arg, val in value.items():
                    if arg != "fa":  # Skip internal flags
                        output_args[arg] = val
        else:
            output_args[key] = value

    # Ensure network module is set
    if "network_module" not in output_args:
        if "guidance_scale" in output_args:
            output_args["network_module"] = "networks.lora_flux"
        else:
            output_args["network_module"] = "networks.lora"

    # File path validation
    file_inputs = [
        {"name": "pretrained_model_name_or_path", "required": True},
        {"name": "output_dir", "required": True},
        {"name": "sample_prompts", "required": False},
        {"name": "logging_dir", "required": False},
    ]

    for file in file_inputs:
        if file["required"] and file["name"] not in output_args:
            passed_validation = False
            errors.append(f"{file['name']} is not found")
            continue
        
        if file["name"] in output_args:
            file_path = output_args[file["name"]]
            if file_path:  # Only validate if path is provided
                # Convert relative paths to absolute for validation
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)
                
                if not Path(file_path).exists():
                    passed_validation = False
                    errors.append(f"{file['name']} input '{output_args[file['name']]}' does not exist")
                    continue
                else:
                    # Store as POSIX path for consistency
                    output_args[file["name"]] = Path(file_path).as_posix()

    return passed_validation, errors, output_args


def validate_dataset_args(args: dict) -> tuple[bool, list[str], dict]:
    """
    Validate dataset arguments section
    
    Args:
        args: Dataset arguments dictionary
        
    Returns:
        tuple: (passed, errors, validated_args)
    """
    passed_validation = True
    errors = []
    output_args = {key: value for key, value in args.items() if value}
    
    # Get subset name
    name = "subset"
    if "name" in output_args:
        name = output_args["name"]
        del output_args["name"]
    
    # Validate image directory
    if "image_dir" not in output_args or not output_args["image_dir"]:
        passed_validation = False
        errors.append(f"Image directory path for '{name}' does not exist")
    else:
        image_dir = output_args["image_dir"]
        
        # Convert relative paths to absolute for validation
        if not os.path.isabs(image_dir):
            image_dir = os.path.abspath(image_dir)
        
        if not Path(image_dir).exists():
            passed_validation = False
            errors.append(f"Image directory path for '{name}' does not exist")
        else:
            # Check if directory contains images
            if not _has_training_images(image_dir):
                passed_validation = False
                errors.append(f"No training images found in directory for '{name}'")
            else:
                # Store as POSIX path for consistency
                output_args["image_dir"] = Path(image_dir).as_posix()
    
    return passed_validation, errors, output_args


def validate_warmup_ratio(args: dict, dataset: dict) -> None:
    """Validate warmup ratio configuration"""
    if "lr_warmup_steps" in args and "warmup_ratio" in args:
        logger.warning("Both lr_warmup_steps and warmup_ratio specified, lr_warmup_steps takes precedence")


def validate_restarts(args: dict, dataset: dict) -> None:
    """Validate learning rate scheduler restarts"""
    if "lr_scheduler_num_cycles" not in args:
        return
    
    if args.get("lr_scheduler") != "cosine_with_restarts":
        logger.warning("lr_scheduler_num_cycles specified but scheduler is not cosine_with_restarts")


def validate_save_tags(dataset: dict) -> dict:
    """Validate and process tag saving configuration"""
    tag_data = {}
    
    # Process tag-related settings
    if "save_tag_frequency" in dataset:
        tag_data["save_frequency"] = dataset["save_tag_frequency"]
    
    return tag_data


def validate_existing_files(args: dict) -> None:
    """Validate existing file configurations"""
    # Check for conflicting file settings
    if args.get("resume") and args.get("weights_to_load"):
        logger.warning("Both resume and weights_to_load specified, resume takes precedence")


def validate_optimizer(args: dict) -> None:
    """Validate optimizer configuration"""
    optimizer = args.get("optimizer_type", "AdamW8bit")
    
    # Valid optimizers from our system + Derrian's custom ones
    valid_optimizers = {
        'AdamW8bit', 'AdamW', 'Lion8bit', 'Lion', 'SGDNesterov8bit', 'SGDNesterov',
        'DAdaptation', 'DAdaptAdam', 'DAdaptAdan', 'DAdaptSGD', 'DAdaptLion',
        'Prodigy', 'CAME'
    }
    
    if optimizer not in valid_optimizers:
        logger.warning(f"Optimizer {optimizer} may not be supported")
    
    # Validate optimizer-specific settings
    if optimizer.startswith("DAdapt") and "learning_rate" in args:
        if float(args["learning_rate"]) != 1.0:
            logger.warning("DAdaptation optimizers typically use learning_rate=1.0")


def validate_sdxl(args: dict) -> bool:
    """
    Determine if this is an SDXL model based on configuration
    
    Args:
        args: Training arguments
        
    Returns:
        bool: True if SDXL model detected
    """
    # Check model path for SDXL indicators
    model_path = args.get("pretrained_model_name_or_path", "").lower()
    if "sdxl" in model_path or "xl" in model_path:
        return True
    
    # Check resolution settings
    resolution = args.get("resolution", 512)
    if isinstance(resolution, (int, float)) and resolution >= 1024:
        return True
    
    # Check for SDXL-specific parameters
    sdxl_indicators = [
        "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk", 
        "no_half_vae"
    ]
    
    for indicator in sdxl_indicators:
        if indicator in args:
            return True
    
    return False


def _has_training_images(directory: str) -> bool:
    """
    Check if directory contains training images
    
    Args:
        directory: Directory path to check
        
    Returns:
        bool: True if training images found
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    try:
        for file in os.listdir(directory):
            _, ext = os.path.splitext(file.lower())
            if ext in image_extensions:
                return True
    except (OSError, PermissionError):
        return False
    
    return False


def process_args(args: dict) -> tuple[bool, str]:
    """
    Process and generate training configuration
    
    Args:
        args: Validated training arguments
        
    Returns:
        tuple: (success, config_path)
    """
    try:
        # This would generate the actual config file
        # For now, just return success
        return True, "config.toml"
    except Exception as e:
        logger.error(f"Failed to process args: {e}")
        return False, ""


def process_dataset_args(args: dict) -> tuple[bool, str]:
    """
    Process and generate dataset configuration
    
    Args:
        args: Validated dataset arguments
        
    Returns:
        tuple: (success, dataset_path)
    """
    try:
        # This would generate the actual dataset file
        # For now, just return success
        return True, "dataset.toml"
    except Exception as e:
        logger.error(f"Failed to process dataset args: {e}")
        return False, ""


# Compatibility aliases for legacy backend imports
validate_args_legacy = validate_args
validate_dataset_args_legacy = validate_dataset_args