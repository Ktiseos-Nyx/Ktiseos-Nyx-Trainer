# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/custom_validation.py
"""
Custom validation system for our widget-based LoRA training interface.

Combines requirements from:
- Derrian's validation (backend compatibility)  
- Kohya's sd_scripts (TOML format requirements)
- Our widget system (config format and paths)

This replaces Derrian's validation which expects GUI JSON format incompatible with our TOML approach.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class LoRAValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class CustomLoRAValidator:
    """
    Custom validator that understands our widget config → TOML → sd_scripts workflow
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        
        # Valid optimizers (from sd_scripts and custom optimizers)
        self.valid_optimizers = {
            'AdamW8bit', 'AdamW', 'Lion8bit', 'Lion', 'SGDNesterov8bit', 'SGDNesterov',
            'DAdaptation', 'DAdaptAdam', 'DAdaptAdan', 'DAdaptSGD', 'DAdaptLion',
            'Prodigy', 'CAME'  # Derrian's custom optimizers
        }
        
        # Valid schedulers
        self.valid_schedulers = {
            'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
            'constant_with_warmup', 'adafactor'
        }
        
        # Valid network modules
        self.valid_network_modules = {
            'networks.lora', 'lycoris.kohya', 'networks.lora_flux'
        }
        
        # Model type to expected file extensions
        self.model_extensions = {'.safetensors', '.ckpt', '.pth'}

    def validate_widget_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate widget configuration before TOML generation.
        
        Args:
            config: Widget configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # === REQUIRED FIELDS ===
        required_fields = {
            'model_path': 'Base model path',
            'dataset_path': 'Dataset directory path', 
            'output_name': 'Output LoRA name',
            'output_dir': 'Output directory',
            'unet_lr': 'U-Net learning rate',
            'epochs': 'Number of epochs'
        }
        
        for field, description in required_fields.items():
            if not config.get(field):
                errors.append(f"❌ Missing required field: {description} ({field})")
        
        # === PATH VALIDATION ===
        if config.get('model_path'):
            model_path = config['model_path']
            if not self._validate_model_path(model_path):
                errors.append(f"❌ Model path not found or invalid: {model_path}")
        
        if config.get('dataset_path'):
            dataset_path = config['dataset_path']
            if not self._validate_dataset_path(dataset_path):
                errors.append(f"❌ Dataset path not found or invalid: {dataset_path}")
        
        # === TRAINING PARAMETER VALIDATION ===
        self._validate_learning_rates(config, errors)
        self._validate_optimizer(config, errors)
        self._validate_scheduler(config, errors)
        self._validate_network_config(config, errors)
        self._validate_training_config(config, errors)
        self._validate_advanced_parameters(config, errors)  # New validation for advanced parameters
        
        return len(errors) == 0, errors

    def validate_generated_tomls(self, config_path: str, dataset_path: str) -> Tuple[bool, List[str]]:
        """
        Validate generated TOML files match sd_scripts requirements.
        
        Args:
            config_path: Path to generated config.toml
            dataset_path: Path to generated dataset.toml
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate config.toml structure
        try:
            import toml
            with open(config_path, 'r') as f:
                config_data = toml.load(f)
            
            # Check required sections (sd_scripts format)
            required_sections = ['model_arguments', 'training_arguments']
            for section in required_sections:
                if section not in config_data:
                    errors.append(f"❌ Missing section in config.toml: {section}")
            
            # Validate model arguments section
            if 'model_arguments' in config_data:
                model_args = config_data['model_arguments']
                if not model_args.get('pretrained_model_name_or_path'):
                    errors.append("❌ Missing pretrained_model_name_or_path in model_arguments")
                else:
                    # Check if model path exists (convert relative to absolute for checking)
                    model_path = model_args['pretrained_model_name_or_path']
                    if not os.path.isabs(model_path):
                        model_path = os.path.join(self.project_root, model_path)
                    if not os.path.exists(model_path):
                        errors.append(f"❌ Model file does not exist: {model_args['pretrained_model_name_or_path']}")
            
            # Validate training arguments section
            if 'training_arguments' in config_data:
                training_args = config_data['training_arguments']
                if not training_args.get('output_dir'):
                    errors.append("❌ Missing output_dir in training_arguments")
                
        except Exception as e:
            errors.append(f"❌ Error reading config.toml: {e}")
        
        # Validate dataset.toml structure
        try:
            with open(dataset_path, 'r') as f:
                dataset_data = toml.load(f)
            
            # Check Kohya dataset format
            if 'datasets' not in dataset_data:
                errors.append("❌ Missing 'datasets' section in dataset.toml")
            else:
                datasets = dataset_data['datasets']
                if not isinstance(datasets, list) or len(datasets) == 0:
                    errors.append("❌ 'datasets' must be a non-empty list")
                else:
                    for i, dataset in enumerate(datasets):
                        if 'subsets' not in dataset:
                            errors.append(f"❌ Missing 'subsets' in dataset {i}")
                        else:
                            for j, subset in enumerate(dataset['subsets']):
                                if 'image_dir' not in subset:
                                    errors.append(f"❌ Missing 'image_dir' in dataset {i}, subset {j}")
                                else:
                                    # Check if image directory exists (relative to project root)
                                    image_dir = subset['image_dir']
                                    if not os.path.isabs(image_dir):
                                        image_dir = os.path.join(self.project_root, image_dir)
                                    if not os.path.exists(image_dir):
                                        errors.append(f"❌ Image directory does not exist: {subset['image_dir']}")
                                    elif not self._has_training_images(image_dir):
                                        errors.append(f"❌ No training images found in: {subset['image_dir']}")
        
        except Exception as e:
            errors.append(f"❌ Error reading dataset.toml: {e}")
        
        return len(errors) == 0, errors

    def _validate_model_path(self, model_path: str) -> bool:
        """Validate model file exists and has correct extension"""
        if not model_path:
            return False
            
        # Convert relative paths to absolute for existence check
        if not os.path.isabs(model_path):
            model_path = os.path.join(self.project_root, model_path)
            
        if not os.path.exists(model_path):
            return False
            
        # Check file extension
        _, ext = os.path.splitext(model_path.lower())
        return ext in self.model_extensions

    def _validate_dataset_path(self, dataset_path: str) -> bool:
        """Validate dataset directory exists and contains training data"""
        if not dataset_path:
            return False
            
        # Convert relative paths to absolute for existence check
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(self.project_root, dataset_path)
            
        if not os.path.exists(dataset_path):
            return False
            
        return self._has_training_images(dataset_path)

    def _has_training_images(self, directory: str) -> bool:
        """Check if directory contains training images"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
        for file in os.listdir(directory):
            _, ext = os.path.splitext(file.lower())
            if ext in image_extensions:
                return True
        return False

    def _validate_learning_rates(self, config: Dict, errors: List[str]):
        """Validate learning rate values"""
        unet_lr = config.get('unet_lr')
        if unet_lr is not None:
            try:
                lr_val = float(unet_lr)
                if lr_val <= 0 or lr_val > 1:
                    errors.append("❌ U-Net learning rate must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("❌ U-Net learning rate must be a valid number")
        
        te_lr = config.get('text_encoder_lr')
        if te_lr is not None:
            try:
                lr_val = float(te_lr)
                if lr_val < 0 or lr_val > 1:
                    errors.append("❌ Text encoder learning rate must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("❌ Text encoder learning rate must be a valid number")

    def _validate_optimizer(self, config: Dict, errors: List[str]):
        """Validate optimizer selection"""
        optimizer = config.get('optimizer')
        if optimizer and optimizer not in self.valid_optimizers:
            errors.append(f"❌ Invalid optimizer: {optimizer}. Valid options: {', '.join(sorted(self.valid_optimizers))}")

    def _validate_scheduler(self, config: Dict, errors: List[str]):
        """Validate scheduler selection"""
        scheduler = config.get('lr_scheduler')
        if scheduler and scheduler not in self.valid_schedulers:
            errors.append(f"❌ Invalid scheduler: {scheduler}. Valid options: {', '.join(sorted(self.valid_schedulers))}")

    def _validate_network_config(self, config: Dict, errors: List[str]):
        """Validate network/LoRA configuration"""
        network_dim = config.get('network_dim')
        if network_dim is not None:
            try:
                dim_val = int(network_dim)
                if dim_val <= 0 or dim_val > 1024:
                    errors.append("❌ Network dimension must be between 1 and 1024")
            except (ValueError, TypeError):
                errors.append("❌ Network dimension must be a valid integer")
        
        network_alpha = config.get('network_alpha')
        if network_alpha is not None:
            try:
                alpha_val = float(network_alpha)
                if alpha_val < 0:
                    errors.append("❌ Network alpha must be non-negative")
            except (ValueError, TypeError):
                errors.append("❌ Network alpha must be a valid number")

    def _validate_training_config(self, config: Dict, errors: List[str]):
        """Validate training configuration parameters"""
        epochs = config.get('epochs')
        if epochs is not None:
            try:
                epoch_val = int(epochs)
                if epoch_val <= 0:
                    errors.append("❌ Number of epochs must be positive")
            except (ValueError, TypeError):
                errors.append("❌ Number of epochs must be a valid integer")
        
        batch_size = config.get('batch_size')
        if batch_size is not None:
            try:
                batch_val = int(batch_size)
                if batch_val <= 0:
                    errors.append("❌ Batch size must be positive")
            except (ValueError, TypeError):
                errors.append("❌ Batch size must be a valid integer")
        
        resolution = config.get('resolution')
        if resolution is not None:
            try:
                res_val = int(resolution)
                if res_val not in [512, 768, 1024]:
                    errors.append("❌ Resolution must be 512, 768, or 1024")
            except (ValueError, TypeError):
                errors.append("❌ Resolution must be a valid integer")

    def _validate_advanced_parameters(self, config: Dict, errors: List[str]):
        """Validate advanced parameters we just added"""

        # Caption dropout validation
        caption_dropout_rate = config.get('caption_dropout_rate')
        if caption_dropout_rate is not None:
            try:
                rate = float(caption_dropout_rate)
                if rate < 0 or rate > 1:
                    errors.append("❌ Caption dropout rate must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("❌ Caption dropout rate must be a valid number")

        caption_tag_dropout_rate = config.get('caption_tag_dropout_rate')
        if caption_tag_dropout_rate is not None:
            try:
                rate = float(caption_tag_dropout_rate)
                if rate < 0 or rate > 1:
                    errors.append("❌ Caption tag dropout rate must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("❌ Caption tag dropout rate must be a valid number")

        # Caption dropout scheduling validation
        caption_dropout_every_n = config.get('caption_dropout_every_n_epochs')
        if caption_dropout_every_n is not None:
            try:
                n_val = int(caption_dropout_every_n)
                if n_val <= 0:
                    errors.append("❌ Caption dropout every N epochs must be positive")
            except (ValueError, TypeError):
                errors.append("❌ Caption dropout every N epochs must be a valid integer")

        # Bucketing validation improvements
        min_bucket = config.get('min_bucket_reso')
        max_bucket = config.get('max_bucket_reso')
        if min_bucket is not None and max_bucket is not None:
            try:
                min_val = int(min_bucket)
                max_val = int(max_bucket)
                if min_val >= max_val:
                    errors.append("❌ Minimum bucket resolution must be less than maximum bucket resolution")
                if min_val < 256:
                    errors.append("❌ Minimum bucket resolution should be at least 256")
                if max_val > 2048:
                    errors.append("⚠️ Maximum bucket resolution above 2048 may cause memory issues")
            except (ValueError, TypeError):
                errors.append("❌ Bucket resolutions must be valid integers")

        # Network args validation for LyCORIS
        if config.get('conv_dim') is not None:
            try:
                conv_dim = int(config['conv_dim'])
                if conv_dim <= 0:
                    errors.append("❌ Convolution dimension must be positive")
                elif conv_dim > 512:
                    errors.append("⚠️ Convolution dimension above 512 may cause memory issues")
            except (ValueError, TypeError):
                errors.append("❌ Convolution dimension must be a valid integer")

        # Validate rank_dropout and module_dropout
        for dropout_name in ['rank_dropout', 'module_dropout']:
            dropout_val = config.get(dropout_name)
            if dropout_val is not None:
                try:
                    rate = float(dropout_val)
                    if rate < 0 or rate > 1:
                        errors.append(f"❌ {dropout_name.replace('_', ' ').title()} must be between 0 and 1")
                except (ValueError, TypeError):
                    errors.append(f"❌ {dropout_name.replace('_', ' ').title()} must be a valid number")

def validate_training_config(config: Dict, project_root: str = None) -> Tuple[bool, List[str]]:
    """
    Convenience function for validating training configuration.
    
    Args:
        config: Widget configuration dictionary
        project_root: Project root directory (optional)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = CustomLoRAValidator(project_root)
    return validator.validate_widget_config(config)

def validate_toml_files(config_path: str, dataset_path: str, project_root: str = None) -> Tuple[bool, List[str]]:
    """
    Convenience function for validating generated TOML files.
    
    Args:
        config_path: Path to config.toml
        dataset_path: Path to dataset.toml  
        project_root: Project root directory (optional)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = CustomLoRAValidator(project_root)
    return validator.validate_generated_tomls(config_path, dataset_path)