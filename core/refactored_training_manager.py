# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/refactored_training_manager.py
"""
Compatibility wrapper for existing widget system.

This provides the same interface as HybridTrainingManager but delegates to the
new KohyaTrainingManager that leverages Kohya's library system.
"""

import logging
import os
# Suppress FutureWarnings at import time
import warnings

from .kohya_training_manager import KohyaTrainingManager

# Suppress FutureWarnings at import time
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

logger = logging.getLogger(__name__)


class RefactoredTrainingManager:
    """
    Compatibility wrapper that maintains the existing interface while using
    the new Kohya-based backend.
    
    This allows widgets to work without changes while benefiting from the
    robust Kohya library system underneath.
    """

    def __init__(self):
        """Initialize with Kohya backend"""
        self.kohya_manager = KohyaTrainingManager()

        # Maintain compatibility with existing widget expectations
        self.project_root = self.kohya_manager.project_root
        self.trainer_dir = self.kohya_manager.trainer_dir
        self.config_dir = self.kohya_manager.config_dir
        self.sd_scripts_dir = self.kohya_manager.sd_scripts_dir
        self.output_dir = self.kohya_manager.output_dir
        self.logging_dir = self.kohya_manager.logging_dir
        self.process = None

        # Legacy attribute compatibility
        self.advanced_optimizers = self._get_legacy_optimizers()
        self.standard_optimizers = self._get_legacy_standard_optimizers()

        # Reduced verbosity for startup - only log if debug mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("RefactoredTrainingManager initialized with Kohya backend")

    def _get_legacy_optimizers(self):
        """Provide legacy optimizer format for widget compatibility"""
        kohya_optimizers = self.kohya_manager.supported_optimizers
        return {name: info for name, info in kohya_optimizers.items() if not info.get('stable', True)}

    def _get_legacy_standard_optimizers(self):
        """Provide legacy standard optimizer format for widget compatibility"""
        kohya_optimizers = self.kohya_manager.supported_optimizers
        return {name: info for name, info in kohya_optimizers.items() if info.get('stable', True)}

    def _create_config_toml(self, config):
        """Legacy method name compatibility"""
        return self.kohya_manager.create_config_toml(config)

    def _create_dataset_toml(self, config):
        """Legacy dataset TOML creation - now handled by Kohya config system"""
        # Kohya's config system handles dataset configuration differently
        # This is maintained for compatibility but delegates to Kohya
        logger.info("Dataset configuration now handled by Kohya config system")
        return None  # Kohya handles this in the main config

    def _find_training_script(self):
        """Legacy method - now uses model type detection"""
        # This method was used by widgets, now we use model type detection
        logger.info("Training script selection now uses automatic model type detection")
        return "auto-detected"

    def start_training(self, config, monitor_widget=None):
        """
        Start training - delegates to Kohya manager
        """
        logger.info("🔄 Delegating training to Kohya backend")
        return self.kohya_manager.start_training(config, monitor_widget)

    def launch_from_files(self, config_paths, monitor_widget=None):
        """
        Launch training from configuration files.
        This is a compatibility method for the TrainingMonitorWidget.
        """
        import toml

        # Find any config file (Kohya generates dynamic filenames)
        config_path = None
        for filename, path in config_paths.items():
            if path and filename.endswith('_config.toml'):
                config_path = path
                break

        # Fallback to old behavior if no dynamic config found
        if not config_path:
            config_path = config_paths.get('config.toml')

        if not config_path or not os.path.exists(config_path):
            logger.error(f"Config file not found at path: {config_path}")
            return

        with open(config_path, 'r') as f:
            config = toml.load(f)

        # The new start_training method expects a dictionary, not a path
        self.start_training(config, monitor_widget)

    def stop_training(self):
        """Stop training - delegates to Kohya manager"""
        self.kohya_manager.stop_training()

    def prepare_config_only(self, config):
        """
        Generate both config and dataset files (despite the name)
        sd_scripts backend requires BOTH config.toml AND dataset.toml
        """
        config_path = self.kohya_manager.create_config_toml(config)
        dataset_path = self.kohya_manager.create_dataset_toml(config)

        # Use the actual filenames instead of hardcoded names
        if config_path and dataset_path:
            config_filename = os.path.basename(config_path)
            dataset_filename = os.path.basename(dataset_path)
            return {
                config_filename: config_path,
                dataset_filename: dataset_path
            }
        else:
            # Fallback to standard names if creation failed
            return {
                "config.toml": config_path,
                "dataset.toml": dataset_path
            }

    def validate_config(self, config):
        """Validate configuration using Kohya's validation"""
        return self.kohya_manager.validate_config(config)

    def get_model_info(self, model_path):
        """Get model information using Kohya's utilities"""
        return self.kohya_manager.get_model_info(model_path)

    # Legacy properties for widget compatibility
    @property
    def lycoris_methods(self):
        """Legacy LyCORIS methods - now handled by Kohya's network modules"""
        return {
            'LoRA': 'networks.lora',
            'LoHa': 'networks.loha',
            'LoKr': 'networks.lokr',
            'DyLoRA': 'networks.dylora',
            'LyCORIS-LoHa': 'lycoris.kohya',
            'LyCORIS-LoKr': 'lycoris.kohya',
            'LyCORIS-LoCon': 'lycoris.kohya',
            'LyCORIS-DyLoRA': 'lycoris.kohya'
        }

    @property
    def experimental_features(self):
        """Legacy experimental features"""
        return {
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'xformers': True,
            'gradient_accumulation_steps': True,
            'lr_warmup_steps': True,
            'scale_weight_norms': True,
            'noise_offset': True,
            'adaptive_noise_scale': True,
            'multires_noise_iterations': True,
            'multires_noise_discount': True,
            'ip_noise_gamma': True,
            'min_snr_gamma': True
        }

    @property
    def logging_options(self):
        """Legacy logging options"""
        return {
            'none': {'enabled': False},
            'wandb': {'enabled': True, 'project': 'lora_training'},
            'tensorboard': {'enabled': True, 'log_dir': self.logging_dir}
        }


# Create an alias for backward compatibility
HybridTrainingManager = RefactoredTrainingManager
