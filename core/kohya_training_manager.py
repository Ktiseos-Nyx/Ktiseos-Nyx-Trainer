# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# core/kohya_training_manager.py
"""
Refactored Training Manager leveraging Kohya's library system instead of custom implementations.

This replaces our custom training managers with a lightweight wrapper around Kohya's existing
strategy pattern and model utilities. We maintain our widget-friendly interface while using
battle-tested backend implementations.
"""

import gc
import os
import subprocess
import sys
from typing import Any, Dict, List

import toml

# Import Kohya's library system (required for training)
kohya_path = os.path.join(os.getcwd(), "trainer", "derrian_backend", "sd_scripts")
sys.path.insert(0, kohya_path)

# Kohya SS library imports (core training logic)
KOHYA_AVAILABLE = False
try:
    import library.config_util as config_util
    import library.train_util as train_util
    from library import model_util
    # Strategy imports for different model types
    from library.strategy_base import (TextEncoderOutputsCachingStrategy,
                                       TextEncodingStrategy, TokenizeStrategy)
    from library.strategy_flux import (FluxTextEncodingStrategy,
                                       FluxTokenizeStrategy)
    from library.strategy_sd import SdTextEncodingStrategy, SdTokenizeStrategy
    from library.strategy_sd3 import (Sd3TextEncodingStrategy,
                                      Sd3TokenizeStrategy)
    from library.strategy_sdxl import (SdxlTextEncoderOutputsCachingStrategy,
                                       SdxlTextEncodingStrategy,
                                       SdxlTokenizeStrategy)
    from library.utils import setup_logging
    setup_logging()
    
    # Also set up our file-based logging for easier debugging
    from .logging_config import setup_file_logging
    setup_file_logging()
    
    # If we got here, Kohya is available
    KOHYA_AVAILABLE = True
except ImportError as e:
    KOHYA_AVAILABLE = False
    raise ImportError(
        f"❌ TRAINING SYSTEM UNAVAILABLE\n"
        f"🔧 SOLUTION: Run 'python installer.py' to install required dependencies\n"
        f"📍 TECHNICAL ERROR: {str(e)}\n"
        f"📝 NOTE: This system requires Kohya's sd-scripts for training functionality"
    ) from e

# Import Derrian's utilities (validation and processing functions)
derrian_path = os.path.join(os.getcwd(), "trainer", "derrian_backend")
sys.path.insert(0, derrian_path)
try:
    # Import specific utility functions from individual Python files
    from utils.process import *
    from utils.validation import *
except ImportError as e:
    raise ImportError(f"❌ Derrian utilities not found: {e}\n"
                     f"🔧 Please run: python installer.py\n"
                     f"📝 This system requires Derrian's validation and processing utilities.") from e

# Logging is set up during import above if Kohya is available
import logging

logger = logging.getLogger(__name__)


class KohyaTrainingManager:
    """
    🧪 HYBRID TRAINING MANAGER 💥
    
    Combines the best of both worlds:
    - Kohya SS library (strategy pattern, model utilities, core training logic)
    - Derrian Distro (validation, processing, experimental features, custom optimizers)
    
    This gives us both the stability of Kohya's proven library system AND
    the advanced features and validation from Derrian's distribution.
    
    Features restored from archived system:
    - Advanced validation using Derrian's validation.py system
    - Experimental features framework (Kohya-compatible only)
    - Enhanced logging (TensorBoard, WandB)
    - Sophisticated optimizer configurations with detailed args
    - Custom optimizers (CAME, StableAdamW, Compass, etc.)
    """

    # Model type detection patterns
    MODEL_TYPE_PATTERNS = {
        'sd15': ['v1-5', 'sd-v1', 'sd_v1'],
        'sd20': ['v2-0', 'sd-v2', 'sd_v2', '768-v-ema'],
        'sdxl': ['xl-base', 'sdxl', 'xl_base', 'xl', '-xl-', '_xl_', 'illustrious', 'pony', 'noobai', 'animagine'],
        'flux': ['flux', 'FLUX'],
        'sd3': ['sd3', 'SD3']
    }

    # Script mapping for different model types
    SCRIPT_MAPPING = {
        'sd15': 'train_network.py',
        'sd20': 'train_network.py',
        'sdxl': 'sdxl_train_network.py',
        'flux': 'flux_train_network.py',
        'sd3': 'sd3_train_network.py'
    }

    def __init__(self):
        """Initialize with current working directory (notebook compatibility)"""
        self.project_root = os.getcwd()
        self.trainer_dir = os.path.join(self.project_root, "trainer")
        self.config_dir = os.path.join(self.project_root, "training_configs")
        self.sd_scripts_dir = os.path.join(self.trainer_dir, "derrian_backend", "sd_scripts")
        self.output_dir = os.path.join(self.project_root, "output")
        self.logging_dir = os.path.join(self.project_root, "logs")

        # Ensure directories exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)

        self.process = None

        # Initialize Kohya strategies for different model types
        self._init_strategies()

        # Detection systems
        self.supported_optimizers = self._get_supported_optimizers()
        self.supported_schedulers = self._get_supported_schedulers()
        self.model_configs = self._init_model_configs()
        self.memory_profiles = self._init_memory_profiles()

        # 🔧 Restore advanced features from archived system
        self._setup_custom_optimizers()
        self.advanced_optimizers = self._init_advanced_optimizers()
        self.standard_optimizers = self._init_standard_optimizers()
        self.lycoris_methods = self._init_lycoris_methods()
        self.experimental_features = self._init_experimental_features()
        self.logging_options = self._init_logging_options()

        # Reduced verbosity for startup - only log if debug mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("KohyaTrainingManager initialized with Kohya library system")

    def _init_strategies(self):
        """Initialize Kohya's strategy system for different model architectures"""

        self.strategies = {
            'sd15': {
                'tokenize': SdTokenizeStrategy,
                'text_encoding': SdTextEncodingStrategy,
            },
            'sd20': {
                'tokenize': SdTokenizeStrategy,
                'text_encoding': SdTextEncodingStrategy,
            },
            'sdxl': {
                'tokenize': SdxlTokenizeStrategy,
                'text_encoding': SdxlTextEncodingStrategy,
                'caching': SdxlTextEncoderOutputsCachingStrategy,
            },
            'flux': {
                'tokenize': FluxTokenizeStrategy,
                'text_encoding': FluxTextEncodingStrategy,
            },
            'sd3': {
                'tokenize': Sd3TokenizeStrategy,
                'text_encoding': Sd3TextEncodingStrategy,
            }
        }

    def _get_supported_optimizers(self) -> Dict[str, Dict]:
        """Get optimizers supported by current Kohya installation"""
        optimizers = {
            # Standard PyTorch optimizers
            'AdamW': {'description': 'AdamW optimizer (recommended)', 'stable': True},
            'AdamW8bit': {'description': '8-bit AdamW (memory efficient)', 'stable': True},
            'SGDNesterov': {'description': 'SGD with Nesterov momentum', 'stable': True},
            'SGDNesterov8bit': {'description': '8-bit SGD Nesterov', 'stable': True},

            # Advanced optimizers (if available)
            'Lion': {'description': 'Lion optimizer', 'stable': True},
            'Lion8bit': {'description': '8-bit Lion optimizer', 'stable': True},
            'DAdaptation': {'description': 'D-Adaptation (auto LR)', 'stable': False},
            'DAdaptAdam': {'description': 'D-Adapt Adam', 'stable': False},
            'DAdaptAdaGrad': {'description': 'D-Adapt AdaGrad', 'stable': False},
            'DAdaptLion': {'description': 'D-Adapt Lion', 'stable': False},
            'Prodigy': {'description': 'Prodigy optimizer', 'stable': False},
        }

        # Check for CAME optimizer (from Derrian distro)
        came_path = os.path.join(self.sd_scripts_dir, "library", "came_optimizer.py")
        if os.path.exists(came_path):
            optimizers['CAME'] = {'description': 'CAME optimizer (Derrian distro)', 'stable': False}

        return optimizers

    def _get_supported_schedulers(self) -> List[str]:
        """Get learning rate schedulers supported by Kohya"""
        return [
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup", "adafactor"
        ]

    def _init_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model configurations for Flux and SD3 variants"""
        return {
            'flux_dev': {
                'model_name': 'black-forest-labs/FLUX.1-dev',
                'script': 'flux_train_network.py',
                'resolution': 1024,
                'text_encoders': ['clip_l', 't5xxl'],
                'memory_base': 12,  # GB VRAM baseline
                'description': 'Flux.1 Dev - High quality diffusion model'
            },
            'flux_schnell': {
                'model_name': 'black-forest-labs/FLUX.1-schnell',
                'script': 'flux_train_network.py',
                'resolution': 1024,
                'text_encoders': ['clip_l', 't5xxl'],
                'memory_base': 10,  # Slightly more efficient
                'description': 'Flux.1 Schnell - Fast inference variant'
            },
            'sd3_medium': {
                'model_name': 'stabilityai/stable-diffusion-3-medium',
                'script': 'sd3_train_network.py',
                'resolution': 1024,
                'text_encoders': ['clip_l', 'clip_g', 't5xxl'],
                'memory_base': 14,  # SD3 is memory hungry
                'description': 'SD3 Medium - Stability AI latest'
            },
            'sd3_large': {
                'model_name': 'stabilityai/stable-diffusion-3-large',
                'script': 'sd3_train_network.py',
                'resolution': 1024,
                'text_encoders': ['clip_l', 'clip_g', 't5xxl'],
                'memory_base': 18,  # Even more memory hungry
                'description': 'SD3 Large - Maximum quality'
            },
            'auraflow': {
                'model_name': 'fal/AuraFlow',
                'script': 'flux_train_network.py',  # Uses Flux-style training
                'resolution': 1024,
                'text_encoders': ['t5xxl'],
                'memory_base': 10,
                'description': 'AuraFlow - Community Flux variant'
            }
        }

    def _init_memory_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Memory optimization profiles for Flux/SD3 training"""
        return {
            'ultra_low_memory': {  # 8-12GB VRAM
                'train_batch_size': 1,
                'gradient_accumulation_steps': 32,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': True,
                'optimizer': 'AdamW8bit',
                'cache_latents': True,
                'cache_latents_to_disk': True,
                'lowram': True,
                'blocks_to_swap': 8,  # Flux/SD3 specific
                'description': 'Maximum memory savings (8-12GB VRAM)'
            },
            'low_memory': {  # 12-16GB VRAM
                'train_batch_size': 1,
                'gradient_accumulation_steps': 16,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': True,
                'optimizer': 'AdamW8bit',
                'cache_latents': True,
                'cache_latents_to_disk': False,
                'lowram': False,
                'blocks_to_swap': 4,
                'description': 'Balanced efficiency (12-16GB VRAM)'
            },
            'standard': {  # 16-20GB VRAM
                'train_batch_size': 2,
                'gradient_accumulation_steps': 8,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': False,
                'optimizer': 'AdamW',
                'cache_latents': True,
                'cache_latents_to_disk': False,
                'lowram': False,
                'blocks_to_swap': 0,
                'description': 'Standard training (16-20GB VRAM)'
            },
            'high_performance': {  # 24GB+ VRAM
                'train_batch_size': 4,
                'gradient_accumulation_steps': 4,
                'mixed_precision': 'bf16',
                'gradient_checkpointing': False,
                'optimizer': 'AdamW',
                'cache_latents': False,  # Keep in memory
                'cache_latents_to_disk': False,
                'lowram': False,
                'blocks_to_swap': 0,
                'description': 'High performance (24GB+ VRAM)'
            }
        }

    def _init_training_presets(self) -> Dict[str, Dict[str, Any]]:
        """Training presets optimized for Flux/SD3 models"""
        return {
            'concept_learning': {
                'learning_rate': 1e-4,
                'text_encoder_lr': 5e-5,
                'epochs': 10,
                'warmup_ratio': 0.1,
                'scheduler': 'cosine_with_restarts',
                'network_dim': 16,
                'network_alpha': 8,
                'description': 'Learn new concepts and objects'
            },
            'style_training': {
                'learning_rate': 8e-5,
                'text_encoder_lr': 4e-5,
                'epochs': 15,
                'warmup_ratio': 0.05,
                'scheduler': 'cosine',
                'network_dim': 32,
                'network_alpha': 16,
                'description': 'Artistic styles and aesthetics'
            },
            'character_training': {
                'learning_rate': 1.2e-4,
                'text_encoder_lr': 6e-5,
                'epochs': 12,
                'warmup_ratio': 0.1,
                'scheduler': 'constant_with_warmup',
                'network_dim': 24,
                'network_alpha': 12,
                'description': 'Characters and specific people'
            },
            'fine_detail': {
                'learning_rate': 6e-5,
                'text_encoder_lr': 3e-5,
                'epochs': 20,
                'warmup_ratio': 0.15,
                'scheduler': 'linear',
                'network_dim': 64,
                'network_alpha': 32,
                'description': 'Fine details and complex scenes'
            }
        }

    def detect_optimal_memory_profile(self, model_type: str) -> str:
        """Auto-detect optimal memory profile based on VRAM and model"""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                model_memory_base = self.model_configs[model_type]['memory_base']

                print(f"🔍 Detected {vram_gb:.1f}GB VRAM")
                print(f"🤖 {model_type} baseline requirement: {model_memory_base}GB")

                # Account for model-specific memory requirements
                available_for_training = vram_gb - model_memory_base

                if available_for_training >= 8:
                    return 'high_performance'
                elif available_for_training >= 4:
                    return 'standard'
                elif available_for_training >= 2:
                    return 'low_memory'
                else:
                    return 'ultra_low_memory'
            else:
                print("⚠️ CUDA not available")
                return 'ultra_low_memory'
        except Exception as e:
            print(f"⚠️ Could not detect VRAM: {e}")
            return 'low_memory'

    def detect_model_type(self, model_path: str) -> str:
        """
        Detect model type from model path using Kohya's model utilities
        """
        if not KOHYA_AVAILABLE:
            logger.warning("Kohya not available, using basic detection")
            return self._basic_model_detection(model_path)

        try:
            # Guard against None/empty model path
            if not model_path:
                logger.error(f"❌ Model path is empty/None: {repr(model_path)} - Check widget model selection!")
                return 'sd15'
                
            # Use Kohya's model detection utilities
            model_path_lower = model_path.lower()

            # Check for explicit patterns first
            for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
                if any(pattern in model_path_lower for pattern in patterns):
                    logger.info(f"Detected model type: {model_type} (pattern match)")
                    return model_type

            # Use file size as additional heuristic for model type detection
            if os.path.exists(model_path) and (model_path.endswith('.safetensors') or model_path.endswith('.ckpt')):
                try:
                    file_size_gb = os.path.getsize(model_path) / (1024**3)  # Size in GB
                    logger.debug(f"Model file size: {file_size_gb:.1f}GB")
                    
                    # File size based detection (common sizes, largest first)
                    if file_size_gb > 10:  # Flux models are much larger (12-24GB)
                        logger.info(f"Detected Flux model based on file size: {file_size_gb:.1f}GB")
                        return 'flux'  
                    elif file_size_gb > 8:  # SD3 models are also large (8-12GB)
                        logger.info(f"Detected SD3 model based on file size: {file_size_gb:.1f}GB")
                        return 'sd3'
                    elif file_size_gb > 5.5:  # SDXL models are typically 6-7GB
                        logger.info(f"Detected SDXL model based on file size: {file_size_gb:.1f}GB")
                        return 'sdxl'
                    else:  # SD 1.5/2.0 models are typically 2-4GB
                        logger.info(f"Detected SD1.5/2.0 model based on file size: {file_size_gb:.1f}GB")
                        return 'sd15'
                        
                except Exception as e:
                    logger.warning(f"File size detection failed: {e}")
                    # Continue to fallback detection

        except Exception as e:
            logger.error(f"Model type detection failed: {e}")

        # Fallback to basic detection
        return self._basic_model_detection(model_path)

    def _basic_model_detection(self, model_path: str) -> str:
        """Basic model type detection fallback"""
        if not model_path:
            return 'sd15'
        model_path_lower = model_path.lower()

        for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
            if any(pattern in model_path_lower for pattern in patterns):
                return model_type

        return 'sd15'  # Safe default

    def get_training_script(self, model_type: str) -> str:
        """Get the appropriate training script for model type"""
        script_name = self.SCRIPT_MAPPING.get(model_type, 'train_network.py')
        script_path = os.path.join(self.sd_scripts_dir, script_name)

        if not os.path.exists(script_path):
            logger.warning(f"Training script not found: {script_path}")
            # Fallback to basic script
            fallback_script = os.path.join(self.sd_scripts_dir, 'train_network.py')
            if os.path.exists(fallback_script):
                return fallback_script
            else:
                raise FileNotFoundError(f"No training script found for {model_type}")

        return script_path

    def create_config_toml(self, config: Dict) -> str:
        """
        Create training configuration TOML using Kohya's config utilities
        """
        # 🚨 RESET DEMON DEBUGGING: Track when/why TOML is being recreated
        import traceback
        logger.warning("🚨 === TOML RECREATION ALERT ===")
        logger.warning("📊 create_config_toml() called!")
        logger.warning(f"📊 Call stack: {[line.strip() for line in traceback.format_stack()[-3:-1]]}")
        logger.warning(f"📊 Config project name: {config.get('output_name', 'lora')}")
        logger.warning("🚨 === END TOML RECREATION ALERT ===")
        
        model_type = config.get('model_type', 'sd15')

        if model_type in self.model_configs:
            return self._create_flux_sd3_config_toml(config)
        else:
            return self._create_standard_config_toml(config)

    def _create_flux_sd3_config_toml(self, config: Dict) -> str:
        """
        Create a TOML config for Flux/SD3 models.
        """
        model_type = config['model_type']
        model_config = self.model_configs[model_type]

        memory_profile_name = config.get('memory_profile') or self.detect_optimal_memory_profile(model_type)
        memory_profile = self.memory_profiles[memory_profile_name]

        preset_name = config.get('training_preset', 'concept_learning')
        preset = self.training_presets[preset_name]

        toml_config = {
            "model_arguments": {
                "pretrained_model_name_or_path": model_config['model_name'],
                "clip_l_path": config.get('clip_l_path'),
                "t5xxl_path": config.get('t5xxl_path'),
                "clip_g_path": config.get('clip_g_path'), # SD3 only
                "cache_latents": memory_profile.get('cache_latents', True),
                "cache_latents_to_disk": memory_profile.get('cache_latents_to_disk', True),
                "lowram": memory_profile.get('lowram', False),
            },
            "network_arguments": {
                "network_dim": config.get('network_dim', preset['network_dim']),
                "network_alpha": config.get('network_alpha', preset['network_alpha']),
                "network_module": 'networks.lora',
            },
            "training_arguments": {
                "output_dir": self.output_dir,
                "logging_dir": self.logging_dir,
                "output_name": config.get('output_name', 'lora'),
                "resolution": f"{model_config['resolution']},{model_config['resolution']}",
                "train_batch_size": memory_profile['train_batch_size'],
                "gradient_accumulation_steps": memory_profile['gradient_accumulation_steps'],
                "max_train_epochs": config.get('epochs', preset['epochs']),
                "learning_rate": config.get('learning_rate', preset['learning_rate']),
                "text_encoder_lr": config.get('text_encoder_lr', preset['text_encoder_lr']),
                "lr_scheduler": preset['scheduler'],
                "lr_warmup_ratio": preset['warmup_ratio'],
                "optimizer_type": memory_profile['optimizer'],
                "mixed_precision": memory_profile['mixed_precision'],
                "gradient_checkpointing": memory_profile['gradient_checkpointing'],
                "save_precision": 'fp16',
                "save_model_as": 'safetensors',
                "save_every_n_epochs": config.get('save_every_n_epochs', 1),
                "seed": config.get('seed', 42),
            },
            "flux_sd3_specific": {
                "blocks_to_swap": memory_profile.get('blocks_to_swap', 0),
            }
        }

        # Add model_type to the top level of the config so the training script can be selected correctly.
        if config.get('model_type'):
            toml_config['model_type'] = config.get('model_type')

        config_path = os.path.join(self.config_dir, f"{config.get('output_name', 'lora')}_config.toml")
        with open(config_path, 'w') as f:
            toml.dump(toml_config, f)

        logger.info(f"Created Flux/SD3 config TOML: {config_path}")
        return config_path

    def _create_standard_config_toml(self, config: Dict) -> str:
        """Fallback TOML config creation for SD1.5/SDXL"""
        config_path = os.path.join(self.config_dir, f"{config.get('output_name', 'lora')}_config.toml")

        # 🎭 === LIN-MANUEL MIRANDA DEBUGGING MODE ===
        logger.info("🎭 === WIDGET CONFIG DEBUG DUMP ===")
        logger.info(f"📊 Full config keys: {list(config.keys())}")
        logger.info(f"📊 Config type: {type(config)}")
        
        # Check if this is structured config from widget
        if 'model_arguments' in config:
            logger.info("🎵 WAIT FOR IT... Structured widget config detected!")
            logger.info(f"📊 model_arguments: {config.get('model_arguments', {})}")
            logger.info(f"📊 training_arguments: {config.get('training_arguments', {})}")
            logger.info(f"📊 network_arguments: {config.get('network_arguments', {})}")
        else:
            logger.info("🎵 HAMILTON'S WAY... Looking for flat config fields")
            logger.info(f"📊 model_path: {repr(config.get('model_path'))}")
            logger.info(f"📊 dataset_dir: {repr(config.get('dataset_dir'))}")
            logger.info(f"📊 dataset_path: {repr(config.get('dataset_path'))}")
            logger.info(f"📊 output_dir: {repr(config.get('output_dir'))}")

        # 🎵 FIXED FIELD MAPPING: Use EXACT widget field names from debug output!
        # Widget debug showed: 'model_path', 'train_batch_size', 'unet_lr', etc.
        # Get network configuration based on LoRA type selection
        network_config = self._get_network_configuration(config.get('lora_type', ''), config)
        
        toml_config = {
            "network_arguments": {
                "network_dim": config.get('network_dim'),           # Widget provides this
                "network_alpha": config.get('network_alpha'),       # Widget provides this
                "conv_dim": config.get('conv_dim'),                 # Widget provides this for LyCORIS
                "conv_alpha": config.get('conv_alpha'),             # Widget provides this for LyCORIS
                **network_config  # Add network_module and network_args based on LoRA type
            },
            "optimizer_arguments": {
                "learning_rate": config.get('unet_lr'),             # Widget provides 'unet_lr'
                "text_encoder_lr": config.get('text_encoder_lr'),   # Widget provides this
                "lr_scheduler": config.get('lr_scheduler'),         # Widget provides this  
                "lr_scheduler_num_cycles": config.get('lr_scheduler_number'),  # Widget provides this
                "lr_warmup_ratio": config.get('lr_warmup_ratio'),   # Widget provides this
                "optimizer_type": config.get('optimizer'),          # Widget provides 'optimizer'
                "max_grad_norm": config.get('max_grad_norm'),       # Widget provides this
                **self._get_optimizer_arguments(config.get('optimizer', ''))  # Add optimizer-specific args
            },
            "training_arguments": {
                "pretrained_model_name_or_path": self._get_absolute_model_path(config.get('model_path', '')),  # Convert to absolute path
                "max_train_epochs": config.get('epochs'),                       # Widget provides 'epochs'
                "train_batch_size": config.get('train_batch_size'),             # Widget provides 'train_batch_size'
                "save_every_n_epochs": config.get('save_every_n_epochs'),       # Widget provides this
                "keep_only_last_n_epochs": config.get('keep_only_last_n_epochs'),  # Widget provides this
                "mixed_precision": config.get('precision'),                     # Widget provides 'precision'
                "output_dir": self.output_dir,                                   # Absolute path to output directory
                "output_name": config.get('project_name', 'lora'),              # Widget provides 'project_name'
                "clip_skip": config.get('clip_skip', 2),                        # Widget provides this
                "save_model_as": "safetensors",                                  # Static format
                "seed": 42,                                                      # Static seed
                # Performance and memory optimization
                "gradient_checkpointing": config.get('gradient_checkpointing', True),  # Widget provides this
                "gradient_accumulation_steps": config.get('gradient_accumulation_steps', 1),  # Widget provides this
                "cache_latents": config.get('cache_latents', True),             # Widget provides this
                "cache_latents_to_disk": config.get('cache_latents_to_disk', True),  # Widget provides this
                "cache_text_encoder_outputs": config.get('cache_text_encoder_outputs', False),  # Widget provides this
                "vae_batch_size": config.get('vae_batch_size', 1),              # Widget provides this
                "no_half_vae": config.get('no_half_vae', False),                # Widget provides this
                # Model variant and training mode settings
                "v2": config.get('v2', False),                                  # Widget provides this
                "v_parameterization": config.get('v_parameterization', False), # Widget provides this
                "zero_terminal_snr": config.get('zero_terminal_snr', False),   # Widget provides this
                "network_train_unet_only": config.get('network_train_unet_only', False),  # Widget provides this
                # Cross attention and precision settings
                "xformers": config.get('cross_attention') == 'xformers',       # Widget provides cross_attention
                "sdpa": config.get('cross_attention') == 'sdpa',               # Widget provides cross_attention
                "fp8_base": config.get('fp8_base', False),                     # Widget provides this
                "full_fp16": config.get('full_fp16', False),                   # Widget provides this
                # Noise and training stability
                "noise_offset": config.get('noise_offset', 0.0),              # Widget provides this
                "min_snr_gamma": config.get('min_snr_gamma') if config.get('min_snr_gamma_enabled') else None,  # Widget provides both
                "ip_noise_gamma": config.get('ip_noise_gamma') if config.get('ip_noise_gamma_enabled') else None,  # Widget provides both
                "multires_noise_iterations": 6 if config.get('multinoise') else None,  # Widget provides multinoise
                "adaptive_noise_scale": config.get('adaptive_noise_scale') if config.get('adaptive_noise_scale', 0) > 0 else None,  # Widget provides this
                # Caption handling
                "caption_dropout_rate": config.get('caption_dropout_rate', 0.0),  # Widget provides this
                "caption_tag_dropout_rate": config.get('caption_tag_dropout_rate', 0.0),  # Widget provides this
                "caption_dropout_every_n_epochs": config.get('caption_dropout_every_n_epochs'),  # Advanced caption dropout scheduling
                "keep_tokens": config.get('keep_tokens', 0),                   # Widget provides this
                "keep_tokens_separator": config.get('keep_tokens_separator'),  # Separator for keep_tokens feature
                "shuffle_caption": config.get('shuffle_caption', False),      # Randomly shuffle caption order
                "secondary_separator": config.get('secondary_separator'),     # Additional separator for captions
                "enable_wildcard": config.get('enable_wildcard', False),      # Support wildcard notation in captions
                # Data augmentation
                "random_crop": config.get('random_crop', False),               # Widget provides this
                "color_aug": config.get('color_aug', False),                  # Color augmentation
                "flip_aug": config.get('flip_aug', False),                    # Horizontal flip augmentation
            },
        }

        # Add model_type to the top level of the config so the training script can be selected correctly.
        if config.get('model_type'):
            toml_config['model_type'] = config.get('model_type')

        # Ensure numeric values are properly typed for TOML
        self._fix_numeric_types(toml_config)
        
        with open(config_path, 'w') as f:
            toml.dump(toml_config, f)

        logger.info(f"Created config TOML: {config_path}")
        return config_path

    def _fix_numeric_types(self, config_dict: Dict):
        """
        Recursively convert string representations of numbers to proper numeric types for TOML.
        Prevents 'TypeError: full_like(): argument fill_value must be Number, not str'
        """
        numeric_fields = {
            'min_snr_gamma', 'max_grad_norm', 'noise_offset', 'lr_warmup_ratio',
            'learning_rate', 'text_encoder_lr', 'unet_lr', 'weight_decay',
            'caption_dropout_rate', 'caption_tag_dropout_rate', 'keep_tokens',
            'train_batch_size', 'gradient_accumulation_steps', 'max_train_epochs',
            'save_every_n_epochs', 'keep_only_last_n_epochs', 'clip_skip',
            'vae_batch_size', 'network_dim', 'network_alpha', 'conv_dim', 'conv_alpha'
        }
        
        def convert_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in numeric_fields and value is not None:
                        try:
                            # Convert string numbers to proper numeric types
                            if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                                obj[key] = float(value) if '.' in value else int(value)
                        except (ValueError, AttributeError):
                            pass  # Keep original value if conversion fails
                    else:
                        convert_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    convert_recursive(item)
        
        convert_recursive(config_dict)

    def create_dataset_toml(self, config: Dict) -> str:
        """
        Create dataset configuration TOML using REAL Kohya format from your resources!
        """
        # 🚨 RESET DEMON DEBUGGING: Track dataset TOML recreation
        import traceback
        logger.warning("🚨 === DATASET TOML RECREATION ALERT ===")
        logger.warning("📊 create_dataset_toml() called!")
        logger.warning(f"📊 Call stack: {[line.strip() for line in traceback.format_stack()[-3:-1]]}")
        logger.warning(f"📊 Dataset path from config: {config.get('dataset_path')}")
        logger.warning("🚨 === END DATASET TOML RECREATION ALERT ===")
        
        dataset_toml_path = os.path.join(self.config_dir, f"{config.get('output_name', 'lora')}_dataset.toml")
        
        # 🎭 DATASET DEBUG: Check critical fields before TOML generation
        logger.info("🎭 === DATASET TOML DEBUG ===")
        logger.info(f"📊 dataset_path from widget: {repr(config.get('dataset_path'))}")
        logger.info(f"📊 resolution from widget: {repr(config.get('resolution'))}")
        logger.info(f"📊 num_repeats from widget: {repr(config.get('num_repeats'))}")
        
        # 🎵 EXACT WORKING TOML STRUCTURE: Match your working dataset.toml format!
        # Filter out None values so TOML only gets fields that are actually set
        
        # Build datasets section
        datasets_section = {}
        subsets_section = {}
        general_section = {}
        
        # Only add fields that aren't None
        if config.get('keep_tokens') is not None:
            datasets_section['keep_tokens'] = config.get('keep_tokens')
            
        # 🚨 CRITICAL FIX: Widget provides 'dataset_path' NOT 'dataset_dir'!
        if config.get('num_repeats') is not None:
            subsets_section['num_repeats'] = config.get('num_repeats')
        if config.get('dataset_path') is not None:                          # FIXED: Use 'dataset_path' from widget
            dataset_path = config.get('dataset_path')

            # Ensure the dataset path is absolute before calculating the relative path.
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.join(self.project_root, dataset_path)

            # Calculate the relative path from the scripts directory to the dataset directory.
            # This is more robust than hardcoding '../../..'.
            subsets_section['image_dir'] = os.path.relpath(dataset_path, self.sd_scripts_dir)
        if config.get('class_tokens') is not None:
            subsets_section['class_tokens'] = config.get('class_tokens')
            
        # 🚨 CRITICAL: Resolution is REQUIRED for training!
        # Kohya expects resolution as a single INTEGER, not a comma-separated string
        resolution = config.get('resolution')
        if resolution is not None:
            # Convert various formats to single integer
            if isinstance(resolution, (list, tuple)):
                # Take the first value from [1024, 1024] → 1024
                general_section['resolution'] = int(resolution[0])
            elif isinstance(resolution, str) and ',' in resolution:
                # Take first value from "1024,1024" → 1024
                general_section['resolution'] = int(resolution.split(',')[0])
            elif isinstance(resolution, (int, str)):
                # Single value - convert to int
                general_section['resolution'] = int(resolution)
        else:
            # Fallback to safe default - training CANNOT proceed without resolution!
            general_section['resolution'] = 512  # Integer, not string!
        if config.get('shuffle_caption') is not None:
            general_section['shuffle_caption'] = config.get('shuffle_caption')
        if config.get('flip_aug') is not None:
            general_section['flip_aug'] = config.get('flip_aug')
        # Always specify caption extension - default to .txt if not provided
        general_section['caption_extension'] = config.get('caption_extension', '.txt')
        
        # Bucketing settings - now properly from widget
        if config.get('enable_bucket') is not None:
            general_section['enable_bucket'] = config.get('enable_bucket')
        if config.get('bucket_no_upscale') is not None:
            general_section['bucket_no_upscale'] = config.get('bucket_no_upscale')
        
        # Handle bucket_reso_steps with SDXL optimization
        bucket_steps = config.get('bucket_reso_steps')
        if bucket_steps is not None:
            general_section['bucket_reso_steps'] = bucket_steps
        elif config.get('sdxl_bucket_optimization'):
            general_section['bucket_reso_steps'] = 32  # SDXL optimized
        else:
            general_section['bucket_reso_steps'] = 64  # Standard
            
        if config.get('min_bucket_reso') is not None:
            general_section['min_bucket_reso'] = config.get('min_bucket_reso')
        if config.get('max_bucket_reso') is not None:
            general_section['max_bucket_reso'] = config.get('max_bucket_reso')
            
        # Caption handling settings from widget
        if config.get('caption_dropout_rate') is not None:
            general_section['caption_dropout_rate'] = config.get('caption_dropout_rate')
        if config.get('caption_tag_dropout_rate') is not None:
            general_section['caption_tag_dropout_rate'] = config.get('caption_tag_dropout_rate')
        if config.get('caption_dropout_every_n_epochs') is not None:
            general_section['caption_dropout_every_n_epochs'] = config.get('caption_dropout_every_n_epochs')
        if config.get('keep_tokens_separator') is not None:
            general_section['keep_tokens_separator'] = config.get('keep_tokens_separator')
        if config.get('secondary_separator') is not None:
            general_section['secondary_separator'] = config.get('secondary_separator')
        if config.get('enable_wildcard') is not None:
            general_section['enable_wildcard'] = config.get('enable_wildcard')

        # Data augmentation settings
        if config.get('color_aug') is not None:
            general_section['color_aug'] = config.get('color_aug')
        if config.get('random_crop') is not None:
            general_section['random_crop'] = config.get('random_crop')

        # Build final structure
        dataset_config = {
            "datasets": [datasets_section] if datasets_section else [{}],
            "general": general_section
        }
        
        # Add subsets to the first dataset
        if subsets_section:
            dataset_config["datasets"][0]["subsets"] = [subsets_section]
        
        with open(dataset_toml_path, 'w') as f:
            toml.dump(dataset_config, f)
            
        logger.info(f"Created dataset TOML: {dataset_toml_path}")
        return dataset_toml_path

    def start_training(self, config: Dict, monitor_widget=None) -> bool:
        """
        Start training using Kohya's training scripts and our configuration
        Handles both flat widget config and structured TOML config formats
        """
        logger.info("🚀 Starting training with Kohya backend")

        try:
            # 🕵️ DETECT CONFIG FORMAT: Widget (flat) vs TOML (structured)
            is_structured_toml = self._is_structured_toml_config(config)
            
            if is_structured_toml:
                logger.info("📋 Received structured TOML config from launch_from_files()")
                # Already in TOML format - write to files and proceed
                return self._launch_training_from_structured_config(config, monitor_widget)
            else:
                logger.info("📋 Received flat widget config from prepare_config_only()")
                # Need to generate TOML files first
                return self._launch_training_from_widget_config(config, monitor_widget)
                
        except Exception as e:
            logger.error(f"💥 Training failed: {e}")
            return False

    def _is_structured_toml_config(self, config: Dict) -> bool:
        """Detect if config is structured TOML format vs flat widget format"""
        # TOML structure has these top-level sections
        toml_sections = ['network_arguments', 'optimizer_arguments', 'training_arguments']
        return any(section in config for section in toml_sections)

    def _launch_training_from_widget_config(self, config: Dict, monitor_widget=None) -> bool:
        """Handle flat widget config - generate TOML then train"""
        logger.info("🍬 Simple wrapper approach - generating TOML and letting sd-scripts do the work")
        
        # 🧠 LIN-MANUEL MIRANDA DEBUGGING MODE: "WHY DO YOU DEBUG LIKE YOU'RE RUNNING OUT OF TIME?"
        logger.info("🎭 === WIDGET CONFIG DEBUG DUMP ===")
        logger.info(f"📊 dataset_dir: {repr(config.get('dataset_dir'))}")
        logger.info(f"📊 dataset_path: {repr(config.get('dataset_path'))}")  
        logger.info(f"📊 output_dir: {repr(config.get('output_dir'))}")
        logger.info(f"📊 model_path: {repr(config.get('model_path'))}")
        logger.info(f"📊 project_name: {repr(config.get('project_name'))}")
        logger.info(f"📊 Full config keys: {list(config.keys())}")
        logger.info("🎭 === END CONFIG DUMP ===")

        # 🍬 PURE CANDY WRAPPER: Use our working TOML generation directly!
        # No more Derrian functions, no more undefined variables, just WORKING CODE!
        logger.info("🍬 Using candy wrapper TOML generation (no Derrian validation)")
        
        # Generate TOML files using our proven working methods
        config_path = self.create_config_toml(config)
        dataset_path = self.create_dataset_toml(config)
        
        logger.info(f"✅ Generated config: {config_path}")
        logger.info(f"✅ Generated dataset config: {dataset_path}")
        
        return self._execute_training_command(config_path, dataset_path, monitor_widget)

    def _launch_training_from_structured_config(self, config: Dict, monitor_widget=None) -> bool:
        """Handle structured TOML config - USE EXISTING FILES, DON'T WRITE NEW ONES!"""
        logger.info("🎯 Using existing TOML files (NOT writing new ones!)")
        
        # 🚨 FIXED: DON'T WRITE TOML FILES! Just use existing ones!
        # The second button should NEVER overwrite the good files from the first button
        # Look for existing files that the first button created (project_name_config.toml format)
        import glob
        config_files = glob.glob(os.path.join(self.config_dir, "*_config.toml"))
        dataset_files = glob.glob(os.path.join(self.config_dir, "*_dataset.toml"))
        
        if config_files:
            config_path = config_files[0]  # Use first match
        else:
            config_path = os.path.join(self.config_dir, "config.toml")  # Fallback
            
        if dataset_files:
            dataset_path = dataset_files[0]  # Use first match  
        else:
            dataset_path = os.path.join(self.config_dir, "dataset.toml")  # Fallback
        
        logger.info(f"📁 Using existing config: {config_path}")
        logger.info(f"📁 Using existing dataset: {dataset_path}")
        
        # Verify files exist
        if not os.path.exists(config_path):
            logger.error(f"❌ Config file not found: {config_path}")
            return False
        if not os.path.exists(dataset_path):
            logger.error(f"❌ Dataset file not found: {dataset_path}")
            return False
        
        logger.info("✅ Found existing TOML files - launching training!")
        return self._execute_training_command(config_path, dataset_path, monitor_widget)

    def _write_structured_config_toml(self, config: Dict) -> str:
        """Write pre-structured TOML config to file"""
        # 🚨 RESET DEMON DEBUGGING: Track structured TOML writes
        import traceback
        logger.warning("🚨 === STRUCTURED CONFIG WRITE ALERT ===")
        logger.warning("📊 _write_structured_config_toml() called!")
        logger.warning(f"📊 Call stack: {[line.strip() for line in traceback.format_stack()[-3:-1]]}")
        logger.warning("🚨 === END STRUCTURED CONFIG ALERT ===")
        
        # Use standard naming that launch_from_files() expects
        config_filename = "config.toml"
        config_path = os.path.join(self.config_dir, config_filename)
        
        # Extract just the training sections for config.toml
        config_sections = {
            'network_arguments': config.get('network_arguments', {}),
            'optimizer_arguments': config.get('optimizer_arguments', {}),
            'training_arguments': config.get('training_arguments', {})
        }
        
        with open(config_path, 'w') as f:
            toml.dump(config_sections, f)
        
        return config_path

    def _write_structured_dataset_toml(self, config: Dict) -> str:
        """Write pre-structured dataset config to file with proper Kohya format"""
        # Use standard naming that launch_from_files() expects
        dataset_filename = "dataset.toml"
        dataset_path = os.path.join(self.config_dir, dataset_filename)
        
        # 🎯 CRITICAL FIX: Build proper Kohya dataset structure, not just a dump!
        # We need the exact structure that the working create_dataset_toml method uses
        
        datasets_section = {}
        subsets_section = {}
        general_section = config.get('general', {})
        
        # Extract from our structured config's datasets array
        if config.get('datasets') and len(config['datasets']) > 0:
            first_dataset = config['datasets'][0]
            if 'subsets' in first_dataset and len(first_dataset['subsets']) > 0:
                subset = first_dataset['subsets'][0]
                
                # Build subsets section
                if subset.get('num_repeats') is not None:
                    subsets_section['num_repeats'] = subset['num_repeats']
                if subset.get('image_dir') is not None:
                    subsets_section['image_dir'] = subset['image_dir']
                    
        # Fix resolution format - Kohya needs INTEGER not "1024,1024" string
        if 'resolution' in general_section:
            resolution = general_section['resolution']
            if isinstance(resolution, str) and ',' in resolution:
                # Convert "1024,1024" → 1024 (integer)
                general_section['resolution'] = int(resolution.split(',')[0])
            elif isinstance(resolution, (int, str)):
                general_section['resolution'] = int(resolution)
                
        # Build final structure matching working create_dataset_toml
        dataset_config = {
            "datasets": [datasets_section] if datasets_section else [{}],
            "general": general_section
        }
        
        # Add subsets to the first dataset if we have any
        if subsets_section:
            dataset_config["datasets"][0]["subsets"] = [subsets_section]
        
        # 🧠 DEBUGGING: Log what we're actually writing
        logger.info("🎯 === STRUCTURED DATASET TOML DEBUG ===")
        logger.info(f"📊 Writing dataset config: {dataset_config}")
        logger.info(f"📊 Subsets section: {subsets_section}")
        logger.info(f"📊 General section: {general_section}")
        logger.info("🎯 === END DATASET TOML DEBUG ===")
        
        with open(dataset_path, 'w') as f:
            toml.dump(dataset_config, f)
        
        return dataset_path

    def _execute_training_command(self, config_path: str, dataset_path: str, monitor_widget=None) -> bool:
        """Execute the actual training command with proper model type detection"""
        try:
            # Read the config to detect model type
            import toml
            with open(config_path, 'r') as f:
                config = toml.load(f)
            
            # Use the user's selected model type from dropdown (not auto-detection!)
            raw_model_type = config.get('model_type', 'sd15')

            # Normalize dropdown values to script mapping keys
            model_type_mapping = {
                'sd1_5_2_0': 'sd15',  # From widget: SD1.5/2.0 -> sd1_5_2_0 -> sd15
                'sdxl': 'sdxl',       # Direct match
                'flux': 'flux',       # Direct match
                'sd3': 'sd3',         # Direct match
                'sd15': 'sd15',       # Direct match for legacy configs
            }

            if raw_model_type not in model_type_mapping:
                error_msg = f"❌ Unknown model type '{raw_model_type}' selected. Supported types: {list(model_type_mapping.keys())}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            model_type = model_type_mapping[raw_model_type]

            # Get the correct training script for the user's choice
            script_name = self.SCRIPT_MAPPING.get(model_type, 'train_network.py')

            logger.info(f"🎯 User selected model type: {model_type}")
            logger.info(f"🎯 Using training script: {script_name}")

            # Optional: Validate that the model file exists (but don't override user choice)
            model_path = config.get('training_arguments', {}).get('pretrained_model_name_or_path', '')
            if model_path and not os.path.exists(model_path):
                logger.warning(f"⚠️ Model file not found: {model_path}")
                logger.info("💡 Training will continue with user's selected type, but may fail if path is wrong")

            # Build training command with correct script
            cmd = [
                os.path.join(self.sd_scripts_dir, script_name),
                "--config_file", config_path,
                "--dataset_config", dataset_path
            ]
            logger.info(f"🚀 Running: {' '.join(cmd[-3:])}")

            # Always use current Python executable for environment-agnostic execution
            # This follows CLAUDE.md requirement: NEVER hardcode paths or environment assumptions
            python_executable = sys.executable
            cmd.insert(0, python_executable)

            # Setup environment
            env = os.environ.copy()
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            # Change to scripts directory for execution
            original_cwd = os.getcwd()
            os.chdir(self.sd_scripts_dir)

            try:
                # Launch training process
                logger.info(f"Executing: {' '.join(cmd)}")

                if monitor_widget:
                    monitor_widget.clear_log()
                    monitor_widget.update_phase("Starting Kohya training...", "info")

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )

                # Monitor output
                for line in iter(self.process.stdout.readline, ''):
                    print(line, end='')
                    if monitor_widget:
                        monitor_widget.parse_training_output(line)

                self.process.stdout.close()
                return_code = self.process.wait()

                if return_code == 0:
                    logger.info("✅ Training completed successfully!")
                    return True
                else:
                    logger.error(f"❌ Training failed with return code: {return_code}")
                    return False

            finally:
                os.chdir(original_cwd)
                self.process = None
                # Force garbage collection after training
                gc.collect()

        except Exception as e:
            logger.error(f"Training error: {e}")
            if monitor_widget:
                monitor_widget.update_phase(f"Training failed: {e}", "error")
            return False

    def _get_training_command(self, config: Dict, config_path: str, dataset_path: str) -> List[str]:
        """
        Generate the training command for the specified model type using both config files.
        """
        model_type = config.get('model_type', 'sd15')
        script_name = self.SCRIPT_MAPPING.get(model_type, 'train_network.py')
        script_path = os.path.join(self.sd_scripts_dir, script_name)

        # Proper Kohya command format with both config files
        cmd = [
            script_path, 
            "--config_file", config_path,
            "--dataset_config", dataset_path
        ]

        # Add model-specific arguments for Flux/SD3
        if model_type in ['flux', 'sd3']:
            model_config = self.model_configs[model_type]
            if config.get('clip_l_path'):
                cmd.extend(["--clip_l_path", config['clip_l_path']])
            if config.get('t5xxl_path'):
                cmd.extend(["--t5xxl_path", config['t5xxl_path']])
            if model_type == 'sd3' and config.get('clip_g_path'):
                cmd.extend(["--clip_g_path", config['clip_g_path']])

        return cmd

    def stop_training(self):
        """Stop the training process"""
        if self.process and self.process.poll() is None:
            logger.info("🛑 Stopping training process...")
            self.process.terminate()
            self.process.wait()
            self.process = None
            logger.info("Training process stopped")

    def validate_config(self, config: Dict) -> List[str]:
        """
        🍬 Candy wrapper mode - no validation, let sd-scripts handle it!
        """
        logger.info("🍬 Skipping validation - letting sd-scripts handle everything!")
        return []  # No errors, we trust sd-scripts

    def get_model_info(self, model_path: str) -> Dict:
        """
        Get model information using Kohya's model utilities
        """
        info = {
            'path': model_path,
            'type': 'unknown',
            'exists': os.path.exists(model_path),
            'size': 0
        }

        if info['exists']:
            info['size'] = os.path.getsize(model_path)
            info['type'] = self.detect_model_type(model_path)

        return info

    # ==================== RESTORED ADVANCED FEATURES ====================
    # Methods restored from the archived training manager system

    def _setup_custom_optimizers(self):
        """Setup Python path for Derrian's custom optimizers"""
        custom_optimizer_path = os.path.join(self.trainer_dir, "derrian_backend", "custom_scheduler", "LoraEasyCustomOptimizer")
        if custom_optimizer_path not in sys.path:
            sys.path.insert(0, custom_optimizer_path)

    def _init_advanced_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize advanced optimizer configurations from archived system"""
        return {
            'StableAdamW': {
                'optimizer_type': 'LoraEasyCustomOptimizer.utils.StableAdamW',
                'args': ['weight_decay=0.01'],
                'description': 'Improved stability AdamW variant'
            },
            'Compass': {
                'optimizer_type': 'LoraEasyCustomOptimizer.compass.Compass',
                'args': ['weight_decay=0.01'],
                'description': 'Compass optimizer for better convergence'
            },
            'LPF_AdamW': {
                'optimizer_type': 'LoraEasyCustomOptimizer.lpfadamw.LPF_AdamW',
                'args': ['weight_decay=0.01'],
                'description': 'Low-pass filtered AdamW'
            }
        }

    def _init_standard_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standard optimizer configurations with detailed args"""
        return {
            'Prodigy': {
                'optimizer_type': 'Prodigy',
                'args': ['decouple=True', 'weight_decay=0.01', 'betas=[0.9,0.999]', 'd_coef=2', 'use_bias_correction=True'],
                'warmup_args': ['safeguard_warmup=True'],
                'description': 'Adaptive learning rate optimizer'
            },
            'AdamW': {
                'optimizer_type': 'AdamW',
                'args': ['weight_decay=0.01', 'betas=[0.9,0.999]'],
                'description': 'Standard AdamW optimizer'
            },
            'AdamW8bit': {
                'optimizer_type': 'bitsandbytes.optim.AdamW8bit',
                'args': ['weight_decay=0.1', 'betas=[0.9,0.99]'],
                'description': '8-bit AdamW for memory efficiency'
            },
            'Came': {
                'optimizer_type': 'LoraEasyCustomOptimizer.came.CAME',
                'args': ['weight_decay=0.04'],
                'description': 'Custom CAME optimizer'
            },
            'DAdaptation': {
                'optimizer_type': 'DAdaptation',
                'args': ['weight_decay=0.01'],
                'description': 'D-Adaptation optimizer'
            },
            'Lion': {
                'optimizer_type': 'Lion',
                'args': ['weight_decay=0.01', 'betas=[0.9,0.99]'],
                'description': 'Lion optimizer'
            }
        }

    def _init_lycoris_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize LyCORIS method configurations - OFFICIAL Algo-List.md specifications"""
        return {
            'locon': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=locon'],
                'description': 'LoCon: LoRA + Convolution layers (combines Linear and Conv LoRA)',
                'recommendations': 'Applies LoRA to both Linear and Convolution layers for better detail capture.'
            },
            'loha': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=loha'],
                'description': 'LoHa: Low-rank Hadamard (dim≤32, alpha≤dim, rank≤dim²)',
                'recommendations': 'Recommended: dim≤32, alpha from 1 to dim. Higher dim may cause unstable loss.'
            },
            'lokr': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=lokr'],
                'description': 'LoKr: Low-rank Kronecker (small/large LoKr configs available)',
                'recommendations': 'Can use "full dimension" by setting very large dimension.'
            },
            'ia3': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=ia3'],
                'description': '(IA)³: Experimental method (requires higher learning rate, <1MB files)',
                'recommendations': 'Requires higher learning rate. Produces very small files (<1 MB).'
            },
            'dylora': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=dylora'],
                'description': 'DyLoRA: Updates one row/column per step (large dim recommended)',
                'recommendations': 'Recommended: large dim with alpha=dim/4 to dim.'
            },
            'boft': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=boft'],
                'description': 'BOFT: Advanced Diag-OFT using butterfly operation for orthogonal matrix',
                'recommendations': 'Advanced version of Diag-OFT with butterfly operations.'
            },
            'glora': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=glora'],
                'description': 'GLoRA: Generalized LoRA (experimental)',
                'recommendations': 'Part of experimental methods, limited documentation available.'
            },
            'full': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=full'],
                'description': 'Full: Native fine-tuning (largest files, potentially best results)',
                'recommendations': 'Can potentially give the best result if tuned correctly. Largest file but no slower training.'
            },
            'diag_oft': {
                'network_module': 'lycoris.kohya',
                'network_args': ['algo=diag-oft'],  # Fixed: uses hyphen not underscore!
                'description': 'Diag-OFT: Preserves hyperspherical energy (converges faster than LoRA)',
                'recommendations': 'Preserves hyperspherical energy. Converges faster than LoRA.'
            }
        }

    def _init_experimental_features(self) -> Dict[str, Dict[str, Any]]:
        """Initialize experimental feature configurations (Kohya-compatible only)"""
        return {
            'multi_resolution': {
                'enabled': False,
                'description': 'Train on multiple resolutions simultaneously',
                'status': 'experimental'
            },
            'advanced_loss_scaling': {
                'enabled': False,
                'description': 'Dynamic loss scaling for mixed precision',
                'status': 'experimental'
            },
            'gradient_checkpointing': {
                'enabled': False,
                'description': 'Trade compute for memory (Kohya native)',
                'status': 'stable'
            }
        }

    def _init_logging_options(self):
        """Initialize logging system options"""
        return {
            'none': {
                'description': 'No logging (standard console output only)',
                'args': []
            },
            'tensorboard': {
                'description': 'TensorBoard logging for training metrics',
                'args': ['--log_with', 'tensorboard']
            },
            'wandb': {
                'description': 'Weights & Biases logging for experiment tracking',
                'args': ['--log_with', 'wandb']
            },
            'both': {
                'description': 'Both TensorBoard and WandB logging',
                'args': ['--log_with', 'all']
            }
        }

    def _detect_repeat_folders(self, dataset_dir):
        """
        Detect existing repeat folders in dataset directory using Kohya's format
        Returns list of tuples: (folder_name, repeats, concept_name)
        """
        if not os.path.exists(dataset_dir):
            return []

        repeat_folders = []

        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                # Use the same logic as our fixed calculator
                try:
                    parts = item.split('_', 1)
                    if len(parts) >= 2:
                        repeats = int(parts[0])
                        concept_name = parts[1]
                        repeat_folders.append((item, repeats, concept_name))
                except ValueError:
                    # Not a repeat folder, skip
                    continue

        return repeat_folders

    def validate_config_with_derrian(self, config: Dict) -> tuple[bool, List[str], Dict, Dict, Dict]:
        """
        🧪 HYBRID VALIDATION - Use Derrian's advanced validation system
        This is the missing piece we lost during refactoring!
        """
        # Try to use Derrian validation, fallback to basic if not available

        try:
            # Convert our config to Derrian's expected format
            derrian_args = self._convert_config_to_derrian_format(config)

            # Use Derrian's comprehensive validation system
            # The validate function was imported via "from utils.validation import *"
            passed, sdxl, errors, validated_args, validated_dataset_args, tag_data = validate(derrian_args)

            if passed:
                logger.info("✅ Derrian validation passed - config is safe for training")
            else:
                logger.warning(f"❌ Derrian validation failed with {len(errors)} errors")
                for error in errors:
                    logger.warning(f"   - {error}")

            return passed, errors, validated_args, validated_dataset_args, tag_data

        except Exception as e:
            logger.error(f"Derrian validation failed with exception: {e}")
            logger.warning("Falling back to basic validation")
            # Fallback to basic validation
            errors = self.validate_config(config)
            return len(errors) == 0, errors, config, {}, {}

    def _convert_config_to_derrian_format(self, config: Dict) -> Dict:
        """Convert our config format to what Derrian's validation expects - REAL conversion from archived version"""
        logger.debug("⚙️ Converting widget config to Derrian backend format...")
        
        # Proper Derrian args structure (from archived training_manager.py)
        derrian_args = {
            "basic": {
                "pretrained_model_name_or_path": config.get('model_path'),
                "output_dir": config.get('output_dir', self.output_dir),
                "output_name": config.get('output_name', 'lora'),
                "save_every_n_epochs": config.get('save_every_n_epochs', 1),
                "keep_only_last_n_epochs": config.get('keep_only_last_n_epochs', 0),
                "train_batch_size": config.get('batch_size', 1),
                "mixed_precision": config.get('mixed_precision', 'fp16'),
                "save_precision": config.get('save_precision', 'fp16'),
                "max_train_epochs": config.get('epochs', 10),
                "learning_rate": config.get('unet_lr', 1e-4),
                "text_encoder_lr": config.get('text_encoder_lr', 5e-5),
                "optimizer_type": config.get('optimizer', 'AdamW8bit'),
                "lr_scheduler": config.get('lr_scheduler', 'cosine'),
                "lr_warmup_ratio": config.get('lr_warmup_ratio', 0.1),
                "network_dim": config.get('network_dim', 16),
                "network_alpha": config.get('network_alpha', 8),
                "clip_skip": config.get('clip_skip', 2),
                "cache_latents": config.get('cache_latents', True),
                "cache_latents_to_disk": config.get('cache_latents_to_disk', False),
                "xformers": config.get('cross_attention') == "xformers",
                "sdpa": config.get('cross_attention') == "sdpa",
            },
            "sdxl": {
                "cache_text_encoder_outputs": config.get('cache_text_encoder_outputs', False),
            },
            "network": {
                "type": config.get('lora_type', 'lora'),
                "algo": config.get('lycoris_method', 'none'),
            }
        }

        # Proper Derrian dataset structure
        derrian_dataset = {
            "general": {
                "resolution": config.get('resolution', 512),
                "shuffle_caption": config.get('shuffle_caption', True),
                "keep_tokens": config.get('keep_tokens', 1),
                "flip_aug": config.get('flip_aug', False),
                "caption_extension": config.get('caption_extension', '.txt'),
                "enable_bucket": config.get('enable_bucket', True),
                "bucket_no_upscale": config.get('bucket_no_upscale', False),
                "bucket_reso_steps": config.get('bucket_reso_steps', 64),
                "min_bucket_reso": config.get('min_bucket_reso', 256),
                "max_bucket_reso": config.get('max_bucket_reso', 2048),
                "batch_size": config.get('batch_size', 1),
            },
            "subsets": [
                {
                    "image_dir": self._ensure_absolute_dataset_path(config.get('dataset_dir', '')),
                    "num_repeats": config.get('num_repeats', 10),
                    "caption_extension": config.get('caption_extension', '.txt'),
                }
            ]
        }

        return {"args": derrian_args, "dataset": derrian_dataset}

    def _ensure_absolute_dataset_path(self, dataset_path):
        """
        Convert dataset path to relative path for Derrian backend compatibility.
        
        Derrian's validation expects relative paths from the project root, not absolute paths.
        The sd_scripts run from trainer/derrian_backend/ so they need relative paths.
        
        Example: "/workspace/project/datasets/3_character" -> "datasets/3_character"
        """
        if not dataset_path:
            return dataset_path

        # If it's an absolute path, make it relative to project root
        if os.path.isabs(dataset_path):
            try:
                # Convert absolute path to relative from project root
                dataset_path = os.path.relpath(dataset_path, self.project_root)
            except ValueError:
                # If paths are on different drives (Windows), keep as absolute
                pass

        # Ensure we're not adding extra path separators
        dataset_path = dataset_path.replace('\\', '/')  # Normalize path separators
        
        return dataset_path

    def _get_optimizer_arguments(self, optimizer_name: str) -> Dict[str, Any]:
        """Get optimizer-specific arguments from standard optimizer configurations"""
        if not optimizer_name:
            return {}
        
        # Handle different optimizer name formats
        optimizer_key = optimizer_name
        if optimizer_name == 'LoraEasyCustomOptimizer.came.CAME':
            optimizer_key = 'Came'
        elif optimizer_name == 'AdamW8bit':
            optimizer_key = 'AdamW8bit'
        
        optimizer_config = self.standard_optimizers.get(optimizer_key, {})
        
        if not optimizer_config:
            return {}
        
        # Parse args list into dictionary
        args_dict = {}
        for arg in optimizer_config.get('args', []):
            if '=' in arg:
                key, value = arg.split('=', 1)
                # Convert string values to appropriate types
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                elif value.startswith('[') and value.endswith(']'):
                    # Parse list values like [0.9,0.999]
                    try:
                        value = eval(value)  # Safe for simple lists like [0.9,0.999]
                    except:
                        pass  # Keep as string if parsing fails
                else:
                    try:
                        # Try to convert to float/int
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                
                args_dict[key] = value
        
        return args_dict

    def _get_absolute_model_path(self, model_path: str) -> str:
        """Convert model path to absolute path for Kohya training"""
        if not model_path:
            return ""
        
        # If already absolute, return as-is
        if os.path.isabs(model_path):
            return model_path
        
        # If just a filename, assume it's in pretrained_model directory
        if os.path.sep not in model_path:
            return os.path.join(self.project_root, "pretrained_model", model_path)
        
        # If relative path, convert to absolute using project root
        return os.path.join(self.project_root, model_path)

    def _get_network_configuration(self, lora_type: str, config: Dict) -> Dict[str, Any]:
        """Get network configuration based on LoRA type selection"""
        if not lora_type:
            return {"network_module": "networks.lora"}  # Default to standard LoRA
        
        # Map widget display names to LyCORIS method keys
        lora_type_mapping = {
            'LoRA': None,  # Standard LoRA
            'LoCon': 'locon',
            'LoKR': 'lokr', 
            'DyLoRA': 'dylora',
            'DoRA (Weight Decomposition)': 'dora',  # Special case - DoRA with standard LoRA
            'LoHa (Hadamard Product)': 'loha',
            '(IA)³ (Few Parameters)': 'ia3',
            'BOFT': 'boft',
            'GLoRA': 'glora',
            'Diag-OFT': 'diag_oft',
            'Full': 'full'
        }
        
        mapped_type = lora_type_mapping.get(lora_type)
        
        # Special handling for DoRA - it's an argument, not a separate algorithm
        if mapped_type == 'dora':
            network_args = ["dora_wd=True"]

            # Add additional DoRA parameters from widget config
            if config.get('dora_alpha'):
                network_args.append(f"dora_alpha={config['dora_alpha']}")

            return {
                "network_module": "networks.lora",
                "network_args": network_args
            }
        
        if mapped_type is None:  # Standard LoRA
            return {"network_module": "networks.lora"}
        
        # Special handling for native sd_scripts modules
        if mapped_type == 'dylora':
            network_args = []

            # Add unit parameter (default to 4 if not specified)
            unit = config.get('dylora_unit', 4)
            network_args.append(f"unit={unit}")

            return {
                "network_module": "networks.dylora",
                "network_args": network_args
            }
        
        # Get LyCORIS configuration
        lycoris_config = self.lycoris_methods.get(mapped_type, {})
        
        if lycoris_config:
            result = {
                "network_module": lycoris_config.get('network_module', 'lycoris.kohya')
            }
            
            # Add network_args if specified
            network_args = lycoris_config.get('network_args', []).copy()  # Start with base args (e.g., ['algo=loha'])

            # Add additional parameters from widget config
            if config.get('conv_dim'):
                network_args.append(f"conv_dim={config['conv_dim']}")

            if config.get('conv_alpha'):
                network_args.append(f"conv_alpha={config['conv_alpha']}")

            if config.get('preset'):  # Like "attn-mlp", "full", "attn-only"
                network_args.append(f"preset={config['preset']}")

            if config.get('rank_dropout'):
                network_args.append(f"rank_dropout={config['rank_dropout']}")

            if config.get('module_dropout'):
                network_args.append(f"module_dropout={config['module_dropout']}")

            if config.get('use_tucker') is not None:  # Boolean parameter
                network_args.append(f"use_tucker={str(config['use_tucker']).lower()}")

            if config.get('decompose_both') is not None:  # Boolean parameter
                network_args.append(f"decompose_both={str(config['decompose_both']).lower()}")

            if config.get('factor'):  # For LoKr
                network_args.append(f"factor={config['factor']}")

            if network_args:
                result["network_args"] = network_args
            
            return result
        
        # Fallback to standard LoRA if mapping fails
        return {"network_module": "networks.lora"}
