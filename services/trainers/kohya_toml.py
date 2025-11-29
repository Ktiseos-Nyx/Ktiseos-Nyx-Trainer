"""
TOML configuration file generation for Kohya training scripts.

Extracts working TOML generation logic from old manager without copying the class structure.
"""

import os
import toml
from pathlib import Path
from typing import Dict, Any
import logging

from services.models.training import TrainingConfig, ModelType

logger = logging.getLogger(__name__)


class KohyaTOMLGenerator:
    """
    Generate Kohya-compatible TOML configuration files.

    Handles both dataset.toml and config.toml generation for SD1.5, SDXL, Flux, SD3.
    """

    def __init__(self, config: TrainingConfig, project_root: Path, sd_scripts_dir: Path):
        """
        Initialize TOML generator.

        Args:
            config: Validated training configuration
            project_root: Absolute path to project root
            sd_scripts_dir: Absolute path to Kohya sd_scripts directory
        """
        self.config = config
        self.project_root = project_root
        self.sd_scripts_dir = sd_scripts_dir

    def generate_dataset_toml(self, output_path: Path) -> None:
        """
        Generate dataset.toml for Kohya training.

        Args:
            output_path: Where to write the TOML file
        """
        # Build dataset configuration sections
        datasets_section = {}
        subsets_section = {}
        general_section = {}

        # Dataset-level settings
        if self.config.keep_tokens > 0:
            datasets_section['keep_tokens'] = self.config.keep_tokens

        # Subset configuration
        subsets_section['num_repeats'] = self.config.num_repeats

        # Convert dataset path to relative path from sd_scripts
        dataset_abs_path = Path(self.config.train_data_dir)
        if not dataset_abs_path.is_absolute():
            dataset_abs_path = self.project_root / dataset_abs_path

        subsets_section['image_dir'] = os.path.relpath(dataset_abs_path, self.sd_scripts_dir)

        # General settings
        general_section['resolution'] = self.config.resolution  # Integer
        general_section['shuffle_caption'] = self.config.shuffle_caption
        general_section['flip_aug'] = self.config.flip_aug
        general_section['caption_extension'] = '.txt'

        # Bucketing
        general_section['enable_bucket'] = self.config.enable_bucket
        general_section['bucket_no_upscale'] = self.config.bucket_no_upscale
        general_section['min_bucket_reso'] = self.config.min_bucket_reso
        general_section['max_bucket_reso'] = self.config.max_bucket_reso

        # Bucket resolution steps (SDXL optimization)
        if self.config.sdxl_bucket_optimization:
            general_section['bucket_reso_steps'] = 32
        else:
            general_section['bucket_reso_steps'] = 64

        # Caption handling
        general_section['caption_dropout_rate'] = self.config.caption_dropout_rate
        general_section['caption_tag_dropout_rate'] = self.config.caption_tag_dropout_rate
        if self.config.caption_dropout_every_n_epochs > 0:
            general_section['caption_dropout_every_n_epochs'] = self.config.caption_dropout_every_n_epochs
        if self.config.keep_tokens_separator:
            general_section['keep_tokens_separator'] = self.config.keep_tokens_separator
        if self.config.secondary_separator:
            general_section['secondary_separator'] = self.config.secondary_separator
        general_section['enable_wildcard'] = self.config.enable_wildcard

        # Data augmentation
        general_section['color_aug'] = self.config.color_aug
        general_section['random_crop'] = self.config.random_crop

        # Build final structure
        dataset_config = {
            "datasets": [datasets_section] if datasets_section else [{}],
            "general": general_section
        }

        # Add subsets
        if subsets_section:
            dataset_config["datasets"][0]["subsets"] = [subsets_section]

        # Write to file
        with open(output_path, 'w') as f:
            toml.dump(dataset_config, f)

        logger.info(f"Generated dataset TOML: {output_path}")

    def generate_config_toml(self, output_path: Path) -> None:
        """
        Generate config.toml for Kohya training.

        Args:
            output_path: Where to write the TOML file
        """
        # Get network configuration based on LoRA type
        network_config = self._get_network_config()

        toml_config = {
            "network_arguments": {
                "network_dim": self.config.network_dim,
                "network_alpha": self.config.network_alpha,
                "conv_dim": self.config.conv_dim,
                "conv_alpha": self.config.conv_alpha,
                **network_config
            },
            "optimizer_arguments": {
                "learning_rate": self.config.unet_lr,
                "text_encoder_lr": self.config.text_encoder_lr,
                "lr_scheduler": self.config.lr_scheduler.value,
                "lr_scheduler_num_cycles": self.config.lr_scheduler_number,
                "lr_warmup_ratio": self.config.lr_warmup_ratio,
                "optimizer_type": self.config.optimizer_type.value,
                "max_grad_norm": self.config.max_grad_norm,
            },
            "training_arguments": self._get_training_arguments(),
        }

        # Write to file
        with open(output_path, 'w') as f:
            toml.dump(toml_config, f)

        logger.info(f"Generated config TOML: {output_path}")

    def _get_network_config(self) -> Dict[str, Any]:
        """Get network configuration based on LoRA type."""
        lora_type = self.config.lora_type.value

        if lora_type == "LoRA":
            return {
                "network_module": "networks.lora"
            }
        elif lora_type == "LoCon":
            return {
                "network_module": "lycoris.kohya"
            }
        elif lora_type == "LoHa":
            return {
                "network_module": "lycoris.kohya",
                "network_args": [f"algo=loha"]
            }
        elif lora_type == "LoKR":
            return {
                "network_module": "lycoris.kohya",
                "network_args": [f"algo=lokr"]
            }
        else:
            return {
                "network_module": "networks.lora"
            }

    def _get_training_arguments(self) -> Dict[str, Any]:
        """Build complete training arguments section."""
        args = {
            "pretrained_model_name_or_path": str(Path(self.config.pretrained_model_name_or_path).resolve()),
            "max_train_epochs": self.config.max_train_epochs,
            "train_batch_size": self.config.train_batch_size,
            "save_every_n_epochs": self.config.save_every_n_epochs,
            "mixed_precision": self.config.mixed_precision.value,
            "output_dir": str(Path(self.config.output_dir).resolve()),
            "output_name": self.config.output_name,
            "clip_skip": self.config.clip_skip,
            "save_model_as": self.config.save_model_as.value,
            "seed": self.config.seed,
            # Performance
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "cache_latents": self.config.cache_latents,
            "cache_latents_to_disk": self.config.cache_latents_to_disk,
            "cache_text_encoder_outputs": self.config.cache_text_encoder_outputs,
            "vae_batch_size": self.config.vae_batch_size,
            "no_half_vae": self.config.no_half_vae,
            # Model settings
            "v2": self.config.v2,
            "v_parameterization": self.config.v_parameterization,
            "zero_terminal_snr": self.config.zero_terminal_snr,
            "network_train_unet_only": self.config.network_train_unet_only,
            # Cross attention
            "xformers": self.config.cross_attention.value == "xformers",
            "sdpa": self.config.cross_attention.value == "sdpa",
            "fp8_base": self.config.fp8_base,
            "full_fp16": self.config.full_fp16,
            # Noise settings
            "noise_offset": self.config.noise_offset,
        }

        # Conditional settings
        if self.config.min_snr_gamma_enabled:
            args["min_snr_gamma"] = self.config.min_snr_gamma
        if self.config.ip_noise_gamma_enabled:
            args["ip_noise_gamma"] = self.config.ip_noise_gamma
        if self.config.multinoise:
            args["multires_noise_iterations"] = 6
            args["multires_noise_discount"] = self.config.multires_noise_discount
        if self.config.adaptive_noise_scale > 0:
            args["adaptive_noise_scale"] = self.config.adaptive_noise_scale

        # Save settings
        if self.config.save_last_n_epochs > 0:
            args["save_last_n_epochs"] = self.config.save_last_n_epochs

        return args
