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

        # Build network arguments with LoRA-type-specific config
        network_args = {
            "network_dim": self.config.network_dim,
            "network_alpha": self.config.network_alpha,
            "conv_dim": self.config.conv_dim,
            "conv_alpha": self.config.conv_alpha,
            **network_config
        }

        # Add LoKR factor if applicable
        if self.config.lora_type == "LoKR" and self.config.factor != -1:
            network_args["factor"] = self.config.factor

        toml_config = {
            "network_arguments": network_args,
            "optimizer_arguments": {
                "learning_rate": self.config.unet_lr,
                "text_encoder_lr": self.config.text_encoder_lr,
                "lr_scheduler": self.config.lr_scheduler,  # Removed .value
                "lr_scheduler_num_cycles": self.config.lr_scheduler_number,
                "lr_warmup_ratio": self.config.lr_warmup_ratio,
                "optimizer_type": self.config.optimizer_type,  # Removed .value
                "max_grad_norm": self.config.max_grad_norm,
                "weight_decay": self.config.weight_decay,
            },
            "training_arguments": self._get_training_arguments(),
        }

        # Write to file
        with open(output_path, 'w') as f:
            toml.dump(toml_config, f)

        logger.info(f"Generated config TOML: {output_path}")

    def _get_network_config(self) -> Dict[str, Any]:
        """Get network configuration based on LoRA type."""
        lora_type = self.config.lora_type  # Removed .value

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
        elif lora_type == "LoKR":  # Changed to match uppercase LOKR enum value just in case, but string compare is case sensitive.
            # Wait, my fix to enum made it "LoKr" (mixed case).
            # If config.lora_type is "LoKr", this check fails if it expects "LoKR"?
            # Let's match against the string value.
            return {
                "network_module": "lycoris.kohya",
                "network_args": [f"algo=lokr"]
            }
        elif lora_type == "LoKr": # Handle the new mixed case
             return {
                "network_module": "lycoris.kohya",
                "network_args": [f"algo=lokr"]
            }
        elif lora_type == "DoRA":
             return {
                "network_module": "lycoris.kohya",
                "network_args": [f"algo=dora"]
            }
        else:
            return {
                "network_module": "networks.lora"
            }

    def _get_training_arguments(self) -> Dict[str, Any]:
        """Build complete training arguments section."""
        args = {
            # Basic training
            "pretrained_model_name_or_path": str(Path(self.config.pretrained_model_name_or_path).resolve()),
            "max_train_epochs": self.config.max_train_epochs,
            "train_batch_size": self.config.train_batch_size,
            "output_dir": str(Path(self.config.output_dir).resolve()),
            "output_name": self.config.output_name,
            "seed": self.config.seed,

            # Learning rate scheduler
            "lr_warmup_steps": self.config.lr_warmup_steps,
            "lr_power": self.config.lr_power,

            # Optimizer
            "weight_decay": self.config.weight_decay,

            # Token/Caption
            "max_token_length": self.config.max_token_length,
            "clip_skip": self.config.clip_skip,
            "weighted_captions": self.config.weighted_captions,
            "no_token_padding": self.config.no_token_padding,

            # Save format
            "save_model_as": self.config.save_model_as,  # Removed .value
            "save_precision": self.config.save_precision,
            "no_metadata": self.config.no_metadata,

            # Save frequency
            "save_every_n_epochs": self.config.save_every_n_epochs,
            "save_every_n_steps": self.config.save_every_n_steps,
            "save_last_n_epochs": self.config.save_last_n_epochs,
            "save_last_n_epochs_state": self.config.save_last_n_epochs_state,
            "save_last_n_steps_state": self.config.save_last_n_steps_state,
            "save_state": self.config.save_state,

            # Sample generation
            "sample_every_n_epochs": self.config.sample_every_n_epochs,
            "sample_every_n_steps": self.config.sample_every_n_steps,
            "sample_sampler": self.config.sample_sampler,

            # Performance & Memory
            "mixed_precision": self.config.mixed_precision,  # Removed .value
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "cache_latents": self.config.cache_latents,
            "cache_latents_to_disk": self.config.cache_latents_to_disk,
            "cache_text_encoder_outputs": self.config.cache_text_encoder_outputs,
            "vae_batch_size": self.config.vae_batch_size,
            "no_half_vae": self.config.no_half_vae,
            "persistent_data_loader_workers": self.config.persistent_data_loader_workers,
            "fp8_base": self.config.fp8_base,
            "full_fp16": self.config.full_fp16,
            "lowram": self.config.lowram,

            # Cross attention
            "xformers": self.config.cross_attention == "xformers",  # Removed .value
            "sdpa": self.config.cross_attention == "sdpa",  # Removed .value

            # Model settings
            "v2": self.config.v2,
            "v_parameterization": self.config.v_parameterization,
            "network_train_unet_only": self.config.network_train_unet_only,

            # Noise & SNR
            "noise_offset": self.config.noise_offset,
            "zero_terminal_snr": self.config.zero_terminal_snr,

            # Loss functions
            "prior_loss_weight": self.config.prior_loss_weight,
        }

        # Conditional: VAE path
        if self.config.vae_path:
            args["vae"] = str(Path(self.config.vae_path).resolve())

        # Conditional: Continue from LoRA
        if self.config.continue_from_lora:
            args["network_weights"] = str(Path(self.config.continue_from_lora).resolve())

        # Conditional: Sample prompts
        if self.config.sample_prompts:
            args["sample_prompts"] = str(Path(self.config.sample_prompts).resolve())

        # Conditional: Logging
        if self.config.logging_dir:
            args["logging_dir"] = str(Path(self.config.logging_dir).resolve())
        if self.config.log_with:
            args["log_with"] = self.config.log_with
        if self.config.log_prefix:
            args["log_prefix"] = self.config.log_prefix

        # Conditional: Optimizer args (JSON string)
        if self.config.optimizer_args:
            args["optimizer_args"] = self.config.optimizer_args

        # Conditional: Max train steps (overrides epochs)
        if self.config.max_train_steps > 0:
            args["max_train_steps"] = self.config.max_train_steps

        # Conditional: SNR & Noise
        if self.config.min_snr_gamma_enabled:
            args["min_snr_gamma"] = self.config.min_snr_gamma
        if self.config.ip_noise_gamma_enabled:
            args["ip_noise_gamma"] = self.config.ip_noise_gamma
        if self.config.multinoise:
            args["multires_noise_iterations"] = 6
            args["multires_noise_discount"] = self.config.multires_noise_discount
        if self.config.adaptive_noise_scale > 0:
            args["adaptive_noise_scale"] = self.config.adaptive_noise_scale

        # Conditional: Network dropout
        if self.config.network_dropout > 0:
            args["network_dropout"] = self.config.network_dropout

        # Conditional: LoRA structure advanced
        if self.config.dim_from_weights:
            args["dim_from_weights"] = True
        if self.config.train_norm:
            args["train_norm"] = True
        if self.config.rank_dropout > 0:
            args["rank_dropout"] = self.config.rank_dropout
        if self.config.module_dropout > 0:
            args["module_dropout"] = self.config.module_dropout

        # Conditional: Block-wise LR
        if self.config.down_lr_weight:
            args["down_lr_weight"] = self.config.down_lr_weight
        if self.config.mid_lr_weight:
            args["mid_lr_weight"] = self.config.mid_lr_weight
        if self.config.up_lr_weight:
            args["up_lr_weight"] = self.config.up_lr_weight
        if self.config.block_lr_zero_threshold:
            args["block_lr_zero_threshold"] = self.config.block_lr_zero_threshold
        if self.config.block_dims:
            args["block_dims"] = self.config.block_dims
        if self.config.block_alphas:
            args["block_alphas"] = self.config.block_alphas
        if self.config.conv_block_dims:
            args["conv_block_dims"] = self.config.conv_block_dims
        if self.config.conv_block_alphas:
            args["conv_block_alphas"] = self.config.conv_block_alphas

        # Flux-specific
        if self.config.model_type == ModelType.FLUX:
            if self.config.ae_path:
                args["ae"] = str(Path(self.config.ae_path).resolve())
            if self.config.clip_l_path:
                args["clip_l"] = str(Path(self.config.clip_l_path).resolve())
            if self.config.t5xxl_path:
                args["t5xxl"] = str(Path(self.config.t5xxl_path).resolve())
            if self.config.t5xxl_max_token_length:
                args["t5xxl_max_token_length"] = self.config.t5xxl_max_token_length
            if self.config.apply_t5_attn_mask:
                args["apply_t5_attn_mask"] = True
            if self.config.guidance_scale != 1.0:
                args["guidance_scale"] = self.config.guidance_scale
            args["timestep_sampling"] = self.config.timestep_sampling
            args["sigmoid_scale"] = self.config.sigmoid_scale
            args["model_prediction_type"] = self.config.model_prediction_type
            if self.config.blocks_to_swap:
                args["blocks_to_swap"] = self.config.blocks_to_swap

        # SD3-specific
        if self.config.model_type == ModelType.SD3:
            if self.config.clip_l_path:
                args["clip_l"] = str(Path(self.config.clip_l_path).resolve())
            if self.config.clip_g_path:
                args["clip_g"] = str(Path(self.config.clip_g_path).resolve())
            if self.config.t5xxl_path:
                args["t5xxl"] = str(Path(self.config.t5xxl_path).resolve())

        # Lumina-specific
        if self.config.model_type == ModelType.LUMINA:
            if self.config.gemma2:
                args["gemma2"] = str(Path(self.config.gemma2).resolve())
            if self.config.gemma2_max_token_length:
                args["gemma2_max_token_length"] = self.config.gemma2_max_token_length
            if self.config.ae_path:
                args["ae"] = str(Path(self.config.ae_path).resolve())

        return args