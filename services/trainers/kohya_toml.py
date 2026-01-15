"""
TOML configuration file generation for Kohya training scripts.

Generates FLAT toml files compatible with sd-scripts train_network.py variants.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import toml
import tomlkit  # Ensure tomlkit is installed (pip install tomlkit)

from services.models.training import ModelType, TrainingConfig

logger = logging.getLogger(__name__)


class KohyaTOMLGenerator:
    """
    Generate Kohya-compatible TOML configuration files.

    Produces:
    1. dataset.toml (Images, folders, resolution)
    2. config.toml (Training args + Network args combined)
    """

    def __init__(self, config: TrainingConfig, project_root: Path, sd_scripts_dir: Path):
        self.config = config
        self.project_root = project_root
        self.sd_scripts_dir = sd_scripts_dir

    def validate_paths(self) -> None:
        """
        Validate critical paths exist before TOML generation.
        Raises FileNotFoundError if required paths are missing.
        """
        errors = []

        # Required paths
        model_path = Path(self.config.pretrained_model_name_or_path)
        if not model_path.exists():
            errors.append(f"Base model not found: {model_path}")

        dataset_path = Path(self.config.train_data_dir)
        if not dataset_path.exists():
            errors.append(f"Training dataset directory not found: {dataset_path}")
        elif not any(dataset_path.iterdir()):
            errors.append(f"Training dataset directory is empty: {dataset_path}")

        output_path = Path(self.config.output_dir)
        if not output_path.parent.exists():
            errors.append(f"Output directory parent does not exist: {output_path.parent}")

        # Optional but important paths
        if self.config.vae_path:
            vae_path = Path(self.config.vae_path)
            if not vae_path.exists():
                logger.warning(f"VAE path specified but not found: {vae_path}")

        if self.config.continue_from_lora:
            lora_path = Path(self.config.continue_from_lora)
            if not lora_path.exists():
                errors.append(f"LoRA to continue from not found: {lora_path}")

        if errors:
            error_msg = "Path validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def generate_dataset_toml(self, output_path: Path) -> None:
        """
        Generate dataset.toml.
        This handles the [general] and [[datasets]] formatting required by Kohya.
        """
        # Validate paths before generating TOML
        self.validate_paths()

        doc = tomlkit.document()

        # [general] section
        general = tomlkit.table()
        # Resolution: For SDXL, use list format [width, height] instead of single integer
        # See: https://github.com/kohya-ss/sd-scripts/issues/XXX
        if self.config.model_type in [ModelType.SDXL, ModelType.SD3, ModelType.FLUX, ModelType.LUMINA]:
            # For newer model types, use list format
            resolution_value = [self.config.resolution, self.config.resolution]
        else:
            # For SD 1.5, single integer is acceptable
            resolution_value = self.config.resolution
        general["resolution"] = resolution_value
        general["shuffle_caption"] = self.config.shuffle_caption
        # general["enable_bucket"] = True # Usually implied, but good to ensure
        doc["general"] = general

        # [[datasets]] list
        datasets = tomlkit.aot()
        dataset = tomlkit.table()

        if self.config.keep_tokens > 0:
            dataset["keep_tokens"] = self.config.keep_tokens

        # Resolution: Use same format as general section (list for SDXL/SD3/Flux/Lumina)
        if self.config.model_type in [ModelType.SDXL, ModelType.SD3, ModelType.FLUX, ModelType.LUMINA]:
            dataset["resolution"] = [self.config.resolution, self.config.resolution]
        else:
            dataset["resolution"] = self.config.resolution
        dataset["batch_size"] = self.config.train_batch_size
        dataset["enable_bucket"] = True # Force bucketing enabled
        dataset["min_bucket_reso"] = self.config.min_bucket_reso
        dataset["max_bucket_reso"] = self.config.max_bucket_reso
        if self.config.bucket_reso_steps:
            dataset["bucket_reso_steps"] = self.config.bucket_reso_steps
        if self.config.bucket_no_upscale:
            dataset["bucket_no_upscale"] = True

        # Augmentation settings
        if self.config.flip_aug:
            dataset["flip_aug"] = True
        if self.config.random_crop:
            dataset["random_crop"] = True
        if self.config.color_aug:
            dataset["color_aug"] = True

        # Caption settings
        if self.config.caption_dropout_rate > 0:
            dataset["caption_dropout_rate"] = self.config.caption_dropout_rate
        if self.config.caption_tag_dropout_rate > 0:
            dataset["caption_tag_dropout_rate"] = self.config.caption_tag_dropout_rate
        if self.config.caption_dropout_every_n_epochs > 0:
            dataset["caption_dropout_every_n_epochs"] = self.config.caption_dropout_every_n_epochs
        if self.config.keep_tokens_separator != "|||":
            dataset["keep_tokens_separator"] = self.config.keep_tokens_separator
        if self.config.secondary_separator:
            dataset["secondary_separator"] = self.config.secondary_separator
        if self.config.enable_wildcard:
            dataset["enable_wildcard"] = True

        # [[datasets.subsets]] list
        subsets_aot = tomlkit.aot()
        subset = tomlkit.table()

        dataset_abs_path = Path(self.config.train_data_dir).resolve()
        if not dataset_abs_path.is_absolute():
            dataset_abs_path = (self.project_root / dataset_abs_path).resolve()

        # 🎯 CRITICAL FIX: Use relative path from sd_scripts directory (where training runs)
        # Training scripts run from trainer/derrian_backend/sd_scripts/, so they need
        # relative paths like "../../../datasets/my_dataset" to find data
        # This matches the old working Jupyter implementation
        # Use .as_posix() to convert Windows backslashes to forward slashes for TOML compatibility
        rel_path = os.path.relpath(dataset_abs_path, self.sd_scripts_dir)
        subset["image_dir"] = Path(rel_path).as_posix()
        subset["num_repeats"] = self.config.num_repeats

        # Add metadata path if you use it, otherwise SD-Scripts scans the folder
        # subset["metadata_file"] = ...

        subsets_aot.append(subset)
        dataset["subsets"] = subsets_aot
        datasets.append(dataset)
        doc["datasets"] = datasets

        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

        logger.info("Generated dataset TOML: %s", output_path)

    def generate_config_toml(self, output_path: Path) -> None:
        """
        Generate the MAIN config file.
        Combines Training Arguments AND Network Arguments into one flat TOML.
        """
        # Validate paths before generating TOML
        self.validate_paths()

        # 1. Get Base Training Args
        args = self._get_training_arguments()

        # 2. Add Network Args (LoRA settings) directly into the same dict
        network_config = self._get_network_config()
        args.update(network_config)

        # Add manual network params
        args["network_dim"] = self.config.network_dim
        args["network_alpha"] = self.config.network_alpha

        if self.config.conv_dim > 0:
            args["conv_dim"] = self.config.conv_dim
        if self.config.conv_alpha > 0:
            args["conv_alpha"] = self.config.conv_alpha

        if self.config.network_dropout > 0:
            args["network_dropout"] = self.config.network_dropout
        if self.config.rank_dropout > 0:
            args["rank_dropout"] = self.config.rank_dropout
        if self.config.module_dropout > 0:
            args["module_dropout"] = self.config.module_dropout
        if self.config.train_norm:
            args["train_norm"] = True

        # LoKR specific
        if self.config.lora_type == "LoKR" and self.config.factor != -1:
            args["factor"] = self.config.factor

        # 3. Dump to file
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            toml.dump(args, f)

        logger.info("Generated main config TOML: %s", output_path)

    def _get_network_config(self) -> Dict[str, Any]:
        """
        Helper to determine module string based on LoRA type.

        Maps UI-friendly LoRA types to LyCORIS algorithm names.
        See: https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md
        """
        lora_type = self.config.lora_type

        # Standard LoRA variants
        if lora_type == "LoRA":
            return {"network_module": "networks.lora"}
        elif lora_type == "LoCon":
            return {"network_module": "lycoris.kohya", "network_args": ["algo=locon"]}
        elif lora_type == "LoHa":
            return {"network_module": "lycoris.kohya", "network_args": ["algo=loha"]}
        elif lora_type in ["LoKR", "LoKr"]:
            return {"network_module": "lycoris.kohya", "network_args": ["algo=lokr"]}
        elif lora_type == "DoRA":
            # DoRA = LoRA with weight decomposition
            # See: https://arxiv.org/abs/2402.09353
            return {
                "network_module": "lycoris.kohya",
                "network_args": ["algo=lora", "weight_decompose=True"]
            }

        # Advanced LyCORIS algorithms
        elif lora_type == "Full":
            # Native fine-tuning (DreamBooth)
            return {"network_module": "lycoris.kohya", "network_args": ["algo=full"]}
        elif lora_type == "IA3":
            # (IA)^3 - requires higher learning rates (5e-3 to 1e-2)
            return {"network_module": "lycoris.kohya", "network_args": ["algo=ia3"]}
        elif lora_type == "DyLoRA":
            # Dynamic LoRA - recommend high dim with alpha=dim/4 to dim
            return {"network_module": "lycoris.kohya", "network_args": ["algo=dylora"]}
        elif lora_type == "GLoRA":
            # Generalized LoRA
            return {"network_module": "lycoris.kohya", "network_args": ["algo=glora"]}
        elif lora_type == "Diag-OFT":
            # Diagonal Orthogonal Finetuning
            return {"network_module": "lycoris.kohya", "network_args": ["algo=diag-oft"]}
        elif lora_type == "BOFT":
            # Butterfly OFT
            return {"network_module": "lycoris.kohya", "network_args": ["algo=boft"]}

        # Default fallback
        else:
            return {"network_module": "networks.lora"}

    def _get_training_arguments(self) -> Dict[str, Any]:
        """Map internal TrainingConfig keys to Kohya CLI argument keys"""
        # Use .as_posix() for all paths to avoid Windows backslash escape issues in TOML
        args = {
            "pretrained_model_name_or_path": str(Path(self.config.pretrained_model_name_or_path).resolve().as_posix()),
            "max_train_epochs": self.config.max_train_epochs,
            # "train_batch_size": self.config.train_batch_size, # Often handled in dataset.toml, but safe to keep here too
            "output_dir": str(Path(self.config.output_dir).resolve().as_posix()),
            "output_name": self.config.output_name,
            "seed": self.config.seed,
            "unet_lr": self.config.unet_lr,
            "text_encoder_lr": self.config.text_encoder_lr,
            "lr_scheduler": self.config.lr_scheduler,
            "lr_scheduler_num_cycles": self.config.lr_scheduler_number,
            "lr_warmup_ratio": self.config.lr_warmup_ratio,
            "lr_warmup_steps": self.config.lr_warmup_steps,
            "lr_power": self.config.lr_power,
            "optimizer_type": self.config.optimizer_type,
            "max_grad_norm": self.config.max_grad_norm,
            "weight_decay": self.config.weight_decay,
            "max_token_length": self.config.max_token_length,
            "clip_skip": self.config.clip_skip,
            "weighted_captions": self.config.weighted_captions,
            "no_token_padding": self.config.no_token_padding,
            "save_model_as": self.config.save_model_as,
            "save_precision": self.config.save_precision,
            "no_metadata": self.config.no_metadata,
            "save_every_n_epochs": self.config.save_every_n_epochs,
            "save_every_n_steps": self.config.save_every_n_steps,
            "save_last_n_epochs": self.config.save_last_n_epochs,
            "save_last_n_epochs_state": self.config.save_last_n_epochs_state,
            "save_last_n_steps_state": self.config.save_last_n_steps_state,
            "save_state": self.config.save_state,
            "sample_every_n_epochs": self.config.sample_every_n_epochs,
            "sample_every_n_steps": self.config.sample_every_n_steps,
            "sample_sampler": self.config.sample_sampler,
            "mixed_precision": self.config.mixed_precision,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "cache_latents": self.config.cache_latents,
            "cache_latents_to_disk": self.config.cache_latents_to_disk,
            "cache_text_encoder_outputs": self.config.cache_text_encoder_outputs,
            "vae_batch_size": self.config.vae_batch_size,
            "no_half_vae": self.config.no_half_vae,
            "persistent_data_loader_workers": 1 if self.config.persistent_data_loader_workers else 0,
            "fp8_base": self.config.fp8_base,
            "full_fp16": self.config.full_fp16,
            "lowram": self.config.lowram,
            "xformers": self.config.cross_attention == "xformers",
            "sdpa": self.config.cross_attention == "sdpa",
            "v2": self.config.v2,
            "v_parameterization": self.config.v_parameterization,
            "network_train_unet_only": self.config.network_train_unet_only,
            "noise_offset": self.config.noise_offset,
            "zero_terminal_snr": self.config.zero_terminal_snr,
            "prior_loss_weight": self.config.prior_loss_weight,
        }

        # ========== NEW FIELDS (Issue #97 Fix) ==========
        # Performance/Memory
        if self.config.cache_text_encoder_outputs_to_disk:
            args["cache_text_encoder_outputs_to_disk"] = True
        if self.config.full_bf16:
            args["full_bf16"] = True
        if self.config.max_data_loader_n_workers:
            args["max_data_loader_n_workers"] = self.config.max_data_loader_n_workers

        # Bucketing
        if self.config.bucket_reso_steps:
            args["bucket_reso_steps"] = self.config.bucket_reso_steps

        # Timestep control
        if self.config.min_timestep is not None:
            args["min_timestep"] = self.config.min_timestep
        if self.config.max_timestep is not None:
            args["max_timestep"] = self.config.max_timestep

        # V-prediction loss variants
        if self.config.scale_v_pred_loss_like_noise_pred:
            args["scale_v_pred_loss_like_noise_pred"] = True
        if self.config.v_pred_like_loss is not None:
            args["v_pred_like_loss"] = self.config.v_pred_like_loss
        if self.config.debiased_estimation_loss:
            args["debiased_estimation_loss"] = True

        # State management
        if self.config.save_state_on_train_end:
            args["save_state_on_train_end"] = True
        if self.config.resume_from_state:
            args["resume"] = str(Path(self.config.resume_from_state).resolve().as_posix())

        # Logging
        if self.config.log_tracker_name:
            args["log_tracker_name"] = self.config.log_tracker_name
        if self.config.log_tracker_config:
            args["log_tracker_config"] = self.config.log_tracker_config
        if self.config.wandb_run_name:
            args["wandb_run_name"] = self.config.wandb_run_name

        # ========== END NEW FIELDS ==========

        # Optional Paths (use .as_posix() to avoid Windows backslash issues)
        if self.config.vae_path:
            args["vae"] = str(Path(self.config.vae_path).resolve().as_posix())
        if self.config.continue_from_lora:
            args["network_weights"] = str(Path(self.config.continue_from_lora).resolve().as_posix())
        if self.config.sample_prompts:
            args["sample_prompts"] = str(Path(self.config.sample_prompts).resolve().as_posix())
        if self.config.logging_dir:
            args["logging_dir"] = str(Path(self.config.logging_dir).resolve().as_posix())
        if self.config.log_with:
            args["log_with"] = self.config.log_with
        if self.config.log_prefix:
            args["log_prefix"] = self.config.log_prefix
        if self.config.optimizer_args:
            args["optimizer_args"] = self.config.optimizer_args

        # Steps vs Epochs handling
        if self.config.max_train_steps > 0:
            args["max_train_steps"] = self.config.max_train_steps

        # Noise Settings
        if self.config.min_snr_gamma_enabled:
            args["min_snr_gamma"] = self.config.min_snr_gamma
        if self.config.ip_noise_gamma_enabled:
            args["ip_noise_gamma"] = self.config.ip_noise_gamma
            if self.config.ip_noise_gamma_random_strength:
                args["ip_noise_gamma_random_strength"] = True
        if self.config.multinoise:
            # Use config value if set, otherwise default to 6
            args["multires_noise_iterations"] = self.config.multires_noise_iterations or 6
            args["multires_noise_discount"] = self.config.multires_noise_discount
        if self.config.adaptive_noise_scale > 0:
            args["adaptive_noise_scale"] = self.config.adaptive_noise_scale

        # Block-wise LR
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

        # Flux/SD3/Lumina Specifics (use .as_posix() to avoid Windows backslash issues)
        if self.config.model_type == ModelType.FLUX:
            if self.config.ae_path:
                args["ae"] = str(Path(self.config.ae_path).resolve().as_posix())
            if self.config.clip_l_path:
                args["clip_l"] = str(Path(self.config.clip_l_path).resolve().as_posix())
            if self.config.t5xxl_path:
                args["t5xxl"] = str(Path(self.config.t5xxl_path).resolve().as_posix())
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

        if self.config.model_type == ModelType.SD3:
            if self.config.clip_l_path:
                args["clip_l"] = str(Path(self.config.clip_l_path).resolve().as_posix())
            if self.config.clip_g_path:
                args["clip_g"] = str(Path(self.config.clip_g_path).resolve().as_posix())
            if self.config.t5xxl_path:
                args["t5xxl"] = str(Path(self.config.t5xxl_path).resolve().as_posix())

        if self.config.model_type == ModelType.LUMINA:
            if self.config.gemma2:
                args["gemma2"] = str(Path(self.config.gemma2).resolve().as_posix())
            if self.config.gemma2_max_token_length:
                args["gemma2_max_token_length"] = self.config.gemma2_max_token_length
            if self.config.ae_path:
                args["ae"] = str(Path(self.config.ae_path).resolve().as_posix())

        return args
