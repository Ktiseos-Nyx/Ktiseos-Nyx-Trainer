"""
Kohya-ss training backend implementation.

Implements BaseTrainer for Kohya's sd-scripts training framework.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List
import logging

from services.trainers.base import BaseTrainer
from services.trainers.kohya_toml import KohyaTOMLGenerator
from services.models.training import TrainingConfig, ModelType
from services.core.exceptions import ValidationError, ConfigError

logger = logging.getLogger(__name__)


class KohyaTrainer(BaseTrainer):
    """
    Kohya-ss training backend implementation.

    Uses Kohya's sd-scripts for training LoRA models across:
    - SD 1.5
    - SDXL
    - Flux
    - SD 3.5
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize Kohya trainer.

        Args:
            config: Validated training configuration
        """
        super().__init__(config)

        # Project paths
        self.project_root = Path.cwd()
        self.sd_scripts_dir = self.project_root / "trainer" / "derrian_backend" / "sd_scripts"
        self.config_dir = self.project_root / "trainer" / "runtime_store"

        # TOML generator
        self.toml_generator = KohyaTOMLGenerator(
            config=config,
            project_root=self.project_root,
            sd_scripts_dir=self.sd_scripts_dir
        )

    async def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate Kohya training configuration.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Check base model exists
        model_path = Path(self.config.pretrained_model_name_or_path)
        if not model_path.is_absolute():
            model_path = self.project_root / model_path
        if not model_path.exists():
            errors.append(f"Model not found: {model_path}")

        # Check dataset directory exists
        dataset_path = Path(self.config.train_data_dir)
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path
        if not dataset_path.exists():
            errors.append(f"Dataset not found: {dataset_path}")

        # Check VAE if specified
        if self.config.vae_path:
            vae_path = Path(self.config.vae_path)
            if not vae_path.is_absolute():
                vae_path = self.project_root / vae_path
            if not vae_path.exists():
                errors.append(f"VAE not found: {vae_path}")

        # Flux/SD3 specific validation
        if self.config.model_type in [ModelType.FLUX, ModelType.SD3]:
            if not self.config.clip_l_path:
                errors.append("Flux/SD3 requires clip_l_path")
            if not self.config.t5xxl_path:
                errors.append("Flux/SD3 requires t5xxl_path")

        # Check sd_scripts exists
        if not self.sd_scripts_dir.exists():
            errors.append(f"Kohya sd_scripts not found at: {self.sd_scripts_dir}")

        # Check training script exists
        try:
            script_path = self.get_script_path()
            if not script_path.exists():
                errors.append(f"Training script not found: {script_path}")
        except Exception as e:
            errors.append(f"Cannot determine training script: {e}")

        return (len(errors) == 0, errors)

    async def prepare_environment(self) -> None:
        """
        Prepare Kohya training environment.

        Creates necessary directories and verifies paths.
        """
        # Create output directory
        output_path = Path(self.config.output_dir)
        if not output_path.is_absolute():
            output_path = self.project_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create logging directory if specified
        if self.config.logging_dir:
            log_path = Path(self.config.logging_dir)
            if not log_path.is_absolute():
                log_path = self.project_root / log_path
            log_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Environment prepared for {self.config.model_type} training")

    async def generate_config_files(self) -> Dict[str, Path]:
        """
        Generate Kohya TOML configuration files.

        Returns:
            Dict with 'dataset' and 'config' paths
        """
        # Generate file paths
        dataset_toml = self.config_dir / f"{self.config.output_name}_dataset.toml"
        config_toml = self.config_dir / f"{self.config.output_name}_config.toml"

        # Generate TOML files
        self.toml_generator.generate_dataset_toml(dataset_toml)
        self.toml_generator.generate_config_toml(config_toml)

        logger.info(f"Generated TOML configs: {dataset_toml.name}, {config_toml.name}")

        return {
            "dataset": dataset_toml,
            "config": config_toml
        }

    def build_command(self) -> list[str]:
        """
        Build Kohya training command.

        Returns:
            Command as list of arguments
        """
        script_path = self.get_script_path()
        config_toml = self.config_dir / f"{self.config.output_name}_config.toml"
        dataset_toml = self.config_dir / f"{self.config.output_name}_dataset.toml"

        # Base command
        command = [
            sys.executable,  # Use same Python interpreter
            str(script_path),
            f"--config_file={config_toml}",
            f"--dataset_config={dataset_toml}",
        ]

        logger.info(f"Built command for {self.get_model_type_name()} training")
        return command

    def get_script_path(self) -> Path:
        """
        Get the appropriate training script for the model type.

        Returns:
            Path to training script
        """
        script_map = {
            ModelType.SD15: "train_network.py",
            ModelType.SDXL: "sdxl_train_network.py",
            ModelType.FLUX: "flux_train_network.py",
            ModelType.SD3: "sd3_train_network.py",
            ModelType.LUMINA: "flux_train_network.py",  # Lumina uses same script
        }

        script_name = script_map.get(self.config.model_type, "train_network.py")
        return self.sd_scripts_dir / script_name
