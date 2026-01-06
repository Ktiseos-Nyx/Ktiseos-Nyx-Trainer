"""
Kohya-ss training backend implementation.

Implements BaseTrainer for Kohya's sd-scripts training framework.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from services.core.exceptions import ConfigError, ValidationError
from services.models.training import ModelType, TrainingConfig
from services.trainers.base import BaseTrainer
from services.trainers.kohya_toml import KohyaTOMLGenerator

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
            config=config, project_root=self.project_root, sd_scripts_dir=self.sd_scripts_dir
        )

    async def start_training(self):
        """
        Launch Kohya training using AsyncIO (Required for JobManager compatibility).
        """
        # 1. Prepare Environment
        await self.prepare_environment()

        # 2. Validate Config
        is_valid, errors = await self.validate_config()
        if not is_valid:
            raise ValidationError("; ".join(errors))

        # 3. Generate the TWO Runtime TOMLs
        dataset_toml_runtime = self.config_dir / f"{self.config.project_name}_dataset.toml"
        self.toml_generator.generate_dataset_toml(dataset_toml_runtime)

        config_toml_runtime = self.config_dir / f"{self.config.project_name}_config.toml"
        self.toml_generator.generate_config_toml(config_toml_runtime)

        # 4. Copy to User Config Folder
        user_config_dir = Path("config")
        user_config_dir.mkdir(exist_ok=True)

        dataset_toml_user = user_config_dir / f"{self.config.project_name}_dataset.toml"
        config_toml_user = user_config_dir / f"{self.config.project_name}_config.toml"

        shutil.copy(dataset_toml_runtime, dataset_toml_user)
        shutil.copy(config_toml_runtime, config_toml_user)

        # 5. Build Command List
        cmd = [
            sys.executable,
            str(self.get_script_path()),
            "--config_file",
            str(config_toml_user.resolve()),
            "--dataset_config",
            str(dataset_toml_user.resolve()),
            "--output_dir",
            str(Path(self.config.output_dir).resolve()),
            "--output_name",
            self.config.project_name,
        ]

        logger.info(f"🚀 Launching Kohya Command: {' '.join(cmd)}")

        # 6. Execute Async Process (The Fix)
        try:
            process = await asyncio.create_subprocess_exec(
                cmd[0],  # Program (python)
                *cmd[1:],  # Arguments
                cwd=self.sd_scripts_dir,
                stdout=asyncio.subprocess.PIPE,  # Capture stdout
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout (Critical so errors show in logs)
                env=os.environ.copy(),
            )
            return process  # Now returns an asyncio Process compatible with JobManager

        except Exception as e:
            logger.error(f"Failed to launch subprocess: {e}")
            raise e

    async def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate Kohya training configuration.

        Returns:
            (is_valid, error_messages)
        """
        from services.core.validation import ValidationError as PathValidationError
        from services.core.validation import validate_dataset_path, validate_model_path, validate_output_path

        errors = []

        # Validate dataset path
        try:
            dataset_path = validate_dataset_path(self.config.train_data_dir)
            if not dataset_path.exists():
                errors.append(f"Dataset not found: {dataset_path}")
        except PathValidationError as e:
            errors.append(f"Invalid dataset path: {e}")

        # Validate base model path
        model_path_str = self.config.pretrained_model_name_or_path
        is_hf_model = "/" in model_path_str and not model_path_str.startswith("/")

        if not is_hf_model:
            try:
                model_path = validate_model_path(model_path_str)
                if not model_path.exists():
                    errors.append(f"Model not found: {model_path}")
            except PathValidationError as e:
                errors.append(f"Invalid model path: {e}")

        # Validate VAE path
        if self.config.vae_path:
            try:
                vae_path = validate_model_path(self.config.vae_path)
                if not vae_path.exists():
                    errors.append(f"VAE not found: {vae_path}")
            except PathValidationError as e:
                errors.append(f"Invalid VAE path: {e}")

        # Validate output directory
        try:
            validate_output_path(self.config.output_name)
        except PathValidationError as e:
            errors.append(f"Invalid output path: {e}")

        # Flux/SD3 specific validation
        if self.config.model_type in [ModelType.FLUX, ModelType.SD3]:
            if not self.config.clip_l_path:
                errors.append("Flux/SD3 requires clip_l_path")
            if not self.config.t5xxl_path:
                errors.append("Flux/SD3 requires t5xxl_path")

        # Check sd_scripts exists
        if not self.sd_scripts_dir.exists():
            errors.append(f"Kohya sd_scripts not found at: {self.sd_scripts_dir}")

        return (len(errors) == 0, errors)

    async def prepare_environment(self) -> None:
        """
        Prepare Kohya training environment.
        Creates necessary directories.
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

    def get_script_path(self) -> Path:
        """
        Get the appropriate training script for the model type and training mode.

        Returns:
            Path to training script
        """
        from services.models.training import TrainingMode

        # Select script based on training mode
        if self.config.training_mode == TrainingMode.CHECKPOINT:
            # Full checkpoint/model training
            script_map = {
                ModelType.SD15: "fine_tune.py",
                ModelType.SDXL: "sdxl_train.py",
                ModelType.FLUX: "flux_train.py",
                ModelType.SD3: "sd3_train.py",
                ModelType.LUMINA: "lumina_train.py",
            }
            default_script = "fine_tune.py"
        else:
            # LoRA network training (default)
            script_map = {
                ModelType.SD15: "train_network.py",
                ModelType.SDXL: "sdxl_train_network.py",
                ModelType.FLUX: "flux_train_network.py",
                ModelType.SD3: "sd3_train_network.py",
                ModelType.LUMINA: "flux_train_network.py",
            }
            default_script = "train_network.py"

        script_name = script_map.get(self.config.model_type, default_script)
        return self.sd_scripts_dir / script_name
