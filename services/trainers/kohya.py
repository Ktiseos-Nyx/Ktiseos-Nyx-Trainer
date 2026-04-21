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
    - SD 1.5, SDXL, Flux, SD3, SD3.5, Lumina, Chroma, Anima, HunyuanImage
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize Kohya trainer.

        Args:
            config: Validated training configuration
        """
        super().__init__(config)

        # Project paths — anchor to the source file, not the process CWD.
        # Path.cwd() was wrong on VastAI/RunPod where the service starts from /root or similar.
        self.project_root = Path(__file__).resolve().parents[2]
        self.sd_scripts_dir = self.project_root / "trainer" / "derrian_backend" / "sd_scripts"
        self.config_dir = self.project_root / "trainer" / "runtime_store"

        # TOML generator
        self.toml_generator = KohyaTOMLGenerator(
            config=config, project_root=self.project_root, sd_scripts_dir=self.sd_scripts_dir
        )

    def _resolved_output_dir(self) -> Path:
        """Resolve output_dir anchored to project_root for relative paths."""
        p = Path(self.config.output_dir)
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()

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
        user_config_dir = self.project_root / "config"
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
            str(self._resolved_output_dir()),
            "--output_name",
            self.config.project_name,
        ]

        # Chroma uses Flux scripts but needs --model_type chroma to switch behavior
        if self.config.model_type == ModelType.CHROMA:
            cmd.extend(["--model_type", "chroma"])

        logger.info(f"🚀 Launching Kohya Command: {' '.join(cmd)}")

        # 6. Execute Async Process (The Fix)
        try:
            # Build env with derrian_backend on PYTHONPATH so custom optimizers
            # (CAME, Compass, etc.) are importable from the sd_scripts cwd
            env = os.environ.copy()
            derrian_dir = str(self.sd_scripts_dir.parent)  # trainer/derrian_backend
            custom_sched_dir = str(self.sd_scripts_dir.parent / "custom_scheduler")  # LoraEasyCustomOptimizer
            existing_pythonpath = env.get("PYTHONPATH", "")
            # Include both derrian_backend and custom_scheduler on PYTHONPATH
            # so LoraEasyCustomOptimizer.came.CAME etc. resolve even if editable install failed
            new_paths = f"{derrian_dir}{os.pathsep}{custom_sched_dir}"
            env["PYTHONPATH"] = f"{new_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else new_paths
            # Force unbuffered stdout so logs stream in real-time instead of arriving
            # all at once when the process exits (Python buffers stdout when piped).
            env["PYTHONUNBUFFERED"] = "1"

            process = await asyncio.create_subprocess_exec(
                cmd[0],  # Program (python)
                "-u",    # Unbuffered — belt-and-suspenders with PYTHONUNBUFFERED=1
                *cmd[1:],  # Arguments
                cwd=self.sd_scripts_dir,
                stdout=asyncio.subprocess.PIPE,  # Capture stdout
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout (Critical so errors show in logs)
                env=env,
                limit=1024 * 1024,  # 1MB line buffer — Kohya can emit very long lines (tqdm bars, tensor dumps)
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
        from services.core.validation import validate_dataset_path, validate_model_path

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

        # Validate output directory is within safe bounds (project root or home).
        # validate_output_path() is designed for filenames, not full directory paths,
        # so we do the containment check inline here.
        try:
            out_resolved = self._resolved_output_dir()
            allowed = [self.project_root, Path.home()]
            if not any(out_resolved == a or out_resolved.is_relative_to(a) for a in allowed):
                errors.append(f"Output directory outside allowed paths: {self.config.output_dir}")
        except (ValueError, OSError) as e:
            errors.append(f"Invalid output path: {e}")

        # Flux/SD3/SD3.5/Chroma specific validation
        if self.config.model_type in [ModelType.FLUX, ModelType.SD3, ModelType.SD35, ModelType.CHROMA]:
            # Chroma doesn't need CLIP-L (only T5XXL + AE)
            if self.config.model_type != ModelType.CHROMA and not self.config.clip_l_path:
                errors.append(f"{self.config.model_type} requires clip_l_path")
            if not self.config.t5xxl_path:
                errors.append(f"{self.config.model_type} requires t5xxl_path")

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
        self._resolved_output_dir().mkdir(parents=True, exist_ok=True)

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
            Dict mapping config names to their file paths
        """
        # Generate runtime TOMLs
        dataset_toml_runtime = self.config_dir / f"{self.config.project_name}_dataset.toml"
        self.toml_generator.generate_dataset_toml(dataset_toml_runtime)

        config_toml_runtime = self.config_dir / f"{self.config.project_name}_config.toml"
        self.toml_generator.generate_config_toml(config_toml_runtime)

        # Copy to user config folder
        user_config_dir = self.project_root / "config"
        user_config_dir.mkdir(exist_ok=True)

        dataset_toml_user = user_config_dir / f"{self.config.project_name}_dataset.toml"
        config_toml_user = user_config_dir / f"{self.config.project_name}_config.toml"

        shutil.copy(dataset_toml_runtime, dataset_toml_user)
        shutil.copy(config_toml_runtime, config_toml_user)

        return {
            "dataset_config": dataset_toml_user,
            "training_config": config_toml_user
        }

    def build_command(self) -> List[str]:
        """
        Build the Kohya training command.

        Returns:
            Command as list of strings
        """
        # Get config file paths (they should exist from generate_config_files)
        user_config_dir = self.project_root / "config"
        dataset_toml_user = user_config_dir / f"{self.config.project_name}_dataset.toml"
        config_toml_user = user_config_dir / f"{self.config.project_name}_config.toml"

        cmd = [
            sys.executable,
            str(self.get_script_path()),
            "--config_file",
            str(config_toml_user.resolve()),
            "--dataset_config",
            str(dataset_toml_user.resolve()),
            "--output_dir",
            str(self._resolved_output_dir()),
            "--output_name",
            self.config.project_name,
        ]

        # Chroma uses Flux scripts but needs --model_type chroma to switch behavior
        if self.config.model_type == ModelType.CHROMA:
            cmd.extend(["--model_type", "chroma"])

        return cmd

    def get_script_path(self) -> Path:
        """
        Get the appropriate training script for the model type and training mode.

        Returns:
            Path to training script

        Raises:
            ValueError: If model_type is not set or invalid
        """
        from services.models.training import TrainingMode

        # Validate model_type is set
        if not self.config.model_type:
            raise ValueError("model_type is required but not set in training config")

        # Select script based on training mode
        if self.config.training_mode == TrainingMode.CHECKPOINT:
            # Full checkpoint/model training
            script_map = {
                ModelType.SD15: "fine_tune.py",
                ModelType.SDXL: "sdxl_train.py",
                ModelType.FLUX: "flux_train.py",
                ModelType.SD3: "sd3_train.py",
                ModelType.SD35: "sd3_train.py",  # SD3.5 uses same script as SD3
                ModelType.LUMINA: "lumina_train.py",
                ModelType.CHROMA: "flux_train.py",  # Chroma uses Flux scripts + --model_type chroma
                ModelType.ANIMA: "anima_train.py",
                # HunyuanImage: LoRA only - no checkpoint script
            }
            if self.config.model_type == ModelType.HUNYUAN_IMAGE:
                raise ValueError("HunyuanImage only supports LoRA training, not checkpoint/finetune")
            default_script = "fine_tune.py"
        else:
            # LoRA network training (default)
            script_map = {
                ModelType.SD15: "train_network.py",
                ModelType.SDXL: "sdxl_train_network.py",
                ModelType.FLUX: "flux_train_network.py",
                ModelType.SD3: "sd3_train_network.py",
                ModelType.SD35: "sd3_train_network.py",  # SD3.5 uses same script as SD3
                ModelType.LUMINA: "lumina_train_network.py",
                ModelType.CHROMA: "flux_train_network.py",  # Chroma uses Flux scripts + --model_type chroma
                ModelType.ANIMA: "anima_train_network.py",
                ModelType.HUNYUAN_IMAGE: "hunyuan_image_train_network.py",
            }
            default_script = "train_network.py"

        script_name = script_map.get(self.config.model_type, default_script)
        script_path = self.sd_scripts_dir / script_name

        # Log script selection for debugging
        logger.info(f"Selected training script: {script_name} for model_type={self.config.model_type}, training_mode={self.config.training_mode}")

        # Validate script exists
        if not script_path.exists():
            logger.error(f"Training script not found: {script_path}")
            raise FileNotFoundError(f"Training script not found: {script_path}")

        return script_path
