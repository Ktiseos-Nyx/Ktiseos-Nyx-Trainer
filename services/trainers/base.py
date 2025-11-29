"""
Abstract base class for training backends.

Defines the interface that all trainers must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio

from services.models.training import TrainingConfig


class BaseTrainer(ABC):
    """
    Abstract base class for LoRA training backends.

    Subclasses must implement:
    - validate_config(): Pre-flight validation
    - prepare_environment(): Setup directories, verify models
    - generate_config_files(): Create trainer-specific config files
    - build_command(): Construct the training command
    - get_script_path(): Path to training script

    This pattern allows for:
    - Different training backends (Kohya, custom, etc.)
    - Multi-GPU support (future)
    - ROCm/CUDA abstraction (future)
    - Checkpoint training (future)
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Validated training configuration
        """
        self.config = config

    @abstractmethod
    async def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate configuration before training starts.

        Checks:
        - Required files exist (model, VAE, etc.)
        - Model type compatibility
        - Parameter combinations are valid
        - Paths are accessible

        Returns:
            (is_valid, error_messages)
            - is_valid: True if config is valid
            - error_messages: List of validation errors (empty if valid)

        Example:
            is_valid, errors = await trainer.validate_config()
            if not is_valid:
                raise ValidationError(errors)
        """
        pass

    @abstractmethod
    async def prepare_environment(self) -> None:
        """
        Prepare training environment.

        Tasks:
        - Create output directories
        - Verify model files exist
        - Check dataset structure
        - Set up logging directories

        Raises:
            ConfigError: If environment setup fails
        """
        pass

    @abstractmethod
    async def generate_config_files(self) -> Dict[str, Path]:
        """
        Generate trainer-specific configuration files.

        For Kohya: Generates dataset.toml and config.toml
        For others: May generate different formats

        Returns:
            Dict mapping config type to file path
            Example: {"dataset": Path("dataset.toml"), "config": Path("config.toml")}

        Raises:
            ConfigError: If config generation fails
        """
        pass

    @abstractmethod
    def build_command(self) -> list[str]:
        """
        Build the training command to execute.

        Returns:
            List of command arguments ready for subprocess
            Example: ["python", "train_network.py", "--config", "config.toml"]

        Note:
            This is NOT async because it only builds the command,
            doesn't perform I/O.
        """
        pass

    @abstractmethod
    def get_script_path(self) -> Path:
        """
        Get path to the training script.

        Returns:
            Absolute path to training script

        Example:
            For Kohya SDXL: /path/to/sd_scripts/sdxl_train_network.py
            For Kohya SD1.5: /path/to/sd_scripts/train_network.py
        """
        pass

    async def start_training(self) -> asyncio.subprocess.Process:
        """
        Start training subprocess (common implementation).

        This method orchestrates the training startup:
        1. Validate configuration
        2. Prepare environment
        3. Generate config files
        4. Start subprocess

        Returns:
            Running subprocess object for monitoring

        Raises:
            ValidationError: If config validation fails
            ConfigError: If environment setup fails
            ProcessError: If subprocess fails to start

        Note:
            Subclasses typically don't override this - they override
            the individual steps (validate, prepare, etc.)
        """
        from services.core.exceptions import ValidationError, ProcessError

        # Step 1: Validate
        is_valid, errors = await self.validate_config()
        if not is_valid:
            raise ValidationError(f"Config validation failed: {', '.join(errors)}")

        # Step 2: Prepare environment
        await self.prepare_environment()

        # Step 3: Generate config files
        await self.generate_config_files()

        # Step 4: Build command
        command = self.build_command()

        # Step 5: Start subprocess
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=Path.cwd(),
            )
            return process

        except Exception as e:
            raise ProcessError(f"Failed to start training: {e}")

    def get_model_type_name(self) -> str:
        """
        Get human-readable model type name.

        Returns:
            Model type as string (e.g., "SDXL", "SD1.5", "Flux")
        """
        return self.config.model_type.value if hasattr(self.config.model_type, 'value') else str(self.config.model_type)
