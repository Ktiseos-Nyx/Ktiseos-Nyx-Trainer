"""
Abstract base class for training backends.

Defines the interface that all trainers must implement.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

# ✅ FIX C0415: Moved imports to top level
from services.core.exceptions import ProcessError, ValidationError
from services.models.training import TrainingConfig


class BaseTrainer(ABC):
    """
    Abstract base class for LoRA training backends.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: Validated training configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration before training starts.
        """
        # ✅ FIX W0107: Removed unnecessary 'pass' because docstring exists

    @abstractmethod
    async def prepare_environment(self) -> None:
        """
        Prepare training environment (directories, etc).
        """
        # ✅ FIX W0107: Removed unnecessary 'pass'

    @abstractmethod
    async def generate_config_files(self) -> Dict[str, Path]:
        """
        Generate trainer-specific configuration files.
        """

    @abstractmethod
    def build_command(self) -> List[str]:
        """
        Build the training command to execute.
        """

    @abstractmethod
    def get_script_path(self) -> Path:
        """
        Get path to the training script.
        """

    # ✅ FIX E1101: Wrapped return type in quotes to satisfy Pylint
    async def start_training(self) -> "asyncio.subprocess.Process":
        """
        Start training subprocess.
        Orchestrates validation, environment prep, config generation, and execution.
        """
        # Step 1: Validate
        is_valid, errors = await self.validate_config()
        if not is_valid:
            error_str = ", ".join(errors)
            self.logger.error("Validation failed: %s", error_str)
            raise ValidationError(f"Config validation failed: {error_str}")

        # Step 2: Prepare environment
        await self.prepare_environment()

        # Step 3: Generate config files
        await self.generate_config_files()

        # Step 4: Build command
        command = self.build_command()
        self.logger.info("Executing training command: %s", " ".join(command))

        # Step 5: Start subprocess
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(Path.cwd()),
                env=os.environ.copy()
            )

            if process:
                self.logger.info("Subprocess started successfully with PID %s", process.pid)

            return process

        except Exception as e:
            self.logger.error("Failed to start subprocess: %s", str(e), exc_info=True)
            # ✅ FIX W0707: Added 'from e' to link the exception chain
            raise ProcessError(f"Failed to start training script: {e}") from e

    def get_model_type_name(self) -> str:
        """Get human-readable model type name."""
        if hasattr(self.config.model_type, 'value'):
            return str(self.config.model_type.value)
        return str(self.config.model_type)
