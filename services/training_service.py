"""
Training service for LoRA model training.

Orchestrates training workflow:
- Configuration validation
- Trainer selection
- Job tracking
- Progress monitoring
"""

import logging
from typing import Optional

from services.models.training import TrainingConfig, TrainingStartResponse
from services.models.job import JobType, JobStatus
from services.trainers import KohyaTrainer
from services.jobs import job_manager
from services.core.exceptions import ValidationError, ConfigError, ProcessError

logger = logging.getLogger(__name__)


class TrainingService:
    """
    High-level service for managing LoRA training.

    Responsibilities:
    - Validate training configuration
    - Create appropriate trainer (currently only Kohya)
    - Start training subprocess
    - Track training progress via JobManager
    - Provide status updates
    """

    async def start_training(self, config: TrainingConfig) -> TrainingStartResponse:
        """
        Start a new training job.

        Args:
            config: Validated training configuration

        Returns:
            TrainingStartResponse with job_id or validation errors

        Raises:
            ValidationError: If config validation fails
            ConfigError: If environment setup fails
            ProcessError: If training fails to start

        Example:
            service = TrainingService()
            response = await service.start_training(config)
            if response.success:
                print(f"Training started: {response.job_id}")
        """
        try:
            # Step 1: Create trainer for this config
            trainer = self._create_trainer(config)

            # Step 2: Start training (validates, prepares, generates configs)
            process = await trainer.start_training()

            # Step 3: Register with job manager
            job_id = job_manager.create_job(
                job_type=JobType.TRAINING,
                process=process
            )

            logger.info(f"Training started: {job_id} ({config.model_type} - {config.project_name})")

            return TrainingStartResponse(
                success=True,
                message=f"Training started for {config.project_name}",
                job_id=job_id
            )

        except ValidationError as e:
            logger.warning(f"Training validation failed: {e}")
            return TrainingStartResponse(
                success=False,
                message=str(e),
                validation_errors=[{"field": "config", "message": str(e), "severity": "error"}]
            )

        except (ConfigError, ProcessError) as e:
            logger.error(f"Training startup failed: {e}")
            return TrainingStartResponse(
                success=False,
                message=f"Failed to start training: {e}",
                validation_errors=[{"field": "system", "message": str(e), "severity": "error"}]
            )

        except Exception as e:
            logger.exception(f"Unexpected error starting training: {e}")
            return TrainingStartResponse(
                success=False,
                message=f"Internal error: {e}",
                validation_errors=[{"field": "system", "message": str(e), "severity": "error"}]
            )

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get current training status.

        Args:
            job_id: Training job identifier

        Returns:
            JobStatus or None if not found

        Example:
            status = await service.get_status("job-abc123")
            if status:
                print(f"Progress: {status.progress}%")
        """
        return await job_manager.get_job_status(job_id)

    async def stop_training(self, job_id: str) -> bool:
        """
        Stop a running training job.

        Args:
            job_id: Training job identifier

        Returns:
            True if stopped, False if not found or already stopped

        Example:
            stopped = await service.stop_training("job-abc123")
        """
        return await job_manager.stop_job(job_id)

    def _create_trainer(self, config: TrainingConfig):
        """
        Create appropriate trainer for the configuration.

        Currently only supports Kohya trainer.
        Future: Could support other trainers based on config.

        Args:
            config: Training configuration

        Returns:
            Trainer instance (currently KohyaTrainer)
        """
        # For now, always use Kohya trainer
        # Future: Could select different trainers based on config flags
        # Example: MultiGPUKohyaTrainer, ROCmTrainer, etc.

        return KohyaTrainer(config)


# Global service instance
training_service = TrainingService()
