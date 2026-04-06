"""
Captioning service for BLIP and GIT natural language captioning.

Orchestrates BLIP/GIT captioning workflow:
- Configuration validation
- Model availability check
- Job tracking
- Progress monitoring
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

from services.models.captioning import (
    BLIPConfig,
    GITConfig,
    CaptioningModel,
    CaptioningStartResponse,
    CaptioningStatusResponse
)
from services.models.job import JobType, JobStatus
from services.jobs import job_manager
from services.core.exceptions import ValidationError
from services.core.validation import validate_dataset_path

logger = logging.getLogger(__name__)


class CaptioningService:
    """
    High-level service for managing BLIP and GIT captioning.

    Responsibilities:
    - Validate captioning configuration
    - Verify captioning scripts exist
    - Start captioning subprocess
    - Track captioning progress via JobManager
    """

    def __init__(self):
        self.project_root = Path.cwd()
        self.blip_script = (
            self.project_root / "trainer" / "derrian_backend" / "sd_scripts" /
            "finetune" / "make_captions.py"
        )
        self.git_script = (
            self.project_root / "trainer" / "derrian_backend" / "sd_scripts" /
            "finetune" / "make_captions_by_git.py"
        )

    async def start_blip_captioning(self, config: BLIPConfig) -> CaptioningStartResponse:
        """
        Start a new BLIP captioning job.

        Args:
            config: Validated BLIP configuration

        Returns:
            CaptioningStartResponse with job_id or validation errors
        """
        try:
            # Step 1: Validate configuration
            await self._validate_config(config.dataset_dir)

            # Step 2: Check script exists
            if not self.blip_script.exists():
                raise ValidationError(
                    f"BLIP captioning script not found at {self.blip_script}"
                )

            # Step 3: Build command
            command = self._build_blip_command(config)

            # Step 4: Start subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root,
            )

            # Step 5: Register with job manager
            job_id = job_manager.create_job(
                job_type=JobType.TAGGING,  # Reuse TAGGING type for now
                process=process
            )

            logger.info(f"BLIP captioning started: {job_id}")

            return CaptioningStartResponse(
                success=True,
                message=f"BLIP captioning started for {Path(config.dataset_dir).name}",
                job_id=job_id
            )

        except ValidationError as e:
            logger.warning(f"BLIP validation failed: {e}")
            return CaptioningStartResponse(
                success=False,
                message=str(e),
                validation_errors=[{"field": "config", "message": str(e), "severity": "error"}]
            )

        except Exception as e:
            logger.exception(f"Unexpected error starting BLIP captioning: {e}")
            return CaptioningStartResponse(
                success=False,
                message=f"Internal error: {e}",
                validation_errors=[{"field": "system", "message": str(e), "severity": "error"}]
            )

    async def start_git_captioning(self, config: GITConfig) -> CaptioningStartResponse:
        """
        Start a new GIT captioning job.

        Args:
            config: Validated GIT configuration

        Returns:
            CaptioningStartResponse with job_id or validation errors
        """
        try:
            # Step 1: Validate configuration
            await self._validate_config(config.dataset_dir)

            # Step 2: Check script exists
            if not self.git_script.exists():
                raise ValidationError(
                    f"GIT captioning script not found at {self.git_script}"
                )

            # Step 3: Build command
            command = self._build_git_command(config)

            # Step 4: Start subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root,
            )

            # Step 5: Register with job manager
            job_id = job_manager.create_job(
                job_type=JobType.TAGGING,  # Reuse TAGGING type for now
                process=process
            )

            logger.info(f"GIT captioning started: {job_id} (model: {config.model_id})")

            return CaptioningStartResponse(
                success=True,
                message=f"GIT captioning started for {Path(config.dataset_dir).name}",
                job_id=job_id
            )

        except ValidationError as e:
            logger.warning(f"GIT validation failed: {e}")
            return CaptioningStartResponse(
                success=False,
                message=str(e),
                validation_errors=[{"field": "config", "message": str(e), "severity": "error"}]
            )

        except Exception as e:
            logger.exception(f"Unexpected error starting GIT captioning: {e}")
            return CaptioningStartResponse(
                success=False,
                message=f"Internal error: {e}",
                validation_errors=[{"field": "system", "message": str(e), "severity": "error"}]
            )

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current captioning status."""
        return await job_manager.get_job_status(job_id)

    async def stop_captioning(self, job_id: str) -> bool:
        """Stop a running captioning job."""
        return await job_manager.stop_job(job_id)

    async def _validate_config(self, dataset_dir: str) -> None:
        """
        Validate captioning configuration.

        Raises:
            ValidationError: If validation fails
        """
        # Validate dataset path
        dataset_path = validate_dataset_path(dataset_dir)

        if not dataset_path.exists():
            raise ValidationError(f"Dataset directory not found: {dataset_dir}")

        # Check for images
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        has_images = any(
            f.suffix.lower() in image_extensions
            for f in dataset_path.iterdir()
            if f.is_file()
        )

        if not has_images:
            raise ValidationError(f"No images found in dataset: {dataset_dir}")

    def _build_blip_command(self, config: BLIPConfig) -> list[str]:
        """Build BLIP captioning command with all parameters."""
        dataset_path = validate_dataset_path(config.dataset_dir)

        command = [
            sys.executable,
            str(self.blip_script),
            str(dataset_path),
            "--caption_extension", config.caption_extension,
            "--caption_weights", config.caption_weights,
            "--batch_size", str(config.batch_size),
            "--top_p", str(config.top_p),
            "--max_length", str(config.max_length),
            "--min_length", str(config.min_length),
        ]

        # Optional workers
        if config.max_workers:
            command.extend(["--max_data_loader_n_workers", str(config.max_workers)])

        # Beam search settings
        if config.beam_search:
            command.append("--beam_search")
            command.extend(["--num_beams", str(config.num_beams)])

        # Boolean flags
        if config.recursive:
            command.append("--recursive")
        if config.debug:
            command.append("--debug")

        logger.debug(f"Built BLIP command: {' '.join(command[:5])}... (+ {len(command)-5} more args)")
        return command

    def _build_git_command(self, config: GITConfig) -> list[str]:
        """Build GIT captioning command with all parameters."""
        dataset_path = validate_dataset_path(config.dataset_dir)

        command = [
            sys.executable,
            str(self.git_script),
            str(dataset_path),
            "--caption_extension", config.caption_extension,
            "--model_id", config.model_id,
            "--batch_size", str(config.batch_size),
            "--max_length", str(config.max_length),
        ]

        # Optional workers
        if config.max_workers:
            command.extend(["--max_data_loader_n_workers", str(config.max_workers)])

        # Boolean flags
        if config.remove_words:
            command.append("--remove_words")
        if config.recursive:
            command.append("--recursive")
        if config.debug:
            command.append("--debug")

        logger.debug(f"Built GIT command: {' '.join(command[:5])}... (+ {len(command)-5} more args)")
        return command


# Global service instance
captioning_service = CaptioningService()
