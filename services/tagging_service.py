"""
Tagging service for WD14 auto-tagging.

Orchestrates WD14 tagger workflow:
- Configuration validation
- Model availability check
- Job tracking
- Progress monitoring
"""

import sys
import asyncio
import subprocess
import logging
from pathlib import Path
from typing import Optional

from services.models.tagging import (
    TaggingConfig,
    TaggingStartResponse,
    TaggingStatusResponse
)
from services.models.job import JobType, JobStatus
from services.jobs import job_manager
from services.core.exceptions import ValidationError, ProcessError
from services.core.validation import validate_dataset_path

logger = logging.getLogger(__name__)


class TaggingService:
    """
    High-level service for managing WD14 auto-tagging.

    Responsibilities:
    - Validate tagging configuration
    - Verify tagger script exists
    - Start tagging subprocess
    - Track tagging progress via JobManager
    """

    def __init__(self):
        self.project_root = Path.cwd()
        # Try custom tagger first (has v3 model support), fallback to vendored
        self.custom_tagger = self.project_root / "custom" / "tag_images_by_wd14_tagger.py"
        self.vendored_tagger = (
            self.project_root / "trainer" / "derrian_backend" / "sd_scripts" /
            "finetune" / "tag_images_by_wd14_tagger.py"
        )

    async def start_tagging(self, config: TaggingConfig) -> TaggingStartResponse:
        """
        Start a new WD14 tagging job.

        Args:
            config: Validated tagging configuration

        Returns:
            TaggingStartResponse with job_id or validation errors
        """
        try:
            # Step 1: Validate configuration
            await self._validate_config(config)

            # Step 2: Check if ONNX is available
            use_onnx = await self._check_onnx_available() if config.use_onnx else False

            # Step 3: Build command
            command = self._build_command(config, use_onnx)

            # Step 4: Start subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root,
            )

            # Step 5: Register with job manager
            job_id = job_manager.create_job(
                job_type=JobType.TAGGING,
                process=process
            )

            logger.info(f"Tagging started: {job_id} (model: {config.model.value})")

            return TaggingStartResponse(
                success=True,
                message=f"Tagging started for {Path(config.dataset_dir).name}",
                job_id=job_id
            )

        except ValidationError as e:
            logger.warning(f"Tagging validation failed: {e}")
            return TaggingStartResponse(
                success=False,
                message=str(e),
                validation_errors=[{"field": "config", "message": str(e), "severity": "error"}]
            )

        except Exception as e:
            logger.exception(f"Unexpected error starting tagging: {e}")
            return TaggingStartResponse(
                success=False,
                message=f"Internal error: {e}",
                validation_errors=[{"field": "system", "message": str(e), "severity": "error"}]
            )

    async def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current tagging status."""
        return await job_manager.get_job_status(job_id)

    async def stop_tagging(self, job_id: str) -> bool:
        """Stop a running tagging job."""
        return await job_manager.stop_job(job_id)

    async def _validate_config(self, config: TaggingConfig) -> None:
        """
        Validate tagging configuration.

        Raises:
            ValidationError: If validation fails
        """
        # Validate dataset path
        dataset_path = validate_dataset_path(config.dataset_dir)

        if not dataset_path.exists():
            raise ValidationError(f"Dataset directory not found: {config.dataset_dir}")

        # Check for images
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        has_images = any(
            f.suffix.lower() in image_extensions
            for f in dataset_path.iterdir()
            if f.is_file()
        )

        if not has_images:
            raise ValidationError(f"No images found in dataset: {config.dataset_dir}")

        # Check tagger script exists
        tagger_script = self._get_tagger_script()
        if not tagger_script.exists():
            raise ValidationError(
                f"WD14 tagger script not found. "
                f"Please ensure the training backend is installed."
            )

    def _get_tagger_script(self) -> Path:
        """Get path to tagger script (custom or vendored)."""
        if self.custom_tagger.exists():
            logger.debug("Using custom WD14 tagger (with v3 support)")
            return self.custom_tagger
        elif self.vendored_tagger.exists():
            logger.debug("Using vendored WD14 tagger")
            return self.vendored_tagger
        else:
            raise FileNotFoundError("WD14 tagger script not found")

    async def _check_onnx_available(self) -> bool:
        """Check if ONNX runtime is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import onnxruntime; print('OK')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            if stdout.decode().strip() == 'OK':
                logger.info("ONNX runtime available - using accelerated inference")
                return True
            else:
                logger.info("ONNX runtime not available - using PyTorch backend")
                return False

        except Exception:
            logger.info("ONNX runtime check failed - using PyTorch backend")
            return False

    def _build_command(self, config: TaggingConfig, use_onnx: bool) -> list[str]:
        """Build WD14 tagger command."""
        tagger_script = self._get_tagger_script()

        # Resolve dataset path
        dataset_path = validate_dataset_path(config.dataset_dir)

        command = [
            sys.executable,
            str(tagger_script),
            str(dataset_path),
            "--repo_id", config.model.value,
            "--model_dir", "wd14_tagger_model",
            "--thresh", str(config.threshold),
            "--batch_size", str(config.batch_size),
            "--max_data_loader_n_workers", str(config.max_workers),
            "--caption_extension", config.caption_extension,
        ]

        # Optional flags
        if config.force_download:
            command.append("--force_download")
        if config.remove_underscore:
            command.append("--remove_underscore")
        if use_onnx:
            command.append("--onnx")

        # Blacklisted tags
        if config.blacklist_tags:
            command.extend(["--undesired_tags", config.blacklist_tags.replace(" ", "")])

        logger.debug(f"Built tagging command: {' '.join(command[:5])}...")
        return command


# Global service instance
tagging_service = TaggingService()
