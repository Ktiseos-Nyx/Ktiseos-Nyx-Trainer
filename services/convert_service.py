"""
Conversion service for image format conversion.

Handles:
- Converting images between formats (JFIF, JPG, PNG, WebP, BMP)
- Job-based progress tracking
- In-place or output-to-new-dataset modes
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional, Set

from PIL import Image

from services.models.convert import (
    ConvertFormatRequest,
    ConvertFormatResponse,
    ConvertJobStatus,
)
from services.models.job import JobType, JobStatusEnum
from services.jobs import job_manager
from services.core.exceptions import ValidationError, NotFoundError
from services.core.validation import (
    validate_dataset_path,
    ALLOWED_IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)

# Format-specific save kwargs
FORMAT_SAVE_KWARGS = {
    "jpg": lambda q: {"quality": q, "optimize": True},
    "jpeg": lambda q: {"quality": q, "optimize": True},
    "webp": lambda q: {"quality": q},
    "png": lambda q: {},
    "bmp": lambda q: {},
}

# Extension mapping for format normalization
FORMAT_EXTENSIONS = {
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "webp": ".webp",
    "png": ".png",
    "bmp": ".bmp",
}


class ConvertService:
    """
    High-level service for image format conversion.

    Uses the coroutine job pattern (like downloads) since conversion
    is pure in-process PIL work with no subprocess needed.
    """

    def __init__(self):
        self._cancelled: Set[str] = set()

    async def start_conversion(
        self, request: ConvertFormatRequest
    ) -> ConvertFormatResponse:
        """
        Start a format conversion job.

        Args:
            request: Conversion request with format, quality, and output mode

        Returns:
            ConvertFormatResponse with job_id for tracking
        """
        try:
            dataset_path = validate_dataset_path(request.dataset_dir)

            if not dataset_path.exists():
                raise NotFoundError(f"Dataset not found: {request.dataset_dir}")

            # Count image files
            image_files = [
                f
                for f in dataset_path.iterdir()
                if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            ]

            if not image_files:
                return ConvertFormatResponse(
                    success=False,
                    message="No image files found in dataset",
                    total_files=0,
                )

            # Determine output directory
            target_ext = FORMAT_EXTENSIONS[request.target_format]
            if request.output_mode == "new_dataset":
                output_dir = dataset_path.parent / f"{dataset_path.name}_{request.target_format}"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = dataset_path

            # Register coroutine job
            job_id = job_manager.run_coroutine_job(
                JobType.CONVERSION,
                lambda job: self._run_conversion(
                    job=job,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    image_files=image_files,
                    target_format=request.target_format,
                    target_ext=target_ext,
                    quality=request.quality,
                    output_mode=request.output_mode,
                ),
            )

            logger.info(
                f"Started conversion job {job_id}: {len(image_files)} files -> .{request.target_format}"
            )

            return ConvertFormatResponse(
                success=True,
                job_id=job_id,
                message=f"Converting {len(image_files)} images to .{request.target_format}",
                total_files=len(image_files),
            )

        except (ValidationError, NotFoundError) as e:
            return ConvertFormatResponse(
                success=False,
                message=str(e),
                total_files=0,
            )

    async def _run_conversion(
        self,
        job,
        dataset_path: Path,
        output_dir: Path,
        image_files: list,
        target_format: str,
        target_ext: str,
        quality: int,
        output_mode: str,
    ) -> dict:
        """
        Background coroutine that performs the actual conversion.

        Updates job.progress and job.current_image as it works.
        Returns a result dict for the job manager.
        """
        converted = 0
        errors = []
        total = len(image_files)
        job.total_images = total
        save_kwargs = FORMAT_SAVE_KWARGS.get(target_format, lambda q: {})(quality)

        for i, src_file in enumerate(image_files):
            # Check cancellation
            if job.job_id in self._cancelled:
                self._cancelled.discard(job.job_id)
                return {"success": False, "error": "Conversion cancelled", "converted": converted}

            try:
                # Determine output filename
                dst_file = output_dir / (src_file.stem + target_ext)

                # Skip if already correct format and in-place mode
                if output_mode == "in-place" and src_file.suffix.lower() == target_ext:
                    continue

                # Open and convert
                # JFIF is just JPEG with a different extension — PIL doesn't
                # always register .jfif for the JPEG decoder, so force format.
                open_kwargs = {"formats": ["JPEG"]} if src_file.suffix.lower() == ".jfif" else {}
                with Image.open(src_file, **open_kwargs) as img:

                    # Handle mode conversion (RGBA -> RGB for JPEG)
                    if img.mode == "RGBA" and target_format in ("jpg", "jpeg"):
                        img = img.convert("RGB")
                    elif img.mode not in ("RGB", "RGBA", "L", "P"):
                        img = img.convert("RGB")

                # Save to temp file first, then replace (atomic-ish)
                if output_mode == "in-place":
                    tmp_file = dst_file.with_suffix(".tmp_convert")
                    img.save(tmp_file, **save_kwargs)
                    tmp_file.replace(dst_file)
                    # Remove original if extension changed
                    if src_file != dst_file and src_file.exists():
                        src_file.unlink()
                else:
                    img.save(dst_file, **save_kwargs)

                converted += 1

                # Update job progress
                job.progress = int((i + 1) / total * 100)
                job.current_image = src_file.name
                job.add_log(f"Converted {src_file.name} -> {dst_file.name}")

            except Exception as e:
                error_msg = f"{src_file.name}: {str(e)}"
                errors.append(error_msg)
                job.add_log(f"ERROR: {error_msg}")
                logger.warning(f"Failed to convert {src_file}: {e}")

        # Copy caption files to output directory (if new_dataset mode)
        if output_mode == "new_dataset" and output_dir != dataset_path:
            for f in dataset_path.iterdir():
                if f.is_file() and f.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                    try:
                        shutil.copy2(f, output_dir / f.name)
                    except Exception as e:
                        logger.warning(f"Failed to copy {f.name}: {e}")

        result = {
            "success": True,
            "converted": converted,
            "total": total,
            "errors": errors,
            "output_dir": str(output_dir),
            "target_format": target_format,
        }

        if errors:
            result["warning"] = f"{len(errors)} files failed to convert"

        return result

    async def get_status(self, job_id: str) -> Optional[ConvertJobStatus]:
        """Get conversion job status."""
        job_status = await job_manager.get_job_status(job_id)
        if not job_status:
            return None

        return ConvertJobStatus(
            job_id=job_status.job_id,
            status=job_status.status.value,
            progress=job_status.progress,
            total_files=job_status.total_images or 0,
            converted_files=int((job_status.progress / 100) * (job_status.total_images or 0)),
            current_file=job_status.current_image,
            logs=job_status.current_step.split("\n") if job_status.current_step else [],
            errors=[job_status.error] if job_status.error else [],
            result=job_status.result,
        )

    async def stop_conversion(self, job_id: str) -> bool:
        """Cancel a running conversion job."""
        self._cancelled.add(job_id)
        return await job_manager.stop_job(job_id)


# Global service instance
convert_service = ConvertService()
