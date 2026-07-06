"""
Service for batch image cropping.

Uses the coroutine job pattern (same as conversion/download services).
PIL-based cropping with progress tracking.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Set, List

from PIL import Image

from services.models.crop import CropRegion, CropRequest, CropResponse, CropJobStatus
from services.models.job import JobType
from services.jobs import job_manager
from services.core.exceptions import ValidationError, NotFoundError
from services.core.validation import validate_dataset_path, ALLOWED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# Format-specific save kwargs
FORMAT_SAVE_KWARGS = {
    "jpg": lambda q: {"quality": q, "optimize": True},
    "webp": lambda q: {"quality": q},
    "png": lambda q: {},
}

FORMAT_EXTENSIONS = {
    "jpg": ".jpg",
    "webp": ".webp",
    "png": ".png",
}


class CropService:
    """
    Batch image cropping service.

    Accepts per-image crop regions (source pixel coordinates) and
    performs PIL crop + resize to target dimensions.
    """

    def __init__(self, max_workers: int = 4):
        self._cancelled: Set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def start_crop(self, request: CropRequest) -> CropResponse:
        """Start a batch crop job."""
        try:
            dataset_path = validate_dataset_path(request.dataset_dir)

            if not dataset_path.exists():
                raise NotFoundError(f"Dataset not found: {request.dataset_dir}")

            if not request.crops:
                return CropResponse(
                    success=False,
                    message="No crop regions provided",
                    total_files=0,
                )

            # Determine output directory
            if request.output_mode == "new_dataset":
                suffix = f"{request.target_width}x{request.target_height}"
                output_dir = dataset_path.parent / f"{dataset_path.name}_{suffix}"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = dataset_path

            # Register coroutine job
            job_id = job_manager.run_coroutine_job(
                JobType.CROP,
                lambda job: self._run_crop(
                    job=job,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    crops=request.crops,
                    target_width=request.target_width,
                    target_height=request.target_height,
                    output_format=request.output_format,
                    quality=request.quality,
                    output_mode=request.output_mode,
                ),
            )

            logger.info(f"Started crop job {job_id}: {len(request.crops)} images")

            return CropResponse(
                success=True,
                job_id=job_id,
                message=f"Cropping {len(request.crops)} images to {request.target_width}x{request.target_height}",
                total_files=len(request.crops),
            )

        except (ValidationError, NotFoundError) as e:
            return CropResponse(success=False, message=str(e), total_files=0)

    def _crop_single(
        self,
        crop: CropRegion,
        dataset_path: Path,
        output_dir: Path,
        target_width: int,
        target_height: int,
        target_ext: str,
        save_kwargs: dict,
        output_mode: str,
    ) -> tuple[bool, str, str]:
        """Sync PIL crop+resize+save — runs in thread pool."""
        # Prevent path traversal: strip any directory components from filename
        safe_name = Path(crop.filename).name
        src_file = (dataset_path / safe_name).resolve()
        if not str(src_file).startswith(str(dataset_path.resolve())):
            return False, crop.filename, "access denied"
        if not src_file.exists():
            return False, crop.filename, "file not found"
        if src_file.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            return False, crop.filename, "file type not allowed"

        with Image.open(src_file) as img:
            src_w, src_h = img.size

            sx = max(0, min(crop.source_x, src_w))
            sy = max(0, min(crop.source_y, src_h))
            sw = max(1, min(crop.source_width, src_w - sx))
            sh = max(1, min(crop.source_height, src_h - sy))

            cropped_img = img.crop((sx, sy, sx + sw, sy + sh))

        cropped_img = cropped_img.resize(
            (target_width, target_height), Image.BICUBIC
        )

        out_file = output_dir / (src_file.stem + target_ext)

        if output_mode == "in-place":
            tmp_file = out_file.with_suffix(".tmp_crop")
            cropped_img.save(tmp_file, **save_kwargs)
            tmp_file.replace(out_file)
            if src_file != out_file and src_file.exists():
                src_file.unlink()
        else:
            cropped_img.save(out_file, **save_kwargs)

        return True, crop.filename, f"({sw}x{sh} -> {target_width}x{target_height})"

    async def _run_crop(
        self,
        job,
        dataset_path: Path,
        output_dir: Path,
        crops: List[CropRegion],
        target_width: int,
        target_height: int,
        output_format: str,
        quality: int,
        output_mode: str,
    ) -> dict:
        """Background coroutine that performs the actual cropping — parallelized."""
        errors: List[str] = []
        total = len(crops)
        job.total_images = total
        target_ext = FORMAT_EXTENSIONS[output_format]
        save_kwargs = FORMAT_SAVE_KWARGS.get(output_format, lambda q: {})(quality)

        semaphore = asyncio.Semaphore(4)
        progress_lock = asyncio.Lock()
        done_count = 0
        loop = asyncio.get_running_loop()

        async def process_one(crop: CropRegion) -> bool:
            nonlocal done_count
            async with semaphore:
                if job.job_id in self._cancelled:
                    return False

                try:
                    ok, name, detail = await loop.run_in_executor(
                        self._executor,
                        self._crop_single,
                        crop, dataset_path, output_dir,
                        target_width, target_height,
                        target_ext, save_kwargs, output_mode,
                    )
                    if ok:
                        job.current_image = name
                        job.add_log(f"Cropped {name} {detail}")
                        async with progress_lock:
                            done_count += 1
                            job.progress = int(done_count / total * 100)
                        return True
                    else:
                        errors.append(f"{name}: {detail}")
                        job.add_log(f"SKIP: {name} {detail}")
                        return False
                except Exception as e:
                    errors.append(f"{crop.filename}: {e}")
                    job.add_log(f"ERROR: {crop.filename}: {e}")
                    logger.warning(f"Failed to crop {dataset_path / crop.filename}: {e}")
                    return False

        results = await asyncio.gather(*[process_one(c) for c in crops], return_exceptions=True)

        cropped = sum(1 for r in results if r is True)

        result = {
            "success": True,
            "cropped": cropped,
            "total": total,
            "errors": errors,
            "output_dir": str(output_dir),
            "target_size": f"{target_width}x{target_height}",
        }

        if errors:
            result["warning"] = f"{len(errors)} files failed to crop"

        return result

    async def get_status(self, job_id: str) -> Optional[CropJobStatus]:
        """Get crop job status."""
        job_status = await job_manager.get_job_status(job_id)
        if not job_status:
            return None

        return CropJobStatus(
            job_id=job_status.job_id,
            status=job_status.status.value,
            progress=job_status.progress,
            total_files=job_status.total_images or 0,
            cropped_files=int((job_status.progress / 100) * (job_status.total_images or 0)),
            current_file=job_status.current_image,
            errors=[job_status.error] if job_status.error else [],
            result=job_status.result,
        )

    async def stop_crop(self, job_id: str) -> bool:
        """Cancel a running crop job."""
        self._cancelled.add(job_id)
        return await job_manager.stop_job(job_id)


crop_service = CropService()
