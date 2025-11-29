"""
Dataset service for file and dataset operations.

Handles:
- Dataset creation
- File browsing
- Image listing
- Dataset metadata
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from services.models.dataset import (
    DatasetInfo,
    FileInfo,
    CreateDatasetRequest,
    DatasetListResponse,
    DatasetFilesResponse
)
from services.core.exceptions import ValidationError, NotFoundError
from services.core.validation import (
    validate_dataset_path,
    validate_image_filename,
    DATASETS_DIR,
    ALLOWED_IMAGE_EXTENSIONS
)

logger = logging.getLogger(__name__)


class DatasetService:
    """
    High-level service for dataset management.

    Responsibilities:
    - Create new datasets
    - List available datasets
    - Browse dataset files
    - Get dataset metadata
    - Validate paths
    """

    def __init__(self):
        self.datasets_dir = DATASETS_DIR
        # Ensure datasets directory exists
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    async def list_datasets(self) -> DatasetListResponse:
        """
        List all datasets in the datasets directory.

        Returns:
            DatasetListResponse with dataset info
        """
        datasets = []

        try:
            for entry in self.datasets_dir.iterdir():
                if entry.is_dir():
                    dataset_info = await self._get_dataset_info(entry)
                    datasets.append(dataset_info)

            # Sort by name
            datasets.sort(key=lambda d: d.name)

            return DatasetListResponse(
                datasets=datasets,
                total=len(datasets)
            )

        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return DatasetListResponse(datasets=[], total=0)

    async def get_dataset(self, dataset_name: str) -> DatasetInfo:
        """
        Get information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetInfo

        Raises:
            NotFoundError: If dataset doesn't exist
        """
        dataset_path = validate_dataset_path(dataset_name)

        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {dataset_name}")

        return await self._get_dataset_info(dataset_path)

    async def create_dataset(self, request: CreateDatasetRequest) -> DatasetInfo:
        """
        Create a new dataset directory.

        Args:
            request: Dataset creation request

        Returns:
            DatasetInfo for the new dataset

        Raises:
            ValidationError: If dataset already exists or name is invalid
        """
        # Validate and create path
        dataset_path = validate_dataset_path(request.name)

        if dataset_path.exists():
            raise ValidationError(f"Dataset already exists: {request.name}")

        # Create directory
        dataset_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created dataset: {request.name}")

        return await self._get_dataset_info(dataset_path)

    async def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset and all its contents.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if deleted successfully

        Raises:
            NotFoundError: If dataset doesn't exist
            ValidationError: If trying to delete parent datasets directory
        """
        dataset_path = validate_dataset_path(dataset_name)

        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {dataset_name}")

        # Safety check: don't delete the datasets root
        if dataset_path.resolve() == self.datasets_dir.resolve():
            raise ValidationError("Cannot delete the datasets root directory")

        # Delete directory and contents
        shutil.rmtree(dataset_path)

        logger.info(f"Deleted dataset: {dataset_name}")
        return True

    async def list_files(self, dataset_name: str) -> DatasetFilesResponse:
        """
        List all files in a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetFilesResponse with file listings

        Raises:
            NotFoundError: If dataset doesn't exist
        """
        dataset_path = validate_dataset_path(dataset_name)

        if not dataset_path.exists():
            raise NotFoundError(f"Dataset not found: {dataset_name}")

        files = []
        image_count = 0

        for entry in sorted(dataset_path.iterdir(), key=lambda p: p.name):
            file_info = self._get_file_info(entry)
            files.append(file_info)

            if file_info.is_image:
                image_count += 1

        return DatasetFilesResponse(
            dataset_name=dataset_name,
            files=files,
            total_files=len(files),
            total_images=image_count
        )

    async def _get_dataset_info(self, dataset_path: Path) -> DatasetInfo:
        """Get metadata for a dataset directory."""
        image_count = 0
        caption_count = 0
        total_size = 0

        # Count files
        for entry in dataset_path.iterdir():
            if entry.is_file():
                total_size += entry.stat().st_size

                if entry.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                    image_count += 1
                elif entry.suffix.lower() == '.txt':
                    caption_count += 1

        # Get timestamps
        stat = dataset_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        modified_at = datetime.fromtimestamp(stat.st_mtime)

        return DatasetInfo(
            name=dataset_path.name,
            path=str(dataset_path.relative_to(self.datasets_dir.parent)),
            image_count=image_count,
            caption_count=caption_count,
            total_size=total_size,
            created_at=created_at,
            modified_at=modified_at
        )

    def _get_file_info(self, file_path: Path) -> FileInfo:
        """Get metadata for a single file."""
        stat = file_path.stat()

        is_image = (
            file_path.is_file() and
            file_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )

        return FileInfo(
            name=file_path.name,
            path=str(file_path),
            type="file" if file_path.is_file() else "dir",
            size=stat.st_size if file_path.is_file() else 0,
            modified=stat.st_mtime,
            is_image=is_image,
            mime_type=self._get_mime_type(file_path) if is_image else None
        )

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for an image file."""
        extension_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return extension_map.get(file_path.suffix.lower(), "application/octet-stream")


# Global service instance
dataset_service = DatasetService()
