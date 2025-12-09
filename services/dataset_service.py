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

    async def upload_files(self, files, dataset_name: str):
        """
        Upload files to a dataset.

        Args:
            files: List of UploadFile objects from FastAPI
            dataset_name: Name of the dataset to upload to

        Returns:
            dict with success status, uploaded files, and errors
        """
        from fastapi import UploadFile
        from services.core.validation import validate_dataset_path, ALLOWED_IMAGE_EXTENSIONS

        # Validate and get dataset path
        dataset_path = validate_dataset_path(dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        errors = []

        for file in files:
            try:
                # Check file extension
                file_path = Path(file.filename)
                if file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                    errors.append(f"{file.filename}: Invalid file type - only images allowed")
                    continue

                # Save file
                destination = dataset_path / file.filename
                content = await file.read()
                destination.write_bytes(content)

                uploaded_files.append(str(destination))

            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")

        return {
            "uploaded_files": uploaded_files,
            "errors": errors
        }

    async def upload_zip(self, file, dataset_name: str):
        """
        Upload and extract a ZIP file to a dataset.

        Args:
            file: UploadFile object from FastAPI (ZIP file)
            dataset_name: Name of the dataset to extract to

        Returns:
            dict with extracted files, errors, and stats
        """
        import zipfile
        import tempfile
        from services.core.validation import validate_dataset_path, ALLOWED_IMAGE_EXTENSIONS

        # Validate dataset path
        dataset_path = validate_dataset_path(dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        extracted_files = []
        errors = []

        try:
            # Save ZIP to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Extract ZIP
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                # Get list of files
                zip_files = zip_ref.namelist()

                for zip_file in zip_files:
                    # Skip directories and hidden files
                    if zip_file.endswith('/') or Path(zip_file).name.startswith('.'):
                        continue

                    # Check if it's an image
                    file_path = Path(zip_file)
                    if file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                        errors.append(f"{zip_file}: Not an image file, skipped")
                        continue

                    try:
                        # Extract file
                        extracted_data = zip_ref.read(zip_file)

                        # Save to dataset (flatten directory structure)
                        destination = dataset_path / file_path.name

                        # Handle duplicate filenames
                        if destination.exists():
                            base = destination.stem
                            ext = destination.suffix
                            counter = 1
                            while destination.exists():
                                destination = dataset_path / f"{base}_{counter}{ext}"
                                counter += 1

                        destination.write_bytes(extracted_data)
                        extracted_files.append(str(destination))

                    except Exception as e:
                        errors.append(f"{zip_file}: {str(e)}")

            # Clean up temp file
            Path(tmp_path).unlink()

        except zipfile.BadZipFile:
            errors.append("Invalid ZIP file")
        except Exception as e:
            errors.append(f"Failed to extract ZIP: {str(e)}")

        return {
            "extracted_files": extracted_files,
            "errors": errors,
            "total_extracted": len(extracted_files)
        }

    async def download_from_url(self, url: str, dataset_name: str):
        """
        Download dataset from URL (HuggingFace, direct link, etc).

        Args:
            url: URL to download from
            dataset_name: Name of dataset to save to

        Returns:
            dict with downloaded files and status
        """
        import aiohttp
        import tempfile
        from services.core.validation import validate_dataset_path

        # Validate dataset path
        dataset_path = validate_dataset_path(dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3600)) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": f"Failed to download: HTTP {response.status}"
                        }

                    # Get filename from URL or Content-Disposition
                    filename = url.split('/')[-1]
                    content_disp = response.headers.get('Content-Disposition', '')
                    if 'filename=' in content_disp:
                        filename = content_disp.split('filename=')[-1].strip('"\'')

                    # Determine if it's a ZIP
                    is_zip = filename.lower().endswith('.zip') or response.headers.get('Content-Type', '').endswith('zip')

                    # Download file
                    if is_zip:
                        # Download to temp file then extract
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                            async for chunk in response.content.iter_chunked(1024 * 1024):
                                tmp.write(chunk)
                            tmp_path = tmp.name

                        # Create a fake UploadFile object for upload_zip
                        class FakeUploadFile:
                            def __init__(self, path):
                                self.path = path
                            async def read(self):
                                return Path(self.path).read_bytes()

                        fake_file = FakeUploadFile(tmp_path)
                        result = await self.upload_zip(fake_file, dataset_name)

                        # Clean up
                        Path(tmp_path).unlink()

                        return {
                            "success": True,
                            "extracted_files": result["extracted_files"],
                            "errors": result["errors"]
                        }
                    else:
                        # Direct image download
                        destination = dataset_path / filename
                        content = await response.read()
                        destination.write_bytes(content)

                        return {
                            "success": True,
                            "downloaded_files": [str(destination)],
                            "errors": []
                        }

        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"Download failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


# Global service instance
dataset_service = DatasetService()
