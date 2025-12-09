"""
Pydantic models for dataset operations.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class FileInfo(BaseModel):
    """Information about a file or directory."""
    name: str
    path: str
    type: str = Field(..., description="'file' or 'dir'")
    size: int = Field(0, description="File size in bytes")
    modified: float = Field(..., description="Last modified timestamp")
    is_image: bool = Field(False, description="Whether file is an image")
    mime_type: Optional[str] = None


class DatasetInfo(BaseModel):
    """Dataset directory information."""
    name: str
    path: str
    image_count: int = 0
    caption_count: int = 0
    total_size: int = 0
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None


class CreateDatasetRequest(BaseModel):
    """Request to create a new dataset."""
    name: str = Field(..., min_length=1, max_length=100)
    parent_dir: Optional[str] = Field(
        "datasets",
        description="Parent directory (default: datasets)"
    )


class UploadImageRequest(BaseModel):
    """Metadata for image upload."""
    dataset_name: str
    overwrite: bool = Field(False, description="Overwrite existing files")


class UploadRequest(BaseModel):
    """Request to upload files to a dataset."""
    dataset_name: str
    file_names: List[str]  # filenames to be uploaded
    overwrite: bool = Field(False, description="Overwrite existing files")


class DatasetListResponse(BaseModel):
    """List of datasets."""
    datasets: List[DatasetInfo]
    total: int


class DatasetFilesResponse(BaseModel):
    """Files in a dataset."""
    dataset_name: str
    files: List[FileInfo]
    total_files: int
    total_images: int
