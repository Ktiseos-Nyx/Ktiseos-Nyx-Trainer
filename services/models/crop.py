"""
Pydantic models for batch image cropping operations.
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class CropRegion(BaseModel):
    """Per-image crop region in source pixel coordinates."""
    filename: str = Field(..., description="Source image filename")
    source_x: float = Field(..., description="Left edge of crop region in source pixels")
    source_y: float = Field(..., description="Top edge of crop region in source pixels")
    source_width: float = Field(..., gt=0, description="Width of crop region in source pixels")
    source_height: float = Field(..., gt=0, description="Height of crop region in source pixels")


class CropRequest(BaseModel):
    """Request to batch-crop images in a dataset."""
    dataset_dir: str = Field(..., description="Dataset directory path or name")
    target_width: int = Field(..., description="Output width in pixels")
    target_height: int = Field(..., description="Output height in pixels")
    output_mode: Literal["new_dataset", "in-place"] = Field(
        "new_dataset",
        description="'new_dataset' creates a new folder, 'in-place' overwrites originals"
    )
    output_format: Literal["webp", "jpg", "png"] = Field(
        "webp", description="Output image format"
    )
    quality: int = Field(
        90, ge=1, le=100,
        description="Output quality for lossy formats (1-100)"
    )
    crops: List[CropRegion] = Field(
        ..., description="Per-image crop regions in source pixel coordinates"
    )


class CropResponse(BaseModel):
    """Response from starting a crop job."""
    success: bool
    job_id: Optional[str] = None
    message: str
    total_files: int = 0


class CropJobStatus(BaseModel):
    """Status of a running or completed crop job."""
    job_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    total_files: int = 0
    cropped_files: int = 0
    current_file: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    result: Optional[dict] = None
