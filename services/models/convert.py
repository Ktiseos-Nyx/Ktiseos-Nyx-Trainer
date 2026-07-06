"""
Pydantic models for image format conversion operations.
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class ConvertFormatRequest(BaseModel):
    """Request to convert images in a dataset to a different format."""
    dataset_dir: str = Field(..., description="Dataset directory path or name")
    target_format: Literal["webp", "jpg", "png", "bmp"] = Field(
        ..., description="Target image format"
    )
    quality: int = Field(
        90, ge=1, le=100,
        description="Output quality for JPEG/WebP (1-100). Ignored for PNG/BMP."
    )
    output_mode: Literal["new_dataset", "in-place"] = Field(
        "new_dataset",
        description="Output mode: 'new_dataset' creates a new folder, 'in-place' overwrites originals"
    )


class ConvertFormatResponse(BaseModel):
    """Response from starting a conversion job."""
    success: bool
    job_id: Optional[str] = None
    message: str
    total_files: int = 0
    converted_files: int = 0
    errors: List[str] = Field(default_factory=list)


class ConvertJobStatus(BaseModel):
    """Status of a running or completed conversion job."""
    job_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    total_files: int = 0
    converted_files: int = 0
    current_file: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    result: Optional[dict] = None
