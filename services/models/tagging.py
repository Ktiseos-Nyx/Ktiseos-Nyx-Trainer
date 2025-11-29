"""
Pydantic models for WD14 image tagging.
"""

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class TaggerModel(str, Enum):
    """Available WD14 tagger models."""
    WD_VIT_LARGE_V3 = "SmilingWolf/wd-vit-large-tagger-v3"
    WD_VIT_TAGGER_V3 = "SmilingWolf/wd-vit-tagger-v3"
    WD_SWINV2_V2 = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
    WD_CONVNEXT_V2 = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    WD_VIT_V2 = "SmilingWolf/wd-v1-4-vit-tagger-v2"


class TaggingConfig(BaseModel):
    """Configuration for WD14 auto-tagging."""

    dataset_dir: str = Field(..., description="Directory containing images to tag")
    model: TaggerModel = Field(
        TaggerModel.WD_VIT_LARGE_V3,
        description="WD14 tagger model to use"
    )
    threshold: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Tag confidence threshold (0-1)"
    )
    blacklist_tags: str = Field(
        "",
        description="Comma-separated tags to exclude"
    )
    caption_extension: str = Field(
        ".txt",
        description="Caption file extension"
    )
    batch_size: int = Field(
        8,
        ge=1,
        le=32,
        description="Batch size for inference"
    )
    max_workers: int = Field(
        2,
        ge=1,
        le=8,
        description="Max data loader workers"
    )
    use_onnx: bool = Field(
        True,
        description="Use ONNX runtime if available (faster)"
    )
    remove_underscore: bool = Field(
        True,
        description="Convert underscores to spaces in tags"
    )
    force_download: bool = Field(
        True,
        description="Force download model to local dir"
    )


class TaggingStartResponse(BaseModel):
    """Response from starting tagging job."""
    success: bool
    message: str
    job_id: Optional[str] = None
    validation_errors: list[dict] = Field(default_factory=list)


class TaggingStatusResponse(BaseModel):
    """Tagging job status."""
    job_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    current_image: Optional[str] = None
    total_images: Optional[int] = None
    error: Optional[str] = None
