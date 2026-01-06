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

    # Required
    dataset_dir: str = Field(..., description="Directory containing images to tag")

    # Model settings
    model: TaggerModel = Field(
        TaggerModel.WD_VIT_LARGE_V3,
        description="WD14 tagger model to use"
    )
    force_download: bool = Field(
        False,
        description="Force download model to local dir"
    )

    # Threshold settings (3 separate thresholds!)
    threshold: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Overall tag confidence threshold (0-1)"
    )
    general_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Threshold for general tags (uses threshold if None)"
    )
    character_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Threshold for character tags (uses threshold if None)"
    )

    # Output settings
    caption_extension: str = Field(
        ".txt",
        description="Caption file extension (.txt, .cap, .caption)"
    )
    caption_separator: str = Field(
        ", ",
        description="Separator between tags"
    )

    # Tag filtering and manipulation
    undesired_tags: str = Field(
        "",
        description="Comma-separated tags to exclude (blacklist)"
    )
    tag_replacement: Optional[str] = Field(
        None,
        description="Tag replacement: 'old1,new1;old2,new2'"
    )

    # Tag ordering
    always_first_tags: Optional[str] = Field(
        None,
        description="Comma-separated tags to always put first (e.g., '1girl,solo')"
    )
    character_tags_first: bool = Field(
        False,
        description="Put character tags before general tags"
    )

    # Rating tags
    use_rating_tags: bool = Field(
        False,
        description="Include rating tags in output"
    )
    use_rating_tags_as_last_tag: bool = Field(
        False,
        description="Put rating tags at the end instead of beginning"
    )

    # Tag processing
    remove_underscore: bool = Field(
        True,
        description="Convert underscores to spaces in tags"
    )
    character_tag_expand: bool = Field(
        False,
        description="Expand 'name_(series)' to 'name, series'"
    )

    # File handling
    append_tags: bool = Field(
        False,
        description="Append to existing captions instead of overwriting"
    )
    recursive: bool = Field(
        False,
        description="Process images in subfolders recursively"
    )

    # Performance
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

    # Debug
    frequency_tags: bool = Field(
        False,
        description="Show tag frequency report after tagging"
    )
    debug: bool = Field(
        False,
        description="Enable debug mode"
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
