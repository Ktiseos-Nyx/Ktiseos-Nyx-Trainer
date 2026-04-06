"""
Models for BLIP and GIT captioning.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CaptioningModel(str, Enum):
    """Supported captioning models"""
    BLIP = "blip"
    GIT = "git"


class BLIPConfig(BaseModel):
    """Configuration for BLIP captioning"""
    dataset_dir: str = Field(..., description="Dataset directory path")
    caption_extension: str = Field(default=".txt", description="Caption file extension")
    caption_weights: str = Field(
        default="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth",
        description="BLIP model weights URL"
    )
    batch_size: int = Field(default=1, ge=1, le=32, description="Inference batch size")
    max_workers: int = Field(default=2, ge=1, le=8, description="DataLoader workers")
    beam_search: bool = Field(default=False, description="Use beam search instead of nucleus sampling")
    num_beams: int = Field(default=1, ge=1, description="Number of beams for beam search")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for nucleus sampling")
    max_length: int = Field(default=75, ge=5, le=200, description="Maximum caption length")
    min_length: int = Field(default=5, ge=1, le=50, description="Minimum caption length")
    recursive: bool = Field(default=False, description="Process subdirectories recursively")
    debug: bool = Field(default=False, description="Enable debug mode")


class GITConfig(BaseModel):
    """Configuration for GIT (GenerativeImage2Text) captioning"""
    dataset_dir: str = Field(..., description="Dataset directory path")
    caption_extension: str = Field(default=".txt", description="Caption file extension")
    model_id: str = Field(
        default="microsoft/git-large-textcaps",
        description="HuggingFace model ID for GIT"
    )
    batch_size: int = Field(default=1, ge=1, le=32, description="Inference batch size")
    max_workers: int = Field(default=2, ge=1, le=8, description="DataLoader workers")
    max_length: int = Field(default=50, ge=5, le=200, description="Maximum caption length")
    remove_words: bool = Field(
        default=True,
        description="Remove 'with the words xxx' artifacts from captions"
    )
    recursive: bool = Field(default=False, description="Process subdirectories recursively")
    debug: bool = Field(default=False, description="Enable debug mode")


class CaptioningStartResponse(BaseModel):
    """Response when captioning starts"""
    success: bool
    message: str
    job_id: Optional[str] = None
    validation_errors: list = []


class CaptioningStatusResponse(BaseModel):
    """Captioning job status"""
    job_id: str
    status: str
    progress: int = 0
    current_image: Optional[str] = None
    total_images: Optional[int] = None
    error: Optional[str] = None
