"""
Pydantic models for model and VAE downloads.
"""

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class DownloadSource(str, Enum):
    """Download source type."""
    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"
    OTHER = "other"


class ModelType(str, Enum):
    """Model file type."""
    MODEL = "model"
    VAE = "vae"
    LORA = "lora"


class DownloadMethod(str, Enum):
    """Download method used."""
    ARIA2C = "aria2c"
    WGET = "wget"
    HF_TRANSFER = "hf-transfer"
    REQUESTS = "requests"


class DownloadConfig(BaseModel):
    """Configuration for downloading a model or VAE."""

    # Required
    url: str = Field(..., description="Download URL (HuggingFace or Civitai)")
    download_dir: str = Field(..., description="Target directory for download")

    # Optional
    filename: Optional[str] = Field(
        None,
        description="Explicit filename (required for Civitai, optional for HuggingFace)"
    )
    api_token: Optional[str] = Field(
        None,
        description="API token/key for authenticated downloads"
    )
    model_type: ModelType = Field(
        ModelType.MODEL,
        description="Type of file being downloaded"
    )

    # Metadata (for tracking)
    model_id: Optional[int] = Field(None, description="Civitai model ID")
    version_id: Optional[int] = Field(None, description="Civitai version ID")


class DownloadResponse(BaseModel):
    """Response from download operation."""
    success: bool
    message: str
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    size_mb: Optional[float] = None
    download_method: Optional[DownloadMethod] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about a downloaded model file."""
    name: str = Field(..., description="Filename")
    path: str = Field(..., description="Full file path")
    size_mb: float = Field(..., description="File size in MB")
    type: ModelType = Field(..., description="Model type")


class ListModelsResponse(BaseModel):
    """Response from listing models."""
    success: bool
    models: list[ModelInfo] = Field(default_factory=list)
    vaes: list[ModelInfo] = Field(default_factory=list)
    loras: list[ModelInfo] = Field(default_factory=list)
    model_dir: str
    vae_dir: str
    lora_dir: Optional[str] = None


class DeleteModelResponse(BaseModel):
    """Response from deleting a model."""
    success: bool
    message: str
    file_path: str
