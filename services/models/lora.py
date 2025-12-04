"""
Pydantic models for LoRA utilities (resize, upload).
"""

from typing import Optional
from pydantic import BaseModel, Field


class LoRAResizeRequest(BaseModel):
    """Request to resize a LoRA model."""
    input_path: str = Field(..., description="Path to input LoRA file")
    output_path: str = Field(..., description="Path to output LoRA file")
    target_dim: int = Field(..., ge=1, le=1024, description="Target dimension/rank")
    target_alpha: Optional[int] = Field(None, description="Target alpha (None = auto)")
    device: str = Field("cpu", description="Device for processing (cpu/cuda)")
    save_precision: str = Field("fp16", description="Save precision (fp16/bf16/fp32)")


class LoRAResizeResponse(BaseModel):
    """Response from LoRA resize operation."""
    success: bool
    message: str
    input_path: str
    output_path: str
    original_dim: Optional[int] = None
    new_dim: Optional[int] = None
    file_size_mb: Optional[float] = None


class HuggingFaceUploadRequest(BaseModel):
    """Request to upload LoRA to HuggingFace."""
    lora_path: str = Field(..., description="Path to LoRA file to upload")
    repo_id: str = Field(..., description="HuggingFace repo ID (username/repo-name)")
    token: str = Field(..., description="HuggingFace API token")
    commit_message: Optional[str] = Field(
        "Upload LoRA model",
        description="Git commit message"
    )
    private: bool = Field(False, description="Make repo private")
    create_repo: bool = Field(True, description="Create repo if doesn't exist")
    # Metadata fields
    model_type: Optional[str] = Field(None, description="Model architecture (SDXL, SD1.5, etc.)")
    base_model: Optional[str] = Field(None, description="Base model used for training")
    trigger_word: Optional[str] = Field(None, description="Trigger word for the LoRA")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    description: Optional[str] = Field(None, description="Model description")


class HuggingFaceUploadResponse(BaseModel):
    """Response from HuggingFace upload."""
    success: bool
    message: str
    repo_url: Optional[str] = None
    commit_hash: Optional[str] = None
    errors: list[str] = Field(default_factory=list)


class LoRAInput(BaseModel):
    """Single LoRA input for merging."""
    path: str = Field(..., description="Path to LoRA file")
    ratio: float = Field(1.0, ge=0.0, le=2.0, description="Merge weight ratio")


class LoRAMergeRequest(BaseModel):
    """Request to merge multiple LoRAs."""
    lora_inputs: list[LoRAInput] = Field(..., min_length=2, description="List of LoRAs to merge")
    output_path: str = Field(..., description="Path to output merged LoRA")
    model_type: str = Field("sd", description="Model type: sd (SD1.5), sdxl, flux, svd")
    device: str = Field("cpu", description="Device for processing (cpu/cuda)")
    save_precision: str = Field("fp16", description="Save precision (fp16/bf16/fp32)")
    precision: str = Field("float", description="Computation precision (float/fp16/bf16)")


class LoRAMergeResponse(BaseModel):
    """Response from LoRA merge operation."""
    success: bool
    message: str
    output_path: Optional[str] = None
    merged_count: Optional[int] = None
    file_size_mb: Optional[float] = None
