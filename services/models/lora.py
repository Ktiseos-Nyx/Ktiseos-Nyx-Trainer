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
    """Request to upload files to HuggingFace Hub."""
    token: str = Field(..., description="HuggingFace API token")
    repo_id: str = Field(..., description="HuggingFace repo ID (owner/repo-name)")
    repo_type: str = Field("model", description="Repository type (model/dataset/space)")
    file_paths: list[str] = Field(..., min_length=1, description="List of local file paths to upload")
    remote_folder: str = Field("", description="Remote folder path within the repo")
    commit_message: str = Field(
        "Upload via Ktiseos-Nyx-Trainer",
        description="Git commit message"
    )
    create_pr: bool = Field(False, description="Create a pull request instead of direct commit")


class HuggingFaceUploadResponse(BaseModel):
    """Response from HuggingFace upload."""
    success: bool
    repo_id: str
    uploaded_files: list[str] = Field(default_factory=list)
    failed_files: list[str] = Field(default_factory=list)
    error: Optional[str] = None


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


class CheckpointInput(BaseModel):
    """Single checkpoint input for merging."""
    path: str = Field(..., description="Path to checkpoint file")
    ratio: float = Field(1.0, ge=0.0, le=2.0, description="Merge weight ratio")


class CheckpointMergeRequest(BaseModel):
    """Request to merge multiple checkpoint models."""
    checkpoint_inputs: list[CheckpointInput] = Field(..., min_length=2, description="List of checkpoints to merge")
    output_path: str = Field(..., description="Path to output merged checkpoint")
    unet_only: bool = Field(False, description="Only merge UNet (keep VAE/TE from first model)")
    device: str = Field("cpu", description="Device for processing (cpu/cuda)")
    save_precision: str = Field("fp16", description="Save precision (fp16/bf16/float)")
    precision: str = Field("float", description="Computation precision (float/fp16/bf16)")
    show_skipped: bool = Field(False, description="Show skipped keys in logs")


class CheckpointMergeResponse(BaseModel):
    """Response from checkpoint merge operation."""
    success: bool
    message: str
    output_path: Optional[str] = None
    merged_count: Optional[int] = None
    file_size_mb: Optional[float] = None
