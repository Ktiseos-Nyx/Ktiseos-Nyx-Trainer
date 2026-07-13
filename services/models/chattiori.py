"""
Chattiori Pydantic models for checkpoint merge and LoRA bake requests/responses.

Wraps Chattiori's merge.py and lora_bake.py CLI arguments.
"""

from typing import Optional
from pydantic import BaseModel, Field


class CheckpointAdvancedMergeRequest(BaseModel):
    """Request model for Chattiori's merge.py advanced checkpoint merging."""

    mode: str = Field(
        ...,
        description="Merging mode: WS, SIG, GEO, MAX, AD, sAD, MD, SIM, TD, TRS, TS, ST, "
        "DARE, ORTHO, SPRSE, NORM, CHAN, FREQ, SWAP, CLIPXOR, XDARE, FWM",
    )
    model_path: str = Field(..., description="Directory containing model files")
    model_0: str = Field(..., description="Filename of model A")
    model_1: str = Field(..., description="Filename of model B")
    model_2: Optional[str] = Field(None, description="Filename of model C (three-model modes)")
    output: str = Field(..., description="Output filename (no extension)")
    alpha: Optional[str] = Field(None, description="Alpha ratio or block-weight string")
    beta: Optional[str] = Field(None, description="Beta ratio")
    device: str = Field("cpu", description="Device to use (cpu/cuda)")
    save_safetensors: bool = Field(True, description="Save as .safetensors")
    save_half: bool = Field(False, description="Save as float16")
    cosine0: bool = Field(False, description="Favor model 0's structure")
    cosine1: bool = Field(False, description="Favor model 1's structure")
    cosine2: bool = Field(False, description="Favor model 2's structure (three models only)")
    vae: Optional[str] = Field(None, description="Path to VAE file")
    prune: bool = Field(False, description="Prune model (remove EMA/unused keys)")
    keep_ema: bool = Field(False, description="Keep EMA weights instead of removing them")
    rebasin: Optional[int] = Field(None, description="ReBasin permutation alignment iterations")
    fine: Optional[str] = Field(None, description="Finetune key pattern")
    seed: Optional[int] = Field(None, description="Random seed for stochastic modes (e.g. DARE)")
    memo: Optional[str] = Field(None, description="Custom metadata note baked into output")


class CheckpointAdvancedMergeResponse(BaseModel):
    """Response model for advanced checkpoint merge result."""

    success: bool = Field(..., description="Whether the merge completed successfully")
    message: str = Field(..., description="Human-readable result or error message")
    output_path: Optional[str] = Field(None, description="Absolute path to the merged output file")
    file_size_mb: Optional[float] = Field(None, description="Output file size in megabytes")
    mode: Optional[str] = Field(None, description="Merge mode used")
    metadata: Optional[dict] = Field(None, description="Additional merge metadata")


class BakeRequest(BaseModel):
    """Request model for Chattiori's lora_bake.py — bake LoRA(s) into a checkpoint."""

    base_model_path: str = Field(..., description="Path to the base checkpoint file")
    lora_paths: list[str] = Field(..., min_length=1, description="Paths to one or more LoRA files")
    lora_ratios: Optional[list[float]] = Field(None, description="Per-LoRA bake ratios (default 1.0 for each)")
    output_path: str = Field(..., description="Output file path (with extension)")
    output_dir: Optional[str] = Field(None, description="Target directory key (output, pretrained_model, comfyui_checkpoints, etc.)")
    text_encoder_path: Optional[str] = Field(None, description="Path to text encoder (legacy Anima mode)")
    device: str = Field("cpu", description="Device to use (cpu/cuda)")
    save_half: bool = Field(False, description="Save as float16")
    save_safetensors: bool = Field(True, description="Save as .safetensors")
    prune: bool = Field(False, description="Prune unused keys from output")
    keep_ema: bool = Field(False, description="Keep EMA weights")
    memo: Optional[str] = Field(None, description="Custom metadata note")
    bake_scale: Optional[float] = Field(None, description="Extra global scale multiplier for baked LoRAs")
    bake_unet_only: bool = Field(False, description="Bake only UNet/DiT modules, skip text encoders")
    bake_clip_scale: Optional[float] = Field(None, description="Scale multiplier for text encoder LoRA modules")


class BakeResponse(BaseModel):
    """Response model for LoRA bake result."""

    success: bool = Field(..., description="Whether the bake completed successfully")
    message: str = Field(..., description="Human-readable result or error message")
    output_path: Optional[str] = Field(None, description="Absolute path to the baked output file")
    file_size_mb: Optional[float] = Field(None, description="Output file size in megabytes")