"""
Utilities API Routes
Handles calculator, LoRA utilities, and HuggingFace uploads via new service layer.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import new service
from services import lora_service
from services.models.lora import (
    LoRAResizeRequest as ServiceResizeRequest,
    HuggingFaceUploadRequest as ServiceHFRequest,
    LoRAMergeRequest as ServiceMergeRequest,
    CheckpointMergeRequest as ServiceCheckpointMergeRequest,
    LoRAInput,
    CheckpointInput,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ========== Calculator Endpoints (Kept from original) ==========

class CalculatorRequest(BaseModel):
    """Request model for step calculator"""
    dataset_path: str
    epochs: int = 10
    batch_size: int = 1


class CalculatorResponse(BaseModel):
    """Response model for step calculator"""
    success: bool
    dataset_path: str
    images: int
    repeats: int
    epochs: int
    batch_size: int
    total_steps: int
    caption: Optional[str] = None
    time_estimate_min: float
    time_estimate_max: float
    recommendation: str


@router.post("/calculator", response_model=CalculatorResponse)
async def calculate_steps(request: CalculatorRequest):
    """
    Calculate training steps using Kohya-compatible logic.
    Automatically detects repeat counts from folder names.
    """
    try:
        dataset_path = request.dataset_path.strip()

        if not dataset_path:
            raise HTTPException(status_code=400, detail="Dataset path is required")

        if request.batch_size <= 0:
            raise HTTPException(status_code=400, detail="Batch size must be greater than zero")

        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset path does not exist: {dataset_path}")

        # Extract Kohya parameters from folder name
        folder_name = os.path.basename(dataset_path)
        repeats, caption = extract_kohya_params(folder_name)

        # Count images in dataset
        images = count_images_in_directory(dataset_path)

        if images == 0:
            raise HTTPException(status_code=400, detail=f"No images found in: {dataset_path}")

        # Calculate total steps using Kohya's exact logic
        total_steps = (images * repeats * request.epochs) // request.batch_size

        # Time estimation (approximate)
        time_estimate_min = (total_steps * 2) / 60  # GPU rental (faster)
        time_estimate_max = (total_steps * 4) / 60  # Home GPU

        # Recommendation
        if total_steps < 500:
            recommendation = "⚠️ Low step count - may underfit. Consider more epochs or repeats."
        elif total_steps > 5000:
            recommendation = "⚠️ High step count - may overfit. Consider fewer epochs."
        else:
            recommendation = "✅ Good step count for most LoRA training scenarios."

        return CalculatorResponse(
            success=True,
            dataset_path=dataset_path,
            images=images,
            repeats=repeats,
            epochs=request.epochs,
            batch_size=request.batch_size,
            total_steps=total_steps,
            caption=caption,
            time_estimate_min=time_estimate_min,
            time_estimate_max=time_estimate_max,
            recommendation=recommendation
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Calculator error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/browse")
async def browse_datasets():
    """
    Browse available datasets in the datasets/ directory.
    Returns list with image counts and repeat detection.
    """
    try:
        import glob

        datasets_pattern = "datasets/*"
        existing_dirs = [d for d in glob.glob(datasets_pattern) if os.path.isdir(d)]

        if not existing_dirs:
            return {"datasets": []}

        # Sort by modification time (most recent first)
        existing_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        datasets = []
        for dataset_dir in existing_dirs:
            folder_name = os.path.basename(dataset_dir)
            repeats, caption = extract_kohya_params(folder_name)
            image_count = count_images_in_directory(dataset_dir)

            datasets.append({
                "path": dataset_dir,
                "name": folder_name,
                "image_count": image_count,
                "repeats": repeats,
                "caption": caption
            })

        return {"datasets": datasets}

    except Exception as e:
        logger.error(f"Browse datasets error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== LoRA Utilities (Using New Service) ==========

class LoRAResizeRequest(BaseModel):
    """Request model for LoRA resizing"""
    input_path: str
    output_path: str
    new_dim: int
    new_alpha: Optional[int] = None
    device: str = "cpu"
    save_precision: str = "fp16"


@router.get("/directories")
async def get_directories():
    """Get project directory paths (relative to trainer root)"""
    return {
        "output": "output",
        "datasets": "datasets",
        "pretrained_model": "pretrained_model",
        "vae": "vae",
    }


@router.get("/lora/resize-dimensions")
async def get_resize_dimensions():
    """Get available LoRA resize dimensions"""
    return {
        "dimensions": [4, 8, 16, 32, 64, 128, 256],
        "default": 32
    }


class ListLoraFilesRequest(BaseModel):
    """Request to list LoRA files"""
    directory: str
    extension: str = "safetensors"
    sort_by: str = "name"


@router.post("/lora/list")
async def list_lora_files(request: ListLoraFilesRequest):
    """List LoRA files in a directory"""
    try:
        import glob
        from pathlib import Path

        directory = Path(request.directory)

        # Security: Ensure directory is within project bounds
        project_root = Path.cwd()
        try:
            directory = directory.resolve()
            if not str(directory).startswith(str(project_root)):
                raise HTTPException(status_code=403, detail="Access denied")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid directory path")

        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)

        # Find LoRA files
        pattern = f"*.{request.extension}"
        files = []

        for file_path in directory.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime
                })

        # Sort files
        if request.sort_by == "date":
            files.sort(key=lambda x: x["modified"], reverse=True)
        elif request.sort_by == "size":
            files.sort(key=lambda x: x["size_mb"], reverse=True)
        else:  # name
            files.sort(key=lambda x: x["name"])

        return {
            "success": True,
            "files": files,
            "directory": str(directory),
            "count": len(files)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list LoRA files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lora/resize")
async def resize_lora(request: LoRAResizeRequest):
    """
    Resize a LoRA model to a different dimension.
    Uses Kohya's resize_lora.py script from vendored backend.
    """
    try:
        # Convert to service request
        service_request = ServiceResizeRequest(
            input_path=request.input_path,
            output_path=request.output_path,
            target_dim=request.new_dim,
            target_alpha=request.new_alpha,
            device=request.device,
            save_precision=request.save_precision
        )

        response = await lora_service.resize_lora(service_request)

        return {
            "success": response.success,
            "message": response.message,
            "input_path": response.input_path,
            "output_path": response.output_path,
            "original_dim": response.original_dim,
            "new_dim": response.new_dim,
            "file_size_mb": response.file_size_mb
        }

    except Exception as e:
        logger.error(f"LoRA resize error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class LoRAMergeRequest(BaseModel):
    """Request model for LoRA merging"""
    lora_inputs: list[dict]  # [{path: str, ratio: float}, ...]
    output_path: str
    model_type: str = "sd"
    device: str = "cpu"
    save_precision: str = "fp16"
    precision: str = "float"


@router.post("/lora/merge")
async def merge_lora(request: LoRAMergeRequest):
    """
    Merge multiple LoRA models into one.
    Uses Kohya's merge scripts from vendored backend.
    """
    try:
        # Convert dict inputs to LoRAInput models
        lora_inputs = [
            LoRAInput(path=lora["path"], ratio=lora.get("ratio", 1.0))
            for lora in request.lora_inputs
        ]

        # Convert to service request
        service_request = ServiceMergeRequest(
            lora_inputs=lora_inputs,
            output_path=request.output_path,
            model_type=request.model_type,
            device=request.device,
            save_precision=request.save_precision,
            precision=request.precision
        )

        response = await lora_service.merge_lora(service_request)

        return {
            "success": response.success,
            "message": response.message,
            "output_path": response.output_path,
            "merged_count": response.merged_count,
            "file_size_mb": response.file_size_mb
        }

    except Exception as e:
        logger.error(f"LoRA merge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Checkpoint Merge (Using Service) ==========

class CheckpointMergeRequest(BaseModel):
    """Request model for checkpoint merging"""
    checkpoint_inputs: list[dict]  # [{path: str, ratio: float}, ...]
    output_path: str
    unet_only: bool = False
    device: str = "cpu"
    save_precision: str = "fp16"
    precision: str = "float"
    show_skipped: bool = False


@router.post("/checkpoint/merge")
async def merge_checkpoint(request: CheckpointMergeRequest):
    """
    Merge multiple checkpoint models into one.
    Uses Kohya's merge_models.py script from vendored backend.
    """
    try:
        # Convert dict inputs to CheckpointInput models
        checkpoint_inputs = [
            CheckpointInput(path=cp["path"], ratio=cp.get("ratio", 1.0))
            for cp in request.checkpoint_inputs
        ]

        # Convert to service request
        service_request = ServiceCheckpointMergeRequest(
            checkpoint_inputs=checkpoint_inputs,
            output_path=request.output_path,
            unet_only=request.unet_only,
            device=request.device,
            save_precision=request.save_precision,
            precision=request.precision,
            show_skipped=request.show_skipped
        )

        response = await lora_service.merge_checkpoint(service_request)

        return {
            "success": response.success,
            "message": response.message,
            "output_path": response.output_path,
            "merged_count": response.merged_count,
            "file_size_mb": response.file_size_mb
        }

    except Exception as e:
        logger.error(f"Checkpoint merge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== HuggingFace Upload (Using New Service) ==========

class HuggingFaceUploadRequest(BaseModel):
    """Request model for HuggingFace upload"""
    lora_path: str
    hf_token: str
    repo_id: str
    private: bool = False
    # Metadata
    model_type: Optional[str] = None
    base_model: Optional[str] = None
    trigger_word: Optional[str] = None
    tags: Optional[str] = None
    description: Optional[str] = None


@router.post("/hf/upload")
async def upload_to_huggingface(request: HuggingFaceUploadRequest):
    """
    Upload a LoRA model to HuggingFace Hub.
    Automatically generates README with metadata.
    """
    try:
        # Convert to service request
        service_request = ServiceHFRequest(
            lora_path=request.lora_path,
            repo_id=request.repo_id,
            token=request.hf_token,
            private=request.private,
            model_type=request.model_type,
            base_model=request.base_model,
            trigger_word=request.trigger_word,
            tags=request.tags,
            description=request.description
        )

        response = await lora_service.upload_to_huggingface(service_request)

        return {
            "success": response.success,
            "message": response.message,
            "repo_url": response.repo_url,
            "commit_hash": response.commit_hash,
            "errors": response.errors
        }

    except Exception as e:
        logger.error(f"HuggingFace upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hf/validate-token")
async def validate_hf_token(hf_token: str):
    """Validate HuggingFace API token"""
    try:
        from huggingface_hub import HfApi

        if not hf_token or not hf_token.strip():
            return {"valid": False, "error": "Token is empty"}

        try:
            api = HfApi(token=hf_token)
            user_info = api.whoami()

            return {
                "valid": True,
                "username": user_info.get("name"),
                "type": user_info.get("type")
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    except Exception as e:
        logger.error(f"Token validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Helper Functions ==========

def extract_kohya_params(folder_name: str):
    """Extract repeat count from Kohya-format folder name"""
    tokens = folder_name.split("_")
    try:
        n_repeats = int(tokens[0])
    except ValueError:
        # No repeat count in folder name - default to 1 like Kohya does
        return 1, folder_name
    caption_by_folder = "_".join(tokens[1:])
    return n_repeats, caption_by_folder


def count_images_in_directory(directory):
    """Count image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    image_count = 0
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_count += 1
    return image_count
