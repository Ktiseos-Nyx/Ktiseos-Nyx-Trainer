"""
Utilities API Routes
Handles calculator, LoRA utilities, and HuggingFace uploads via new service layer.
"""

import json
import logging
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.core.validation import (
    validate_path_within, validate_dataset_path, validate_output_path,
    PROJECT_ROOT, DATASETS_DIR, OUTPUT_DIR, MODELS_DIR, VAE_DIR,
)
from services.core.exceptions import ValidationError

# Import new service
from services import lora_service
from services.models.lora import (
    LoRAResizeRequest as ServiceResizeRequest,
    HuggingFaceUploadRequest as ServiceHFRequest,
    LoRAMergeRequest as ServiceMergeRequest,
    LoRAToCheckpointRequest as ServiceLoRAToCheckpointRequest,
    CheckpointMergeRequest as ServiceCheckpointMergeRequest,
    LoRAInput,
    CheckpointInput,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _comfyui_model_dirs() -> dict[str, Path]:
    """
    Resolve ComfyUI's model subdirectories for use in merge/bake file listings
    and input path validation.

    Returns e.g. ``{"comfyui_loras": Path, "comfyui_checkpoints": Path}``.
    If the standard subfolder (loras/, checkpoints/) doesn't exist under the
    ComfyUI models root, the root itself is returned as a fallback so users
    on custom setups still see their files.
    """
    from api.routes.settings import get_comfyui_models_path

    dirs: dict[str, Path] = {}
    base = get_comfyui_models_path()
    if not base:
        return dirs
    base_path = Path(base)
    found = False
    for key, sub in (("comfyui_loras", "loras"), ("comfyui_checkpoints", "checkpoints")):
        candidate = base_path / sub
        if candidate.is_dir():
            dirs[key] = candidate
            found = True
    if not found and base_path.is_dir():
        dirs["comfyui_checkpoints"] = base_path
        dirs["comfyui_loras"] = base_path
    return dirs


def _model_input_dirs() -> list[Path]:
    """
    Allowed source directories for merge/resize *inputs*: the trainer's
    ``output/`` and ``pretrained_model/`` plus any present ComfyUI model dirs.
    Merge *outputs* stay confined to ``OUTPUT_DIR`` — we never write into a
    model source directory.
    """
    return [OUTPUT_DIR, MODELS_DIR, *_comfyui_model_dirs().values()]


def _validate_model_input(user_path: str) -> Path:
    """
    Validate a merge/resize *input* path against the allowed model dirs, tolerating
    symlinks that physically live inside a model dir.

    Why: LoRA Manager downloads checkpoints into symlinked subfolders that point at
    its own library, outside ``ComfyUI/models/checkpoints``. ``validate_path_within``
    calls ``.resolve()`` (follows the symlink) and would reject the off-tree target,
    so those models were unselectable. Here we check the path's *physical* location
    (``abspath`` collapses ``..`` but does NOT follow symlinks), so a model the user
    placed under a model dir is accepted regardless of where the symlink points.

    Security: this is a single-user, local tool — the only symlinks here are ones the
    user created in their own model library, so trusting them is acceptable. ``..``
    traversal is still blocked (abspath normalises it). Falls back to the strict
    resolved check for non-symlinked absolute paths. Output paths are NOT validated
    with this — they stay strict via ``validate_path_within``.
    """
    phys = Path(os.path.abspath(user_path))
    for d in _model_input_dirs():
        base = Path(os.path.abspath(str(d)))
        if phys == base or phys.is_relative_to(base):
            return phys
    return validate_path_within(user_path, _model_input_dirs())


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

        # Security: confine to datasets directory
        try:
            validate_path_within(dataset_path, [DATASETS_DIR])
        except ValidationError:
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

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
    device: Literal["cpu", "cuda"] = "cpu"
    save_precision: Literal["float", "fp16", "bf16"] = "fp16"


@router.get("/directories")
async def get_directories():
    """Get project directory paths for use with the file listing API.

    Includes ComfyUI loras/checkpoints dirs when present so the merge tools
    can list models from the ComfyUI ecosystem, not just the trainer dirs.
    """
    dirs = {
        "output": str(OUTPUT_DIR),
        "datasets": str(DATASETS_DIR),
        "pretrained_model": str(MODELS_DIR),
        "vae": str(VAE_DIR),
    }
    for key, path in _comfyui_model_dirs().items():
        dirs[key] = str(path)
    return dirs


@router.get("/block-weight-presets")
async def get_block_weight_presets():
    """
    Return named block-weight (LBW) presets for the merge tools, keyed by
    architecture: 'sd' presets are 26 values, 'sdxl' presets are 20 values.
    """
    presets_file = PROJECT_ROOT / "presets" / "block_weights.json"
    try:
        with open(presets_file, encoding="utf-8") as f:
            data = json.load(f)
        return {"sd": data.get("sd", {}), "sdxl": data.get("sdxl", {})}
    except (FileNotFoundError, ValueError):
        return {"sd": {}, "sdxl": {}}


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
    file_extension: str = "safetensors"
    sort_by: str = "name"


@router.post("/lora/list")
async def list_lora_files(request: ListLoraFilesRequest):
    """List LoRA files in a directory"""
    try:
        import glob
        from pathlib import Path

        directory = Path(request.directory)

        # Security: confine to trainer output/pretrained_model or ComfyUI model dirs
        try:
            directory = _validate_model_input(str(directory))
        except ValidationError:
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)

        # Walk the whole tree, following symlinked subdirs (LoRA Manager symlinks its
        # download folders in — plain rglob skips them). Supports comma-separated
        # extensions like "safetensors,ckpt".
        suffixes = tuple(f".{ext.strip().lower()}" for ext in request.file_extension.split(","))
        files = []
        seen = set()

        for root, _dirs, filenames in os.walk(directory, followlinks=True):
            for fn in filenames:
                if not fn.lower().endswith(suffixes):
                    continue
                file_path = Path(root) / fn
                key = str(file_path)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    stat = file_path.stat()  # follows symlink; skips broken links
                except OSError:
                    continue
                size_mb = round(stat.st_size / (1024 * 1024), 2)
                files.append({
                    "name": file_path.name,
                    "path": key,
                    "size_mb": size_mb,
                    "size_formatted": f"{size_mb:.1f} MB",
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
        # Security: input from any model dir; output always to output/
        try:
            _validate_model_input(request.input_path)
            validate_path_within(request.output_path, [OUTPUT_DIR])
        except ValidationError:
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

        # Convert to service request
        service_request = ServiceResizeRequest(
            input_path=request.input_path,
            output_path=request.output_path,
            target_dim=request.new_dim,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LoRA resize error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class LoRAMergeRequest(BaseModel):
    """Request model for LoRA merging"""
    lora_inputs: list[dict]  # [{path: str, ratio: float}, ...]
    output_path: str
    model_type: Literal["sd", "sdxl", "flux", "svd"] = "sd"
    device: Literal["cpu", "cuda"] = "cpu"
    save_precision: Literal["float", "fp16", "bf16"] = "fp16"
    precision: Literal["float", "fp16", "bf16"] = "float"


@router.post("/lora/merge")
async def merge_lora(request: LoRAMergeRequest):
    """
    Merge multiple LoRA models into one.
    Uses Kohya's merge scripts from vendored backend.
    """
    try:
        # Security: inputs from any model dir; output always to output/
        try:
            for lora in request.lora_inputs:
                _validate_model_input(lora["path"])
            validate_path_within(request.output_path, [OUTPUT_DIR])
        except ValidationError:
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

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
            precision=request.precision,
        )

        response = await lora_service.merge_lora(service_request)

        return {
            "success": response.success,
            "message": response.message,
            "output_path": response.output_path,
            "merged_count": response.merged_count,
            "file_size_mb": response.file_size_mb
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LoRA merge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class LoRAToCheckpointRequest(BaseModel):
    """Request model for baking LoRA(s) into a base checkpoint."""
    base_model_path: str
    text_encoder_path: Optional[str] = None  # Required for Anima
    lora_inputs: list[dict]  # [{path: str, ratio: float}, ...]
    output_path: str
    model_type: Literal["sd", "sdxl", "anima"] = "sdxl"
    device: Literal["cpu", "cuda"] = "cpu"
    save_precision: Literal["float", "fp16", "bf16"] = "fp16"
    precision: Literal["float", "fp16", "bf16"] = "float"


@router.post("/lora/merge-to-checkpoint")
async def merge_lora_to_checkpoint(request: LoRAToCheckpointRequest):
    """
    Bake one or more LoRAs into a base SD/SDXL checkpoint, producing a full
    standalone checkpoint (the "LoRA -> checkpoint" merge).
    Uses Kohya's merge scripts (--sd_model) from the vendored backend.
    """
    try:
        # Security: base + LoRA inputs from any model dir; output always to output/
        try:
            _validate_model_input(request.base_model_path)
            for lora in request.lora_inputs:
                _validate_model_input(lora["path"])
            validate_path_within(request.output_path, [OUTPUT_DIR])
        except ValidationError:
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

        lora_inputs = [
            LoRAInput(path=lora["path"], ratio=lora.get("ratio", 1.0))
            for lora in request.lora_inputs
        ]

        service_request = ServiceLoRAToCheckpointRequest(
            base_model_path=request.base_model_path,
            text_encoder_path=request.text_encoder_path,
            lora_inputs=lora_inputs,
            output_path=request.output_path,
            model_type=request.model_type,
            device=request.device,
            save_precision=request.save_precision,
            precision=request.precision,
        )

        response = await lora_service.merge_lora_to_checkpoint(service_request)

        return {
            "success": response.success,
            "message": response.message,
            "output_path": response.output_path,
            "merged_count": response.merged_count,
            "file_size_mb": response.file_size_mb,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LoRA-to-checkpoint merge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Block-Weighted Checkpoint Merge ==========

class BlockWeightedMergeRequest(BaseModel):
    """Request model for block-weighted (MBW) checkpoint merge."""
    model_a_path: str
    model_b_path: str
    model_c_path: Optional[str] = None  # Required for Add/Triple/Twice
    output_path: str
    mode: Literal["weight", "add", "triple", "twice"] = "weight"
    block_weights: list[float]  # 26 values for SD1.5, 20 for SDXL
    block_weights_c: Optional[list[float]] = None  # Second curve for Triple/Twice
    base_alpha: float = 0.5
    base_alpha_c: Optional[float] = None  # Second base_alpha for Twice
    device: Literal["cpu", "cuda"] = "cpu"


class DetectArchRequest(BaseModel):
    """Request to detect checkpoint architecture."""
    checkpoint_path: str


@router.post("/checkpoint/detect-arch")
async def detect_checkpoint_arch(request: DetectArchRequest):
    """Detect architecture of a safetensors checkpoint and return block info."""
    try:
        from services.block_weight_merge import load_checkpoint, detect_architecture, arch_block_count, arch_block_names

        _validate_model_input(request.checkpoint_path)
        sd = load_checkpoint(request.checkpoint_path, device="cpu")
        arch = detect_architecture(list(sd.keys()))
        return {
            "arch": arch,
            "block_count": arch_block_count(arch),
            "block_names": arch_block_names(arch),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/checkpoint/merge-weighted")
async def merge_checkpoint_weighted(request: BlockWeightedMergeRequest):
    """
    Block-weighted (MBW) checkpoint merge — like SuperMerger.

    Modes:
      - weight:   A·(1−α) + B·α
      - add:      A + (B − C)·α  (requires model C)
      - triple:   A·(1−α−β) + B·α + C·β  (requires model C + block_weights_c)
      - twice:    merge(A, B) → merge(AB, C)  (requires model C + block_weights_c)

    block_weights are per-block blend values (26 for SD1.5, 20 for SDXL).
    base_alpha controls non-UNet keys (TE, VAE).
    """
    try:
        from services.block_weight_merge import (
            load_checkpoint, save_checkpoint,
            detect_architecture, arch_block_count,
            merge_weight, merge_add, merge_triple, merge_twice,
        )

        # Validate all input paths
        for p in [request.model_a_path, request.model_b_path, request.model_c_path]:
            if p:
                _validate_model_input(p)
        validate_path_within(request.output_path, [OUTPUT_DIR])

        # Load models
        model_a = load_checkpoint(request.model_a_path, device=request.device)
        model_b = load_checkpoint(request.model_b_path, device=request.device)

        arch = detect_architecture(list(model_a.keys()))

        # Validate block weight count matches detected architecture
        expected_blocks = arch_block_count(arch)
        if len(request.block_weights) != expected_blocks:
            raise HTTPException(
                status_code=400,
                detail=f"Detected architecture '{arch}' expects {expected_blocks} block weights, got {len(request.block_weights)}"
            )

        # Dispatch merge
        if request.mode == "weight":
            merged = merge_weight(model_a, model_b, request.block_weights, request.base_alpha, arch)
        elif request.mode == "add":
            if not request.model_c_path:
                raise HTTPException(status_code=400, detail="model_c_path required for 'add' mode")
            model_c = load_checkpoint(request.model_c_path, device=request.device)
            merged = merge_add(model_a, model_b, model_c, request.block_weights, request.base_alpha, arch)
        elif request.mode == "triple":
            if not request.model_c_path or not request.block_weights_c:
                raise HTTPException(status_code=400, detail="model_c_path and block_weights_c required for 'triple' mode")
            model_c = load_checkpoint(request.model_c_path, device=request.device)
            merged = merge_triple(model_a, model_b, model_c, request.block_weights, request.block_weights_c, request.base_alpha, arch)
        elif request.mode == "twice":
            if not request.model_c_path or not request.block_weights_c:
                raise HTTPException(status_code=400, detail="model_c_path and block_weights_c required for 'twice' mode")
            model_c = load_checkpoint(request.model_c_path, device=request.device)
            bc = request.base_alpha_c or 0.5
            merged = merge_twice(model_a, model_b, model_c, request.block_weights, request.block_weights_c, request.base_alpha, bc, arch)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")

        # Save
        save_checkpoint(merged, request.output_path)

        file_size_mb = os.path.getsize(request.output_path) / (1024 * 1024)

        return {
            "success": True,
            "message": f"Block-weighted merge ({request.mode}) successful",
            "output_path": request.output_path,
            "mode": request.mode,
            "file_size_mb": round(file_size_mb, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Block-weighted merge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Checkpoint Merge (Using Service) ==========

class CheckpointMergeRequest(BaseModel):
    """Request model for checkpoint merging"""
    checkpoint_inputs: list[dict]  # [{path: str, ratio: float}, ...]
    output_path: str
    unet_only: bool = False
    device: Literal["cpu", "cuda"] = "cpu"
    save_precision: Literal["float", "fp16", "bf16"] = "fp16"
    precision: Literal["float", "fp16", "bf16"] = "float"
    show_skipped: bool = False


@router.post("/checkpoint/merge")
async def merge_checkpoint(request: CheckpointMergeRequest):
    """
    Merge multiple checkpoint models into one.
    Uses Kohya's merge_models.py script from vendored backend.
    """
    try:
        # Security: inputs from any model dir (incl. ComfyUI); output always to output/
        try:
            for cp in request.checkpoint_inputs:
                _validate_model_input(cp["path"])
            validate_path_within(request.output_path, [OUTPUT_DIR])
        except ValidationError:
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkpoint merge error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== HuggingFace Upload (Using New Service) ==========

class HuggingFaceUploadRequest(BaseModel):
    """Request model for HuggingFace upload — matches frontend interface"""
    hf_token: str
    owner: str
    repo_name: str
    repo_type: Literal["model", "dataset", "space"] = "model"
    selected_files: list[str]
    remote_folder: str = ""
    commit_message: str = "Upload via Ktiseos-Nyx-Trainer"
    create_pr: bool = False


@router.post("/hf/upload")
async def upload_to_huggingface(request: HuggingFaceUploadRequest):
    """
    Upload files to HuggingFace Hub.
    Supports multiple files, remote folders, and pull request creation.
    """
    try:
        # Security: confine uploaded files to output directory
        try:
            for file_path in request.selected_files:
                validate_path_within(file_path, [OUTPUT_DIR])
        except ValidationError:
            raise HTTPException(status_code=403, detail="Access denied: file path outside allowed directories")

        repo_id = f"{request.owner}/{request.repo_name}"

        service_request = ServiceHFRequest(
            token=request.hf_token,
            repo_id=repo_id,
            repo_type=request.repo_type,
            file_paths=request.selected_files,
            remote_folder=request.remote_folder,
            commit_message=request.commit_message,
            create_pr=request.create_pr,
        )

        response = await lora_service.upload_to_huggingface(service_request)

        return {
            "success": response.success,
            "repo_id": response.repo_id,
            "uploaded_files": response.uploaded_files,
            "failed_files": response.failed_files,
            "error": response.error,
        }

    except Exception as e:
        logger.error(f"HuggingFace upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ValidateTokenRequest(BaseModel):
    hf_token: str


@router.post("/hf/validate-token")
async def validate_hf_token(request: ValidateTokenRequest):
    """Validate HuggingFace API token"""
    try:
        from huggingface_hub import HfApi

        if not request.hf_token or not request.hf_token.strip():
            return {"valid": False, "error": "Token is empty"}

        try:
            api = HfApi(token=request.hf_token)
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
