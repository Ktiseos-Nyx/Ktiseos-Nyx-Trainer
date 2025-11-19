"""
Utilities API Routes
Handles calculator, LoRA utilities, and HuggingFace uploads.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utilities_manager import UtilitiesManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize utilities manager
utilities_manager = UtilitiesManager()

# Active WebSocket connections for upload progress
upload_connections: List[WebSocket] = []


# ========== Pydantic Models ==========

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


class ListFilesRequest(BaseModel):
    """Request model for listing files"""
    directory: str
    file_extension: str = "safetensors"
    sort_by: str = "name"  # 'name' or 'date'


class ListFilesResponse(BaseModel):
    """Response model for file listing"""
    success: bool
    directory: str
    files: List[Dict[str, Any]]


class HuggingFaceUploadRequest(BaseModel):
    """Request model for HuggingFace upload"""
    hf_token: str
    owner: str
    repo_name: str
    repo_type: str = "model"
    selected_files: List[str]
    remote_folder: str = ""
    commit_message: str = "Upload via Ktiseos-Nyx-Trainer 🤗"
    create_pr: bool = False


class HuggingFaceUploadResponse(BaseModel):
    """Response model for HuggingFace upload"""
    success: bool
    repo_id: Optional[str] = None
    total_files: int
    uploaded_files: List[str]
    failed_files: List[Dict[str, str]]
    hf_transfer_active: bool = False
    error: Optional[str] = None


# ========== Calculator Endpoints ==========

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


# ========== LoRA File Management ==========

@router.post("/lora/list", response_model=ListFilesResponse)
async def list_lora_files(request: ListFilesRequest):
    """
    List LoRA files in a directory.
    Supports filtering by extension and sorting.
    """
    try:
        directory = request.directory.strip()

        if not os.path.isdir(directory):
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

        file_paths = utilities_manager.get_files_in_directory(
            directory,
            request.file_extension,
            request.sort_by
        )

        files = []
        for file_path in file_paths:
            path = Path(file_path)
            size = path.stat().st_size

            files.append({
                "name": path.name,
                "path": str(path),
                "size": size,
                "size_formatted": utilities_manager.format_file_size(size),
                "modified": path.stat().st_mtime
            })

        return ListFilesResponse(
            success=True,
            directory=directory,
            files=files
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List files error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lora/resize-dimensions")
async def get_resize_dimensions():
    """Get available LoRA resize dimensions"""
    return {
        "dimensions": [16, 32, 64, 128],
        "default": 32
    }


class LoRAResizeRequest(BaseModel):
    """Request model for LoRA resizing"""
    input_path: str
    output_path: str
    new_dim: int
    new_alpha: int


class LoRAResizeResponse(BaseModel):
    """Response model for LoRA resizing"""
    success: bool
    input_path: str
    output_path: str
    new_dim: int
    new_alpha: int
    message: str
    error: Optional[str] = None


@router.post("/lora/resize", response_model=LoRAResizeResponse)
async def resize_lora(request: LoRAResizeRequest):
    """
    Resize a LoRA model to a different dimension.
    Uses Derrian's enhanced resize script or Kohya's standard script.
    """
    try:
        if not os.path.exists(request.input_path):
            raise HTTPException(status_code=404, detail=f"Input file not found: {request.input_path}")

        if not request.output_path:
            raise HTTPException(status_code=400, detail="Output path is required")

        if request.new_dim <= 0 or request.new_alpha <= 0:
            raise HTTPException(status_code=400, detail="Dimension and alpha must be greater than 0")

        logger.info(f"Resizing LoRA: {request.input_path} -> {request.output_path} (dim={request.new_dim}, alpha={request.new_alpha})")

        # Call the utilities manager resize method
        success = utilities_manager.resize_lora(
            input_path=request.input_path,
            output_path=request.output_path,
            new_dim=request.new_dim,
            new_alpha=request.new_alpha
        )

        if not success:
            return LoRAResizeResponse(
                success=False,
                input_path=request.input_path,
                output_path=request.output_path,
                new_dim=request.new_dim,
                new_alpha=request.new_alpha,
                message="Resize failed",
                error="LoRA resize operation failed. Check server logs for details."
            )

        return LoRAResizeResponse(
            success=True,
            input_path=request.input_path,
            output_path=request.output_path,
            new_dim=request.new_dim,
            new_alpha=request.new_alpha,
            message=f"Successfully resized LoRA to dim={request.new_dim}, alpha={request.new_alpha}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LoRA resize error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== HuggingFace Upload ==========

@router.post("/hf/upload", response_model=HuggingFaceUploadResponse)
async def upload_to_huggingface(request: HuggingFaceUploadRequest):
    """
    Upload files to HuggingFace Hub.
    Supports multiple files, progress tracking, and pull requests.
    """
    try:
        # Progress callback placeholder (can be extended with WebSocket for real-time updates)
        uploaded_files = []

        def progress_callback(current, total, filename):
            logger.info(f"Uploading {current}/{total}: {filename}")
            uploaded_files.append(filename)

        result = utilities_manager.upload_multiple_files_to_huggingface(
            hf_token=request.hf_token,
            owner=request.owner,
            repo_name=request.repo_name,
            repo_type=request.repo_type,
            selected_files=request.selected_files,
            remote_folder=request.remote_folder,
            commit_message=request.commit_message,
            create_pr=request.create_pr,
            progress_callback=progress_callback
        )

        if not result.get("success"):
            return HuggingFaceUploadResponse(
                success=False,
                total_files=0,
                uploaded_files=[],
                failed_files=[],
                error=result.get("error", "Unknown error")
            )

        return HuggingFaceUploadResponse(
            success=True,
            repo_id=result.get("repo_id"),
            total_files=result.get("total_files", 0),
            uploaded_files=result.get("uploaded_files", []),
            failed_files=result.get("failed_files", []),
            hf_transfer_active=result.get("hf_transfer_active", False)
        )

    except Exception as e:
        logger.error(f"HuggingFace upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hf/validate-token")
async def validate_hf_token(hf_token: str):
    """Validate HuggingFace API token"""
    try:
        from huggingface_hub import HfApi, login

        if not hf_token or not hf_token.strip():
            return {"valid": False, "error": "Token is empty"}

        try:
            login(token=hf_token)
            api = HfApi()
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


# ========== WebSocket for Upload Progress ==========

async def broadcast_upload_progress(message: Dict[str, Any]):
    """
    Broadcast upload progress to all connected WebSocket clients.
    """
    disconnected = []
    for connection in upload_connections:
        try:
            await connection.send_json(message)
        except Exception:
            disconnected.append(connection)

    # Clean up disconnected clients
    for conn in disconnected:
        upload_connections.remove(conn)


@router.websocket("/hf/upload-progress")
async def upload_progress_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time HuggingFace upload progress.
    Sends updates as files are uploaded.
    """
    await websocket.accept()
    upload_connections.append(websocket)

    try:
        logger.info("Client connected to upload progress WebSocket")

        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to upload progress stream"
        })

        # Keep connection alive
        while True:
            await asyncio.sleep(1)

            # Send heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": asyncio.get_event_loop().time()
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected from upload progress WebSocket")
        upload_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        if websocket in upload_connections:
            upload_connections.remove(websocket)
