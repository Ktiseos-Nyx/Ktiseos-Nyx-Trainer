"""
Dataset API Routes
Handles dataset management, tagging, and caption editing via new service layer.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel

# Import new services
from services import dataset_service, tagging_service, caption_service
from services.models.dataset import CreateDatasetRequest
from services.models.tagging import TaggingConfig, TaggerModel
from services.models.caption import (
    AddTriggerWordRequest,
    RemoveTagsRequest,
    ReplaceTextRequest,
    ReadCaptionRequest,
    WriteCaptionRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ========== Dataset Management ==========

@router.get("/list")
async def list_datasets():
    """List all datasets with metadata"""
    try:
        response = await dataset_service.list_datasets()
        return {
            "datasets": [d.dict() for d in response.datasets],
            "total": response.total
        }
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_dataset(request: CreateDatasetRequest):
    """Create a new dataset directory"""
    try:
        dataset_info = await dataset_service.create_dataset(request)
        return dataset_info.dict()
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get dataset information and metadata"""
    try:
        dataset_info = await dataset_service.get_dataset(dataset_name)
        return dataset_info.dict()
    except Exception as e:
        logger.error(f"Failed to get dataset: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_name}/files")
async def list_dataset_files(dataset_name: str):
    """List all files in a dataset"""
    try:
        response = await dataset_service.list_files(dataset_name)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to list files: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its contents"""
    try:
        await dataset_service.delete_dataset(dataset_name)
        return {"success": True, "message": f"Dataset {dataset_name} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))


# ========== WD14 Tagging ==========

class TaggingRequest(BaseModel):
    """Request to start WD14 tagging"""
    dataset_dir: str
    model: str = "SmilingWolf/wd-vit-large-tagger-v3"
    threshold: float = 0.35
    blacklist_tags: str = ""
    batch_size: int = 8
    use_onnx: bool = True


@router.post("/tag")
async def start_tagging(request: TaggingRequest):
    """
    Start WD14 auto-tagging on a dataset.
    Returns job_id for monitoring progress.
    """
    try:
        # Convert to TaggingConfig
        config = TaggingConfig(
            dataset_dir=request.dataset_dir,
            model=request.model,
            threshold=request.threshold,
            blacklist_tags=request.blacklist_tags,
            batch_size=request.batch_size,
            use_onnx=request.use_onnx,
        )

        response = await tagging_service.start_tagging(config)

        return {
            "success": response.success,
            "message": response.message,
            "job_id": response.job_id,
            "validation_errors": response.validation_errors
        }

    except Exception as e:
        logger.error(f"Failed to start tagging: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tag/status/{job_id}")
async def get_tagging_status(job_id: str):
    """Get tagging job status and progress"""
    try:
        status = await tagging_service.get_status(job_id)

        if not status:
            raise HTTPException(status_code=404, detail="Tagging job not found")

        return {
            "job_id": status.job_id,
            "status": status.status.value,
            "progress": status.progress,
            "current_image": status.current_image,
            "total_images": status.total_images,
            "error": status.error
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tagging status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tag/stop/{job_id}")
async def stop_tagging(job_id: str):
    """Stop a running tagging job"""
    try:
        stopped = await tagging_service.stop_tagging(job_id)

        if not stopped:
            raise HTTPException(status_code=404, detail="Job not found or already stopped")

        return {"success": True, "message": f"Tagging {job_id} stopped"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop tagging: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Caption Editing ==========

@router.post("/captions/add-trigger")
async def add_trigger_word(request: AddTriggerWordRequest):
    """Add trigger word to all captions in a dataset"""
    try:
        response = await caption_service.add_trigger_word(request)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to add trigger word: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/remove-tags")
async def remove_tags(request: RemoveTagsRequest):
    """Remove specific tags from all captions"""
    try:
        response = await caption_service.remove_tags(request)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to remove tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/replace")
async def replace_text(request: ReplaceTextRequest):
    """Replace text in all captions (supports regex)"""
    try:
        response = await caption_service.replace_text(request)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to replace text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/read")
async def read_caption(request: ReadCaptionRequest):
    """Read a single caption file"""
    try:
        response = await caption_service.read_caption(request)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to read caption: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/captions/write")
async def write_caption(request: WriteCaptionRequest):
    """Write a single caption file"""
    try:
        response = await caption_service.write_caption(request)
        return response.dict()
    except Exception as e:
        logger.error(f"Failed to write caption: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== File Upload ==========
# Note: This is kept from old routes for file uploads
# Can be enhanced later with proper upload service

@router.post("/upload-batch")
async def upload_batch(
    files: list[UploadFile] = File(...),
    dataset_name: str = "my_dataset"
):
    """
    Upload multiple files to a dataset.
    Creates dataset directory if it doesn't exist.
    """
    try:
        from services.core.validation import validate_dataset_path, ALLOWED_IMAGE_EXTENSIONS

        # Validate and get dataset path
        dataset_path = validate_dataset_path(dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        errors = []

        for file in files:
            try:
                # Check file extension
                file_path = Path(file.filename)
                if file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                    errors.append(f"{file.filename}: Invalid file type")
                    continue

                # Save file
                destination = dataset_path / file.filename
                content = await file.read()
                destination.write_bytes(content)

                uploaded_files.append(str(destination))

            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")

        return {
            "success": len(uploaded_files) > 0,
            "uploaded": len(uploaded_files),
            "files": uploaded_files,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Failed to upload files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: WebSocket routes for real-time logs are in services/websocket.py
# Use: ws://host/ws/jobs/{job_id}/logs for tagging log streaming
# Use: ws://host/ws/jobs/{job_id}/status for tagging status updates
