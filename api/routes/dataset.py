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


@router.get("/images-with-tags")
async def get_images_with_tags(dataset_path: str):
    """Get all images in a dataset with their associated tags"""
    try:
        from services.core.validation import validate_dataset_path, ALLOWED_IMAGE_EXTENSIONS

        # Validate dataset path
        dataset_dir = validate_dataset_path(dataset_path)

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

        # Find all images
        images = []
        for ext in ALLOWED_IMAGE_EXTENSIONS:
            images.extend(dataset_dir.glob(f"*{ext}"))
            images.extend(dataset_dir.glob(f"*{ext.upper()}"))

        # Build response with tags
        result = []
        for img_path in sorted(images):
            # Find corresponding caption file
            caption_path = img_path.with_suffix('.txt')
            tags = []

            if caption_path.exists():
                try:
                    tags_text = caption_path.read_text(encoding='utf-8').strip()
                    tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
                except Exception as e:
                    logger.warning(f"Failed to read tags for {img_path}: {e}")

            result.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "tags": tags,
                "has_tags": len(tags) > 0
            })

        return {
            "images": result,
            "total": len(result)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get images with tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    # Required
    dataset_dir: str

    # Model settings
    model: str = "SmilingWolf/wd-vit-large-tagger-v3"
    force_download: bool = False

    # Thresholds (3 separate!)
    threshold: float = 0.35
    general_threshold: Optional[float] = None
    character_threshold: Optional[float] = None

    # Output settings
    caption_extension: str = ".txt"
    caption_separator: str = ", "

    # Tag filtering
    undesired_tags: str = ""  # Formerly blacklist_tags
    tag_replacement: Optional[str] = None

    # Tag ordering
    always_first_tags: Optional[str] = None
    character_tags_first: bool = False

    # Rating tags
    use_rating_tags: bool = False
    use_rating_tags_as_last_tag: bool = False

    # Tag processing
    remove_underscore: bool = True
    character_tag_expand: bool = False

    # File handling
    append_tags: bool = False
    recursive: bool = False

    # Performance
    batch_size: int = 8
    max_workers: int = 2
    use_onnx: bool = True

    # Debug
    frequency_tags: bool = False
    debug: bool = False


@router.post("/tag")
async def start_tagging(request: TaggingRequest):
    """
    Start WD14 auto-tagging on a dataset.
    Returns job_id for monitoring progress.
    """
    try:
        # Convert to TaggingConfig (map all fields!)
        config = TaggingConfig(
            dataset_dir=request.dataset_dir,
            model=request.model,
            force_download=request.force_download,
            threshold=request.threshold,
            general_threshold=request.general_threshold,
            character_threshold=request.character_threshold,
            caption_extension=request.caption_extension,
            caption_separator=request.caption_separator,
            undesired_tags=request.undesired_tags,
            tag_replacement=request.tag_replacement,
            always_first_tags=request.always_first_tags,
            character_tags_first=request.character_tags_first,
            use_rating_tags=request.use_rating_tags,
            use_rating_tags_as_last_tag=request.use_rating_tags_as_last_tag,
            remove_underscore=request.remove_underscore,
            character_tag_expand=request.character_tag_expand,
            append_tags=request.append_tags,
            recursive=request.recursive,
            batch_size=request.batch_size,
            max_workers=request.max_workers,
            use_onnx=request.use_onnx,
            frequency_tags=request.frequency_tags,
            debug=request.debug,
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


# ========== BLIP/GIT Captioning ==========

from services import captioning_service
from services.models.captioning import BLIPConfig, GITConfig


@router.post("/caption/blip")
async def start_blip_captioning(config: BLIPConfig):
    """
    Start BLIP natural language captioning.
    Returns job_id for monitoring progress.
    """
    try:
        response = await captioning_service.start_blip_captioning(config)

        return {
            "success": response.success,
            "message": response.message,
            "job_id": response.job_id,
            "validation_errors": response.validation_errors
        }

    except Exception as e:
        logger.error(f"Failed to start BLIP captioning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/caption/git")
async def start_git_captioning(config: GITConfig):
    """
    Start GIT natural language captioning.
    Returns job_id for monitoring progress.
    """
    try:
        response = await captioning_service.start_git_captioning(config)

        return {
            "success": response.success,
            "message": response.message,
            "job_id": response.job_id,
            "validation_errors": response.validation_errors
        }

    except Exception as e:
        logger.error(f"Failed to start GIT captioning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/caption/status/{job_id}")
async def get_captioning_status(job_id: str):
    """Get captioning job status and progress"""
    try:
        status = await captioning_service.get_status(job_id)

        if not status:
            raise HTTPException(status_code=404, detail="Captioning job not found")

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
        logger.error(f"Failed to get captioning status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/caption/stop/{job_id}")
async def stop_captioning(job_id: str):
    """Stop a running captioning job"""
    try:
        stopped = await captioning_service.stop_captioning(job_id)

        if not stopped:
            raise HTTPException(status_code=404, detail="Job not found or already stopped")

        return {"success": True, "message": f"Captioning {job_id} stopped"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop captioning: {e}", exc_info=True)
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


# ========== Tag Management (Direct) ==========

class UpdateTagsRequest(BaseModel):
    """Request to update tags for a single image"""
    image_path: str
    tags: list[str]


@router.post("/update-tags")
async def update_tags_endpoint(request: UpdateTagsRequest):
    """Update tags for a single image"""
    try:
        from pathlib import Path
        from services.core.validation import validate_path

        # Get image and caption file paths
        img_path = Path(request.image_path)
        caption_path = img_path.with_suffix('.txt')

        # Write tags to caption file
        caption_path.write_text(', '.join(request.tags), encoding='utf-8')

        logger.info(f"Updated tags for {img_path.name}: {len(request.tags)} tags")

        return {
            "success": True,
            "image_path": request.image_path,
            "tags": request.tags
        }
    except Exception as e:
        logger.error(f"Failed to update tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class BulkTagOperationRequest(BaseModel):
    """Request for bulk tag operation"""
    dataset_path: str
    operation: str  # "add", "remove", "replace"
    tags: list[str]
    replace_with: str = ""


@router.post("/bulk-tag-operation")
async def bulk_tag_operation_endpoint(request: BulkTagOperationRequest):
    """Bulk tag operation on all images in dataset"""
    try:
        from pathlib import Path
        from services.core.validation import validate_dataset_path, ALLOWED_IMAGE_EXTENSIONS

        dataset_dir = validate_dataset_path(request.dataset_path)
        modified_count = 0

        # Find all caption files
        for ext in ALLOWED_IMAGE_EXTENSIONS:
            for img_path in dataset_dir.glob(f"*{ext}"):
                caption_path = img_path.with_suffix('.txt')

                if not caption_path.exists():
                    continue

                # Read current tags
                current_tags = caption_path.read_text(encoding='utf-8').strip().split(', ')
                current_tags = [t.strip() for t in current_tags if t.strip()]

                # Apply operation
                if request.operation == "add":
                    current_tags.extend([t for t in request.tags if t not in current_tags])
                elif request.operation == "remove":
                    current_tags = [t for t in current_tags if t not in request.tags]
                elif request.operation == "replace":
                    current_tags = [request.replace_with if t in request.tags else t for t in current_tags]

                # Write back
                caption_path.write_text(', '.join(current_tags), encoding='utf-8')
                modified_count += 1

        logger.info(f"Bulk {request.operation} operation: modified {modified_count} files")

        return {
            "success": True,
            "modified_count": modified_count,
            "operation": request.operation
        }
    except Exception as e:
        logger.error(f"Bulk tag operation failed: {e}", exc_info=True)
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
