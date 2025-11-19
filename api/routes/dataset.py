"""
Dataset API Routes
Handles dataset upload, tagging, preparation, and management.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from core.log_streamer import get_tagging_log_streamer
from shared_managers import get_dataset_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Get global tagging log streamer
tagging_log_streamer = get_tagging_log_streamer()


class DatasetInfo(BaseModel):
    """Dataset information"""

    name: str
    path: str
    image_count: int
    tags_present: bool


class TaggingRequest(BaseModel):
    """Request to tag images"""

    dataset_path: str
    model: str = "wd14-vit-v2"  # Default tagger model
    threshold: float = 0.35
    batch_size: int = 4


@router.get("/list")
async def list_datasets():
    """List available datasets"""
    try:
        # Use /workspace/datasets if it exists, otherwise use local datasets directory
        if Path("/workspace/datasets").exists():
            datasets_dir = Path("/workspace/datasets")
        else:
            # Local development - use project datasets directory
            datasets_dir = Path(__file__).parent.parent.parent / "datasets"
            datasets_dir.mkdir(exist_ok=True)

        if not datasets_dir.exists():
            return {"datasets": []}

        datasets = []
        for item in datasets_dir.iterdir():
            if item.is_dir():
                # Count images
                image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                images = [f for f in item.rglob("*") if f.suffix.lower() in image_exts]

                # Check if tags exist
                tags_exist = any(
                    (item / f.stem).with_suffix(".txt").exists() for f in images
                )

                datasets.append(
                    DatasetInfo(
                        name=item.name,
                        path=str(item),
                        image_count=len(images),
                        tags_present=tags_exist,
                    )
                )

        return {"datasets": datasets}

    except Exception as e:
        logger.error(f"Failed to list datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-batch")
async def upload_dataset_batch(
    files: List[UploadFile] = File(...), dataset_name: str = "new_dataset"
):
    """
    Upload multiple images at once (drag & drop support).
    Much better than Jupyter widgets!
    """
    try:
        dataset_dir = Path("/workspace/datasets") / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = []

        for file in files:
            # Check if it's an image
            if not file.content_type or not file.content_type.startswith("image/"):
                logger.warning(f"Skipping non-image file: {file.filename}")
                continue

            file_path = dataset_dir / file.filename

            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            uploaded_files.append(str(file_path))
            logger.info(f"Uploaded: {file_path}")

        return {
            "success": True,
            "dataset": dataset_name,
            "uploaded": len(uploaded_files),
            "files": uploaded_files,
        }

    except Exception as e:
        logger.error(f"Failed to upload dataset batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tag")
async def tag_dataset(request: TaggingRequest, background_tasks: BackgroundTasks):
    """
    Auto-tag images using WD14 tagger.
    Runs in background to avoid blocking.
    """
    try:
        dataset_manager = get_dataset_manager()

        # Validate path
        dataset_path = Path(request.dataset_path)
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Start tagging in background
        background_tasks.add_task(_run_tagging, dataset_manager, request)

        return {
            "success": True,
            "message": "Tagging started in background",
            "dataset": str(dataset_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start tagging: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _run_tagging(dataset_manager, request: TaggingRequest):
    """Background task for tagging"""
    try:
        logger.info(f"Starting tagging for: {request.dataset_path}")

        # Clear any old logs
        tagging_log_streamer.clear()

        # Map frontend model names to repo IDs
        model_mapping = {
            "wd14-vit-v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
            "wd14-vit-v3": "SmilingWolf/wd-vit-large-tagger-v3",
            "wd14-swinv2-v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
            "wd14-convnext-v2": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
            "wd14-moat-v2": "SmilingWolf/wd-v1-4-moat-tagger-v2",
        }

        tagger_model = model_mapping.get(request.model, request.model)

        # Call dataset_manager tagging method
        # Note: This is synchronous, so we need to run in executor
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()

        success = await loop.run_in_executor(
            executor,
            dataset_manager.tag_images,
            request.dataset_path,
            "anime",  # method
            tagger_model,
            request.threshold,
            "",  # blacklist_tags
            ".txt",  # caption_extension
        )

        if success:
            logger.info(f"Tagging completed successfully for: {request.dataset_path}")
        else:
            logger.error(f"Tagging failed for: {request.dataset_path}")

    except Exception as e:
        logger.error(f"Tagging failed: {e}", exc_info=True)


@router.get("/preview")
async def preview_dataset(dataset_path: str, limit: int = 20):
    """Get preview of dataset images (for gallery view)"""
    try:
        path = Path(dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [
            str(f.relative_to(path))
            for f in path.rglob("*")
            if f.suffix.lower() in image_exts
        ][:limit]

        return {"dataset": str(path), "images": images, "total": len(images)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preview dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_dataset(name: str):
    """Create a new empty dataset directory"""
    try:
        dataset_dir = Path("/workspace/datasets") / name

        if dataset_dir.exists():
            raise HTTPException(status_code=400, detail="Dataset already exists")

        dataset_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Created dataset: {dataset_dir}")

        return {"success": True, "name": name, "path": str(dataset_dir)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}")
async def delete_dataset(name: str):
    """Delete a dataset"""
    try:
        dataset_dir = Path("/workspace/datasets") / name

        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        import shutil

        shutil.rmtree(dataset_dir)
        logger.info(f"Deleted dataset: {dataset_dir}")

        return {"success": True, "message": f"Deleted dataset: {name}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========== Tag Management Endpoints ==========


class ImageTagsResponse(BaseModel):
    """Response with image and its tags"""

    image_path: str
    image_name: str
    tags: List[str]
    has_tags: bool


class UpdateTagsRequest(BaseModel):
    """Request to update tags for a single image"""

    image_path: str
    tags: List[str]


class BulkTagOperation(BaseModel):
    """Request for bulk tag operations"""

    dataset_path: str
    operation: str  # 'add', 'remove', 'replace'
    tags: List[str]
    replace_with: Optional[str] = ""


class TriggerWordRequest(BaseModel):
    """Request to inject trigger word"""

    dataset_path: str
    trigger_word: str
    position: str = "start"  # 'start' or 'end'


@router.get("/images-with-tags")
async def get_images_with_tags(dataset_path: str):
    """
    Get all images in a dataset with their tags.
    Returns a list of images with their associated caption tags.
    """
    try:
        path = Path(dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images_with_tags = []

        for img_file in sorted(path.rglob("*")):
            if img_file.suffix.lower() not in image_exts:
                continue

            # Look for corresponding .txt file
            txt_file = img_file.with_suffix(".txt")

            tags = []
            has_tags = False

            if txt_file.exists():
                try:
                    tags_content = txt_file.read_text(encoding="utf-8").strip()
                    tags = [
                        tag.strip() for tag in tags_content.split(",") if tag.strip()
                    ]
                    has_tags = True
                except Exception as e:
                    logger.warning(f"Failed to read tags for {img_file.name}: {e}")

            images_with_tags.append(
                {
                    "image_path": str(img_file),
                    "image_name": img_file.name,
                    "tags": tags,
                    "has_tags": has_tags,
                }
            )

        return {
            "dataset_path": str(path),
            "images": images_with_tags,
            "total": len(images_with_tags),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get images with tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-tags")
async def update_image_tags(request: UpdateTagsRequest):
    """
    Update tags for a single image.
    Writes tags to the corresponding .txt file.
    """
    try:
        img_path = Path(request.image_path)
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Write tags to .txt file
        txt_file = img_path.with_suffix(".txt")
        tags_string = ", ".join(request.tags)

        txt_file.write_text(tags_string, encoding="utf-8")
        logger.info(f"Updated tags for {img_path.name}: {len(request.tags)} tags")

        return {"success": True, "image": str(img_path), "tags": request.tags}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-tag-operation")
async def bulk_tag_operation(request: BulkTagOperation):
    """
    Perform bulk tag operations across all images in a dataset.
    Operations: add, remove, replace
    """
    try:
        dataset_manager = get_dataset_manager()
        dataset_dir = request.dataset_path

        if not Path(dataset_dir).exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        if request.operation == "add":
            # Add tags to all images
            for tag in request.tags:
                # We'll implement a simple add by reading each file and appending
                path = Path(dataset_dir)
                image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

                for img_file in path.rglob("*"):
                    if img_file.suffix.lower() not in image_exts:
                        continue

                    txt_file = img_file.with_suffix(".txt")
                    existing_tags = []

                    if txt_file.exists():
                        existing_tags = [
                            t.strip()
                            for t in txt_file.read_text(encoding="utf-8").split(",")
                            if t.strip()
                        ]

                    if tag not in existing_tags:
                        existing_tags.append(tag)

                    txt_file.write_text(", ".join(existing_tags), encoding="utf-8")

            return {"success": True, "operation": "add", "tags": request.tags}

        elif request.operation == "remove":
            # Use dataset_manager's remove_tags method
            dataset_manager.remove_tags(dataset_dir, request.tags)
            return {"success": True, "operation": "remove", "tags": request.tags}

        elif request.operation == "replace":
            # Use dataset_manager's search_and_replace_tags method
            dataset_manager.search_and_replace_tags(
                dataset_dir,
                search_tags=request.tags,
                replace_with=request.replace_with or "",
                search_mode="OR",
            )
            return {
                "success": True,
                "operation": "replace",
                "tags": request.tags,
                "replace_with": request.replace_with,
            }

        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid operation: {request.operation}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk tag operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inject-trigger-word")
async def inject_trigger_word(request: TriggerWordRequest):
    """
    Inject a trigger word into all captions in a dataset.
    Position can be 'start' or 'end'.
    """
    try:
        path = Path(request.dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        modified_count = 0

        for img_file in path.rglob("*"):
            if img_file.suffix.lower() not in image_exts:
                continue

            txt_file = img_file.with_suffix(".txt")
            if not txt_file.exists():
                continue

            tags = [
                t.strip()
                for t in txt_file.read_text(encoding="utf-8").split(",")
                if t.strip()
            ]

            # Check if trigger word already exists
            if request.trigger_word in tags:
                continue

            # Add trigger word at specified position
            if request.position == "start":
                tags.insert(0, request.trigger_word)
            else:  # end
                tags.append(request.trigger_word)

            txt_file.write_text(", ".join(tags), encoding="utf-8")
            modified_count += 1

        logger.info(
            f"Injected trigger word '{request.trigger_word}' into {modified_count} files"
        )

        return {
            "success": True,
            "trigger_word": request.trigger_word,
            "position": request.position,
            "modified": modified_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject trigger word: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/logs")
async def tagging_logs_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time tagging logs.
    Streams logs from WD14/BLIP tagging subprocess via log_streamer.
    """
    await websocket.accept()
    tagging_log_streamer.add_connection(websocket)

    try:
        logger.info("Client connected to tagging logs WebSocket")

        # Send initial connection message
        await websocket.send_json(
            {"type": "connected", "message": "✓ Connected to tagging logs"}
        )

        # Broadcast pending logs every 0.1 seconds
        while True:
            await tagging_log_streamer.broadcast_pending()
            await asyncio.sleep(0.1)  # 10 updates per second

    except WebSocketDisconnect:
        logger.info("Client disconnected from tagging logs WebSocket")
        tagging_log_streamer.remove_connection(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        tagging_log_streamer.remove_connection(websocket)
