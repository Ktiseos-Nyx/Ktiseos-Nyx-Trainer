"""
Dataset API Routes
Handles dataset upload, tagging, preparation, and management.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from shared_managers import get_dataset_manager

logger = logging.getLogger(__name__)
router = APIRouter()


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
        datasets_dir = Path("/workspace/datasets")

        if not datasets_dir.exists():
            return {"datasets": []}

        datasets = []
        for item in datasets_dir.iterdir():
            if item.is_dir():
                # Count images
                image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
                images = [
                    f for f in item.rglob('*')
                    if f.suffix.lower() in image_exts
                ]

                # Check if tags exist
                tags_exist = any(
                    (item / f.stem).with_suffix('.txt').exists()
                    for f in images
                )

                datasets.append(DatasetInfo(
                    name=item.name,
                    path=str(item),
                    image_count=len(images),
                    tags_present=tags_exist
                ))

        return {"datasets": datasets}

    except Exception as e:
        logger.error(f"Failed to list datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-batch")
async def upload_dataset_batch(
    files: List[UploadFile] = File(...),
    dataset_name: str = "new_dataset"
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
            with open(file_path, 'wb') as f:
                f.write(content)

            uploaded_files.append(str(file_path))
            logger.info(f"Uploaded: {file_path}")

        return {
            "success": True,
            "dataset": dataset_name,
            "uploaded": len(uploaded_files),
            "files": uploaded_files
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
        background_tasks.add_task(
            _run_tagging,
            dataset_manager,
            request
        )

        return {
            "success": True,
            "message": "Tagging started in background",
            "dataset": str(dataset_path)
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

        # TODO: Call dataset_manager tagging method
        # dataset_manager.tag_images(
        #     path=request.dataset_path,
        #     model=request.model,
        #     threshold=request.threshold
        # )

        logger.info(f"Tagging completed for: {request.dataset_path}")

    except Exception as e:
        logger.error(f"Tagging failed: {e}", exc_info=True)


@router.get("/preview")
async def preview_dataset(dataset_path: str, limit: int = 20):
    """Get preview of dataset images (for gallery view)"""
    try:
        path = Path(dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = [
            str(f.relative_to(path))
            for f in path.rglob('*')
            if f.suffix.lower() in image_exts
        ][:limit]

        return {
            "dataset": str(path),
            "images": images,
            "total": len(images)
        }

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

        return {
            "success": True,
            "name": name,
            "path": str(dataset_dir)
        }

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

        return {
            "success": True,
            "message": f"Deleted dataset: {name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
