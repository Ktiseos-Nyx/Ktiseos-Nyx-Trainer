"""
API routes for model and VAE downloads.
"""

import glob
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Lazy-load ModelManager to avoid circular imports
_model_manager = None

def get_model_manager():
    global _model_manager
    if _model_manager is None:
        from core.managers import ModelManager
        _model_manager = ModelManager()
    return _model_manager


class DownloadRequest(BaseModel):
    url: str
    download_type: str  # "model" or "vae"
    model_type: Optional[str] = None  # "sdxl", "sd15", "flux", "sd3.5"


class ModelInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    type: str  # "model" or "vae"


@router.post("/download")
async def download_model_or_vae(request: DownloadRequest):
    """
    Download a model or VAE from HuggingFace or Civitai.

    Returns download status and file information.
    """
    try:
        manager = get_model_manager()

        # Determine target directory
        if request.download_type == "vae":
            target_dir = manager.vae_dir
        else:
            target_dir = manager.pretrained_model_dir

        # Download the file
        result = manager.download_model_or_vae(
            url=request.url,
            download_dir=target_dir
        )

        if result.get("success"):
            file_path = result.get("file_path")
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024) if file_path and os.path.exists(file_path) else 0

            return {
                "success": True,
                "message": result.get("message", "Download completed"),
                "file_path": file_path,
                "file_name": os.path.basename(file_path) if file_path else None,
                "size_mb": round(file_size_mb, 2),
                "type": request.download_type
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Download failed"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_models_and_vaes():
    """
    List all downloaded models and VAEs.

    Returns separate lists for models and VAEs with file information.
    """
    try:
        manager = get_model_manager()

        def get_files_info(directory: str, file_type: str) -> List[ModelInfo]:
            files = []
            if os.path.exists(directory):
                # Look for safetensors, ckpt, and pt files
                patterns = ["*.safetensors", "*.ckpt", "*.pt"]
                for pattern in patterns:
                    for file_path in glob.glob(os.path.join(directory, pattern)):
                        size_bytes = os.path.getsize(file_path)
                        size_mb = size_bytes / (1024 * 1024)

                        files.append({
                            "name": os.path.basename(file_path),
                            "path": file_path,
                            "size_mb": round(size_mb, 2),
                            "type": file_type
                        })

            # Sort by name
            files.sort(key=lambda x: x["name"].lower())
            return files

        models = get_files_info(manager.pretrained_model_dir, "model")
        vaes = get_files_info(manager.vae_dir, "vae")

        return {
            "success": True,
            "models": models,
            "vaes": vaes,
            "model_dir": manager.pretrained_model_dir,
            "vae_dir": manager.vae_dir
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{file_type}/{file_name}")
async def delete_model_or_vae(file_type: str, file_name: str):
    """
    Delete a model or VAE file.

    Args:
        file_type: Either "model" or "vae"
        file_name: Name of the file to delete
    """
    try:
        manager = get_model_manager()

        if file_type == "vae":
            target_dir = manager.vae_dir
        elif file_type == "model":
            target_dir = manager.pretrained_model_dir
        else:
            raise HTTPException(status_code=400, detail="Invalid file_type. Must be 'model' or 'vae'")

        file_path = os.path.join(target_dir, file_name)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Delete the file
        os.remove(file_path)

        return {
            "success": True,
            "message": f"Deleted {file_name}",
            "file_path": file_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/popular")
async def get_popular_models():
    """
    Get a list of popular/recommended models and VAEs with direct download URLs.

    Useful for quick-start suggestions.
    """
    return {
        "success": True,
        "models": {
            "sdxl": [
                {
                    "name": "SDXL Base 1.0",
                    "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                    "description": "Official SDXL base model from Stability AI"
                },
                {
                    "name": "Pony Diffusion V6 XL",
                    "url": "https://civitai.com/api/download/models/290640",
                    "description": "Popular anime/cartoon model"
                }
            ],
            "sd15": [
                {
                    "name": "SD 1.5",
                    "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                    "description": "Official SD 1.5 model"
                }
            ],
            "flux": [
                {
                    "name": "FLUX.1 Dev",
                    "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
                    "description": "FLUX.1 development model"
                }
            ]
        },
        "vaes": [
            {
                "name": "SDXL VAE",
                "url": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                "description": "Official SDXL VAE"
            },
            {
                "name": "SD 1.5 VAE (MSE)",
                "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                "description": "Improved VAE for SD 1.5"
            }
        ]
    }
