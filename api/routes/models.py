"""
API routes for model and VAE downloads.
"""

import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import model_service
from services.jobs.job_manager import job_manager
from services.models.job import JobStatus, JobStatusEnum, JobType, JobCreateResponse
from services.models.model_download import (
    DownloadConfig,
    ModelType,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _run_download_job(config: DownloadConfig, download_type: ModelType, job) -> dict:
    """
    Coroutine body for a DOWNLOAD job: run the download and return a JSON-safe
    dict (enums coerced to their values) for storage on ``job.result``. A failed
    download keeps ``success=False`` so the job is marked FAILED, not COMPLETED.

    Receives the Job so download progress can be mirrored onto ``job.progress``
    (monotonic — never moves backwards) for the status-polling client.
    """
    def _on_progress(pct: int) -> None:
        if pct > job.progress:
            job.progress = pct

    resp = await model_service.download_model_or_vae(config, progress_callback=_on_progress)
    method = resp.download_method
    return {
        "success": resp.success,
        "message": resp.message,
        "file_path": resp.file_path,
        "file_name": resp.file_name,
        "size_mb": resp.size_mb,
        "download_method": getattr(method, "value", method),
        "error": resp.error,
        "type": getattr(download_type, "value", download_type),
    }


class DownloadRequest(BaseModel):
    url: str
    download_type: ModelType = ModelType.MODEL
    filename: Optional[str] = None  # Required for Civitai, optional for HuggingFace
    model_type: Optional[str] = None  # "sdxl", "sd15", "flux", "sd3.5"
    # Destination routing: "training" (default) or "comfyui"
    destination: Optional[str] = "training"
    # ComfyUI subfolder when destination="comfyui"
    # e.g. "checkpoints", "diffusion_models", "vae", "loras", "text_encoders"
    comfyui_folder: Optional[str] = None


@router.post("/download", response_model=JobCreateResponse)
async def download_model_or_vae(request: DownloadRequest):
    """
    Start a model/VAE download from HuggingFace or Civitai.

    Returns immediately with a ``job_id``; the actual download runs in the
    background. Poll ``GET /models/download/status/{job_id}`` for progress and
    the final file info. (Synchronous downloads previously held the HTTP request
    open for the whole transfer and timed out into 502s behind the tunnel.)
    """
    try:
        # Import settings to get API keys
        from api.routes.settings import get_api_keys
        api_keys = get_api_keys()

        # Determine API token based on URL
        api_token = None
        if "huggingface.co" in request.url:
            api_token = api_keys.get("huggingface_token")
        elif "civitai.com" in request.url:
            api_token = api_keys.get("civitai_api_key")

        # Determine target directory based on destination + download type
        if request.destination == "comfyui":
            from api.routes.settings import get_comfyui_models_path
            import os as _os
            comfyui_models = get_comfyui_models_path()
            if not comfyui_models:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "ComfyUI models path is not configured. "
                        "Set it in Settings → ComfyUI or via the COMFYUI_MODELS_PATH env var."
                    )
                )
            folder = request.comfyui_folder or "checkpoints"
            target_dir = _os.path.join(comfyui_models, folder)
            _os.makedirs(target_dir, exist_ok=True)
        elif request.download_type == ModelType.VAE:
            target_dir = str(model_service.vae_dir)
        elif request.download_type == ModelType.LORA:
            target_dir = str(model_service.lora_dir)
        else:
            target_dir = str(model_service.pretrained_model_dir)

        # Create download config
        config = DownloadConfig(
            url=request.url,
            download_dir=target_dir,
            filename=request.filename,
            api_token=api_token,
            model_type=request.download_type
        )

        # Run the download as a background job and return immediately. Holding the
        # request open for the whole multi-GB download is what timed out into 502s
        # behind the cloudflared tunnel; the client polls /download/status/{job_id}.
        job_id = job_manager.run_coroutine_job(
            JobType.DOWNLOAD,
            lambda job: _run_download_job(config, request.download_type, job),
        )
        return JobCreateResponse(
            job_id=job_id,
            status=JobStatusEnum.RUNNING,
            message="Download started",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/status/{job_id}", response_model=JobStatus)
async def get_download_status(job_id: str):
    """
    Get the status of a background download job.

    Returns the job's status (running/completed/failed), progress, and — on
    completion — the ``result`` payload with file_path/file_name/size_mb/method.
    """
    status = await job_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Download job not found")
    return status


@router.get("/list")
async def list_models_and_vaes():
    """
    List all downloaded models and VAEs.

    Returns separate lists for models and VAEs with file information.
    """
    try:
        result = await model_service.list_models()

        if result.success:
            return {
                "success": True,
                "models": [m.dict() for m in result.models],
                "vaes": [v.dict() for v in result.vaes],
                "loras": [lora.dict() for lora in result.loras],
                "model_dir": result.model_dir,
                "vae_dir": result.vae_dir,
                "lora_dir": result.lora_dir
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to list models")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{file_type}/{file_name}")
async def delete_model_or_vae(file_type: str, file_name: str):
    """
    Delete a model or VAE file.

    Args:
        file_type: Either "model", "vae", or "lora"
        file_name: Name of the file to delete
    """
    try:
        # Determine target directory
        if file_type == "vae":
            target_dir = model_service.vae_dir
        elif file_type == "model":
            target_dir = model_service.pretrained_model_dir
        elif file_type == "lora":
            target_dir = model_service.lora_dir
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid file_type. Must be 'model', 'vae', or 'lora'"
            )

        file_path = target_dir / Path(file_name).name

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Delete using service
        result = await model_service.delete_model(str(file_path))

        if result.success:
            return {
                "success": True,
                "message": result.message,
                "file_path": result.file_path
            }
        else:
            raise HTTPException(status_code=500, detail=result.message)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
async def cancel_downloads():
    """
    Cancel all running model downloads.

    Kills aria2c, wget, and hf-transfer processes.
    """
    try:
        import psutil

        killed = []

        # Find and kill download processes using psutil
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info['name']
                if proc_name in ['aria2c', 'wget', 'hf-transfer']:
                    proc.terminate()  # Send SIGTERM
                    killed.append(f"{proc_name} (PID: {proc.info['pid']})")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if killed:
            return {
                "success": True,
                "message": f"Cancelled downloads: {', '.join(killed)}",
                "killed_processes": killed
            }
        else:
            return {
                "success": False,
                "message": "No active downloads found",
                "killed_processes": []
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/popular")
async def get_popular_models():
    """
    Get a list of supported models and VAEs with direct download URLs.

    Fallback endpoint — the primary source of truth is the Node.js route
    at frontend/app/api/models/popular/route.ts. This Python version exists
    as a safety net in case the server.js whitelist is misconfigured.
    """
    return {
        "success": True,
        "models": {
            "sdxl": [
                {
                    "name": "SDXL Base 1.0",
                    "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                    "filename": "sd_xl_base_1.0.safetensors",
                    "description": "Official SDXL base model from Stability AI",
                },
                {
                    "name": "Illustrious XL",
                    "url": "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors",
                    "filename": "Illustrious-XL-v0.1.safetensors",
                    "description": "High-quality anime/illustration model",
                },
                {
                    "name": "Pony Diffusion V6 XL",
                    "url": "https://civitai.com/api/download/models/290640",
                    "filename": "ponyDiffusionV6XL_v6.safetensors",
                    "description": "Anime/cartoon model from Civitai",
                },
                {
                    "name": "NoobAI XL v1.1 (EPS)",
                    "url": "https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors",
                    "filename": "NoobAI-XL-v1.1.safetensors",
                    "description": "NoobAI XL v1.1 — Illustrious-based anime model (EPS prediction)",
                },
            ],
            "sd15": [
                {
                    "name": "SD 1.5",
                    "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                    "filename": "v1-5-pruned-emaonly.safetensors",
                    "description": "Official SD 1.5 model",
                },
            ],
            "flux": [
                {
                    "name": "FLUX.1 Dev (Gated)",
                    "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
                    "filename": "flux1-dev.safetensors",
                    "description": "FLUX.1 development model — requires HuggingFace login & license acceptance",
                },
                {
                    "name": "FLUX.1 Schnell",
                    "url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
                    "filename": "flux1-schnell.safetensors",
                    "description": "FLUX.1 fast inference model — Apache 2.0 license, no login required",
                },
            ],
            "sd3.5": [
                {
                    "name": "SD 3.5 Large (Gated)",
                    "url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors",
                    "filename": "sd3.5_large.safetensors",
                    "description": "Stability AI SD 3.5 Large — may require HuggingFace login & license acceptance",
                },
            ],
            "chroma": [
                {
                    "name": "Chroma1 Base",
                    "url": "https://huggingface.co/lodestones/Chroma1-Base/resolve/main/Chroma1-Base.safetensors",
                    "filename": "Chroma1-Base.safetensors",
                    "description": "Chroma base model by Lodestone — no CLIP-L needed, T5-XXL only",
                },
                {
                    "name": "Chroma1 HD",
                    "url": "https://huggingface.co/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors",
                    "filename": "Chroma1-HD.safetensors",
                    "description": "Chroma HD model by Lodestone — higher resolution variant",
                },
            ],
            "anima": [
                {
                    "name": "Anima Base v1.0 (Diffusion Model)",
                    "url": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
                    "filename": "anima-base-v1.0.safetensors",
                    "description": "Anima base model v1.0 by Circlestone Labs — Qwen3 + T5 dual encoder architecture",
                },
                {
                    "name": "Anima Preview 3 (Diffusion Model)",
                    "url": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview3-base.safetensors",
                    "filename": "anima-preview3-base.safetensors",
                    "description": "Anima preview 3 base — replaces the original preview release",
                },
                {
                    "name": "Anima Text Encoder (Qwen3 0.6B)",
                    "url": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
                    "filename": "qwen_3_06b_base.safetensors",
                    "description": "Qwen3 0.6B text encoder for Anima — required component",
                },
            ],
            "lumina": [
                {
                    "name": "Lumina Image 2.0 (Diffusion Model)",
                    "url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors",
                    "filename": "lumina_2_model_bf16.safetensors",
                    "description": "Lumina Image 2.0 diffusion model (5.2GB) — 2B parameter flow-based DiT",
                },
                {
                    "name": "Lumina Gemma2 Text Encoder",
                    "url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/text_encoders/gemma_2_2b_fp16.safetensors",
                    "filename": "gemma_2_2b_fp16.safetensors",
                    "description": "Gemma2 2B text encoder for Lumina — required component (5.2GB)",
                },
            ],
            "hunyuanimage": [
                {
                    "name": "HunyuanImage 3.0 (Manual Download)",
                    "url": "",
                    "filename": "",
                    "description": "168GB model (32 sharded files) — too large for web download. Run: huggingface-cli download tencent/HunyuanImage-3.0 --local-dir ./pretrained_model/hunyuanimage",
                    "manualOnly": True,
                    "repoUrl": "https://huggingface.co/tencent/HunyuanImage-3.0",
                },
            ],
        },
        "vaes": [
            {
                "name": "SDXL VAE",
                "url": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                "filename": "sdxl_vae.safetensors",
                "description": "Official SDXL VAE",
            },
            {
                "name": "SD 1.5 VAE (MSE)",
                "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                "filename": "vae-ft-mse-840000-ema-pruned.safetensors",
                "description": "Improved VAE for SD 1.5",
            },
            {
                "name": "Flux VAE",
                "url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
                "filename": "flux_ae.safetensors",
                "description": "Flux autoencoder (335MB) — same VAE for both Dev and Schnell, from ungated Schnell repo",
            },
            {
                "name": "Anima VAE",
                "url": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
                "filename": "qwen_image_vae.safetensors",
                "description": "Anima-specific VAE — does NOT use Flux VAE",
            },
            {
                "name": "Chroma VAE",
                "url": "https://huggingface.co/lodestones/Chroma1-Base/resolve/main/vae/diffusion_pytorch_model.safetensors",
                "filename": "chroma_vae.safetensors",
                "description": "Chroma VAE from Lodestone — included in Chroma repos",
            },
            {
                "name": "Lumina VAE",
                "url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors",
                "filename": "lumina_ae.safetensors",
                "description": "Lumina Image 2.0 autoencoder (335MB)",
            },
        ],
    }
