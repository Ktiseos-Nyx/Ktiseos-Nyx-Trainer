"""API routes for the additive multi-source download system.

Exposes registered source adapters (Arc En Ciel, future Civitai, HuggingFace, …)
as searchable, browsable, downloadable endpoints behind ``/api/sources/*``.

See ``docs/specs/2026-07-03-additive-download-system.md``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services import model_service
from services.jobs.job_manager import job_manager
from services.models.job import JobStatus, JobStatusEnum, JobType, JobCreateResponse
from services.models.model_download import DownloadConfig, ModelType
from services.sources import get_adapter, list_adapters
from services.sources.base import DestType, SearchQuery

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / response models ──────────────────────────────────────────────


class SourceInfo(BaseModel):
    name: str
    credential_kind: str
    base_model_classes: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    term: str = ""
    sort: Optional[str] = None
    page: int = 1
    limit: int = 20
    base_model: Optional[str] = None
    model_type: Optional[str] = None


class SourceDownloadRequest(BaseModel):
    model_id: str
    version_id: str
    destination: str = "training"
    comfyui_folder: Optional[str] = None


# ── Dest-type → directory mapping (spec §4.4) ─────────────────────────────


def _resolve_target_dir(
    dest_type: DestType,
    destination: str,
    comfyui_folder: Optional[str] = None,
) -> str:
    """Map a normalized ``dest_type`` + user's ``destination`` to a concrete
    directory path.  Raises ``HTTPException`` when a required path is missing.
    """
    if destination == "comfyui":
        from api.routes.settings import get_comfyui_models_path
        import os as _os
        comfyui_models = get_comfyui_models_path()
        if not comfyui_models:
            raise HTTPException(
                status_code=400,
                detail=(
                    "ComfyUI models path is not configured. "
                    "Set it in Settings -> ComfyUI or via the COMFYUI_MODELS_PATH env var."
                ),
            )
        folder = comfyui_folder or _DEST_COMFYUI_FOLDER.get(dest_type, "checkpoints")
        target_dir = _os.path.join(comfyui_models, folder)
        _os.makedirs(target_dir, exist_ok=True)
        return target_dir

    # "training" destination
    dir_key = _DEST_TRAINING_DIR.get(dest_type, "pretrained_model")
    dir_map = {
        "pretrained_model": model_service.pretrained_model_dir,
        "vae": model_service.vae_dir,
        "output": model_service.lora_dir,
    }
    target = dir_map.get(dir_key, model_service.pretrained_model_dir)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


# Default ComfyUI subfolder per normalized type (spec §4.4 table).
_DEST_COMFYUI_FOLDER: dict[DestType, str] = {
    DestType.CHECKPOINT: "checkpoints",
    DestType.LORA: "loras",
    DestType.VAE: "vae",
    DestType.EMBEDDING: "embeddings",
    DestType.OTHER: "checkpoints",
}

# Training directory key per normalized type.
_DEST_TRAINING_DIR: dict[DestType, str] = {
    DestType.CHECKPOINT: "pretrained_model",
    DestType.LORA: "output",
    DestType.VAE: "vae",
    DestType.EMBEDDING: "pretrained_model",
    DestType.OTHER: "pretrained_model",
}


def _download_type_from_dest(dest_type: DestType) -> ModelType:
    """Best-effort mapping from normalized destination to the legacy ModelType
    enum so the job manager and progress tracking stay consistent.
    """
    if dest_type == DestType.VAE:
        return ModelType.VAE
    if dest_type == DestType.LORA:
        return ModelType.LORA
    return ModelType.MODEL


# ── Background-job runner ──────────────────────────────────────────────────


async def _run_source_download_job(
    config: DownloadConfig,
    download_type: ModelType,
    job,
) -> dict:
    """Coroutine body for a source-adapter download job."""
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


# ── Routes ─────────────────────────────────────────────────────────────────


@router.get("", response_model=list[SourceInfo])
async def list_sources():
    """List all registered model-source adapters."""
    result: list[SourceInfo] = []
    for adapter in list_adapters():
        try:
            classes = await _safe_base_model_classes(adapter)
        except Exception:
            classes = []
        result.append(SourceInfo(
            name=adapter.name,
            credential_kind=adapter.credential_kind,
            base_model_classes=classes,
        ))
    return result


@router.get("/{name}/search")
async def search_source(
    name: str,
    term: str = Query(""),
    sort: Optional[str] = Query(None),
    page: int = Query(1),
    limit: int = Query(20),
    base_model: Optional[str] = Query(None),
    model_type: Optional[str] = Query(None),
):
    """Search a model source.

    Returns a paginated list of model summaries.
    """
    adapter = get_adapter(name)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Unknown source: {name}")
    query = SearchQuery(
        term=term,
        sort=sort,
        page=page,
        limit=limit,
        base_model=base_model,
        model_type=model_type,
    )
    try:
        result = adapter.search(query)
    except Exception as exc:
        logger.exception("Search failed for source '%s': %s", name, exc)
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}")
    return result.model_dump()


@router.get("/{name}/models/{model_id}")
async def get_source_model(name: str, model_id: str):
    """Get full model detail including all versions."""
    adapter = get_adapter(name)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Unknown source: {name}")
    try:
        detail = adapter.get_model(model_id)
    except Exception as exc:
        logger.exception("Failed to get model '%s' from '%s': %s", model_id, name, exc)
        raise HTTPException(status_code=502, detail=f"Failed to get model: {exc}")
    return detail.model_dump()


@router.post("/{name}/download", response_model=JobCreateResponse)
async def download_from_source(name: str, request: SourceDownloadRequest):
    """Download a model version from a source.

    Returns immediately with a ``job_id``; the actual download runs in the
    background. Poll ``GET /api/models/download/status/{job_id}`` for progress.
    """
    adapter = get_adapter(name)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Unknown source: {name}")

    # Resolve version metadata via the adapter.
    try:
        versions = adapter.get_versions(request.model_id)
    except Exception as exc:
        logger.exception("Failed to get versions for '%s': %s", request.model_id, exc)
        raise HTTPException(status_code=502, detail=f"Failed to get versions: {exc}")

    target = next(
        (v for v in versions if v.version_id == request.version_id),
        None,
    )
    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"Version {request.version_id} not found for model {request.model_id}",
        )

    # Only download PUBLISHED versions (spec §6.4).
    if target.status and target.status.upper() != "PUBLISHED":
        raise HTTPException(
            status_code=400,
            detail=f"Version status is '{target.status}' — only PUBLISHED versions can be downloaded",
        )

    # Resolve the concrete download URL + metadata.
    try:
        spec = adapter.resolve_download(target)
    except Exception as exc:
        logger.exception("Failed to resolve download for version '%s': %s", request.version_id, exc)
        raise HTTPException(status_code=502, detail=f"Failed to resolve download: {exc}")

    # Map destination type to a filesystem directory.
    dest_type = spec.dest_type or target.dest_type
    target_dir = _resolve_target_dir(dest_type, request.destination, request.comfyui_folder)
    download_type = _download_type_from_dest(dest_type)

    config = DownloadConfig(
        url=spec.url,
        download_dir=target_dir,
        filename=spec.filename,
        expected_sha256=spec.sha256,
        allowed_redirect_hosts=spec.allowed_redirect_hosts,
        headers=spec.headers,
        model_type=download_type,
    )

    job_id = job_manager.run_coroutine_job(
        JobType.DOWNLOAD,
        lambda job: _run_source_download_job(config, download_type, job),
    )
    return JobCreateResponse(
        job_id=job_id,
        status=JobStatusEnum.RUNNING,
        message=f"Downloading {spec.filename} from {name}",
    )


@router.get("/{name}/models/{model_id}/versions")
async def get_source_versions(name: str, model_id: str):
    """Get all versions for a model."""
    adapter = get_adapter(name)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Unknown source: {name}")
    try:
        versions = adapter.get_versions(model_id)
    except Exception as exc:
        logger.exception("Failed to get versions for '%s': %s", model_id, exc)
        raise HTTPException(status_code=502, detail=f"Failed to get versions: {exc}")
    return [v.model_dump() for v in versions]


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _safe_base_model_classes(adapter) -> list[str]:
    """Call ``base_model_classes`` in a thread to avoid blocking the event loop.

    The Arc adapter makes an HTTP call to ``/models/classes``; run it off the
    event loop so the endpoint stays responsive.
    """
    import asyncio
    return await asyncio.to_thread(adapter.base_model_classes)
