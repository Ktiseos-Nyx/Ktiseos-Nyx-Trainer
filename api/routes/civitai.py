"""
Civitai API proxy endpoints for browsing and downloading models.

Attribution: Inspired by sd-webui-civbrowser extension
https://github.com/SignalFlagZ/sd-webui-civbrowser
"""

import aiohttp
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel

router = APIRouter()

CIVITAI_API_BASE = "https://civitai.com/api/v1"


def get_civitai_headers():
    """Get headers for Civitai API requests, including API key if available."""
    from api.routes.settings import get_api_keys

    headers = {
        "Content-Type": "application/json",
    }

    # Add API key if available
    api_keys = get_api_keys()
    if api_keys.get("civitai_api_key"):
        headers["Authorization"] = f"Bearer {api_keys['civitai_api_key']}"

    return headers


@router.get("/models")
async def browse_models(
    limit: int = Query(default=20, ge=1, le=100, description="Number of models to return"),
    page: Optional[int] = Query(default=None, ge=1, description="Page number (for non-search queries)"),
    cursor: Optional[str] = Query(default=None, description="Cursor for pagination (required for search queries)"),
    query: Optional[str] = Query(default=None, description="Search query by model name"),
    tag: Optional[str] = Query(default=None, description="Search by tag"),
    username: Optional[str] = Query(default=None, description="Search by creator username"),
    types: Optional[str] = Query(default=None, description="Model types (Checkpoint, LORA, VAE, etc)"),
    baseModel: Optional[str] = Query(default=None, description="Base model (SD 1.5, SDXL, Pony, Flux, etc)"),
    sort: str = Query(default="Highest Rated", description="Sort order"),
    period: str = Query(default="AllTime", description="Time period for stats"),
    nsfw: bool = Query(default=False, description="Include NSFW content"),
):
    """
    Browse Civitai models with filters.

    Proxy endpoint for Civitai API /v1/models endpoint.

    Note: Civitai API requires cursor-based pagination when using search queries.
    Use 'page' for browsing without search, 'cursor' when searching.
    """
    try:
        params = {
            "limit": limit,
            "sort": sort,
            "period": period,
        }

        # Use cursor-based pagination for search queries, page-based otherwise
        # Search can be by query (name), tag, or username
        has_search = query or tag or username

        if has_search:
            if query:
                params["query"] = query
            if tag:
                params["tag"] = tag
            if username:
                params["username"] = username
            if cursor:
                params["cursor"] = cursor
        else:
            # Default to page 1 if not specified and no cursor
            params["page"] = page if page is not None else 1

        if types:
            params["types"] = types
        if baseModel:
            params["baseModel"] = baseModel
        if not nsfw:
            params["nsfw"] = "false"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CIVITAI_API_BASE}/models",
                params=params,
                headers=get_civitai_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "data": data
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Civitai API error: {error_text}"
                    )

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Civitai: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def get_model_details(model_id: int):
    """
    Get detailed information about a specific Civitai model.

    Proxy endpoint for Civitai API /v1/models/{id} endpoint.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CIVITAI_API_BASE}/models/{model_id}",
                headers=get_civitai_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "data": data
                    }
                elif response.status == 404:
                    raise HTTPException(status_code=404, detail="Model not found")
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Civitai API error: {error_text}"
                    )

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Civitai: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags")
async def get_tags(
    limit: int = Query(default=20, ge=1, le=200, description="Number of tags to return"),
    query: Optional[str] = Query(default=None, description="Search query for tags"),
):
    """
    Get available Civitai tags for filtering.

    Proxy endpoint for Civitai API /v1/tags endpoint.
    """
    try:
        params = {
            "limit": limit,
        }

        if query:
            params["query"] = query

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CIVITAI_API_BASE}/tags",
                params=params,
                headers=get_civitai_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "data": data
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Civitai API error: {error_text}"
                    )

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Civitai: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-versions/{version_id}")
async def get_model_version(version_id: int):
    """
    Get information about a specific model version.

    Proxy endpoint for Civitai API /v1/model-versions/{id} endpoint.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CIVITAI_API_BASE}/model-versions/{version_id}",
                headers=get_civitai_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "data": data
                    }
                elif response.status == 404:
                    raise HTTPException(status_code=404, detail="Model version not found")
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Civitai API error: {error_text}"
                    )

    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to Civitai: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DownloadModelRequest(BaseModel):
    model_id: int
    version_id: int
    download_url: str
    filename: str
    model_type: str = "model"  # "model" or "vae"


@router.post("/download")
async def download_civitai_model(request: DownloadModelRequest):
    """
    Download a model from Civitai using the existing download infrastructure.

    This endpoint constructs the proper Civitai download URL with API key
    and passes it to the existing model download system.
    """
    try:
        from api.routes.settings import get_api_keys

        # Get API key for authenticated download
        api_keys = get_api_keys()
        civitai_key = api_keys.get("civitai_api_key", "")

        # Construct download URL with API key if available
        download_url = request.download_url
        if civitai_key and "?" not in download_url:
            download_url = f"{download_url}?token={civitai_key}"
        elif civitai_key:
            download_url = f"{download_url}&token={civitai_key}"

        # Import models API to reuse download logic
        from api.routes.models import get_model_manager

        manager = get_model_manager()

        # Determine target directory
        if request.model_type == "vae":
            target_dir = manager.vae_dir
        else:
            target_dir = manager.pretrained_model_dir

        # Use existing download infrastructure
        result = manager.download_model_or_vae(
            url=download_url,
            download_dir=target_dir
        )

        if result.get("success"):
            return {
                "success": True,
                "message": f"Downloaded {request.filename}",
                "file_path": result.get("file_path"),
                "model_id": request.model_id,
                "version_id": request.version_id,
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Download failed"))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
