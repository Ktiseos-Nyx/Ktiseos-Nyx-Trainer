"""Source-adapter framework for the additive download system.

A *source adapter* knows how to browse, describe, and resolve a download URL for
one model site (Arc En Ciel, Civitai, …). It does **not** perform the download
itself — it produces a :class:`DownloadSpec` that the shared, hardened download
engine (``services.model_service``) consumes.

These models are source-agnostic; each adapter maps its site's JSON into them.
See ``docs/specs/2026-07-03-additive-download-system.md`` for the design.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class DestType(str, Enum):
    """Normalized destination category, used to route a file to a directory.

    Maps to concrete folders in the download route (spec §4.4); e.g. ``lora`` ->
    ``output/`` (training) or ``ComfyUI/models/loras/`` (generation).
    """
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    EMBEDDING = "embedding"
    OTHER = "other"


class SearchQuery(BaseModel):
    """A source-agnostic search request."""
    term: str = ""
    sort: Optional[str] = None
    page: int = 1
    limit: int = 20
    base_model: Optional[str] = None
    model_type: Optional[str] = None


class ModelSummary(BaseModel):
    """Lightweight model card for search-result grids."""
    source: str
    model_id: str
    title: str
    type: Optional[str] = None
    base_model: Optional[str] = None
    cover_url: Optional[str] = None
    nsfw: bool = False


class ModelVersion(BaseModel):
    """A downloadable version of a model, with the metadata needed to resolve it.

    Arc En Ciel serves one file per version, so the file fields live directly on
    the version rather than in a ``files[]`` array. ``resolve_download`` uses
    ``model_id`` + ``version_id`` (canonical URL) or ``external_download_url``.
    """
    source: str
    model_id: str
    version_id: str
    name: Optional[str] = None
    base_model: Optional[str] = None
    model_type: Optional[str] = None
    status: Optional[str] = None
    file_name: Optional[str] = None
    original_name: Optional[str] = None
    file_size_kb: Optional[float] = None
    sha256: Optional[str] = None
    sha256_webui: Optional[str] = None
    external_download_url: Optional[str] = None
    file_scan_status: Optional[str] = None
    cover_url: Optional[str] = None
    dest_type: DestType = DestType.OTHER


class ModelDetail(ModelSummary):
    """Full model detail: a summary plus its description and versions."""
    description: Optional[str] = None
    versions: list[ModelVersion] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A page of search results."""
    source: str
    items: list[ModelSummary] = Field(default_factory=list)
    page: int = 1
    has_more: bool = False


class DownloadSpec(BaseModel):
    """Everything the download engine needs to fetch one file safely.

    Maps directly onto ``services.models.model_download.DownloadConfig`` in the
    download route (spec §4.3).
    """
    url: str
    filename: str
    sha256: Optional[str] = None
    size: Optional[int] = None
    headers: Optional[dict[str, str]] = None
    allowed_redirect_hosts: list[str] = Field(default_factory=list)
    dest_type: DestType = DestType.OTHER


@runtime_checkable
class SourceAdapter(Protocol):
    """Interface every model source implements.

    ``credential_kind`` is one of ``"none"`` (Arc v1 pull), ``"api_key"``, or
    ``"link_key"`` (Arc Link, later) — it tells the UI whether/what credential to
    ask for.
    """

    name: str
    credential_kind: str

    def search(self, query: SearchQuery) -> SearchResult: ...

    def get_model(self, model_id: str) -> ModelDetail: ...

    def get_versions(self, model_id: str) -> list[ModelVersion]: ...

    def resolve_download(
        self, version: ModelVersion, credential: Optional[str] = None
    ) -> DownloadSpec: ...

    def base_model_classes(self) -> list[str]: ...
