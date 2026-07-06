"""Arc En Ciel source adapter.

Maps Arc's public API (verified live via ``scripts/arc_api_probe.py``, 2026-07-04)
into the source-agnostic models in :mod:`services.sources.base`. v1 pull is
**unauthenticated** — no API key. See the additive-download-system spec §4.2.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from services.sources.base import (
    DestType,
    DownloadSpec,
    ModelDetail,
    ModelSummary,
    ModelVersion,
    SearchQuery,
    SearchResult,
    SourceAdapter,  # noqa: F401  (documents the Protocol this satisfies)
)

ARC_BASE = "https://arcenciel.io/api"
ARC_UPLOADS = "https://arcenciel.io/uploads"
# Hosts a download may be redirected to (passed to the engine's allowlist).
ARC_REDIRECT_HOSTS = ["arcenciel.io", "uploads.arcenciel.io"]
HF_REDIRECT_HOSTS = ["huggingface.co", "hf.co"]
USER_AGENT = "Ktiseos-Nyx-Trainer/arc-adapter (+https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer)"
TIMEOUT = 20

# Arc model type (on the model) -> our normalized destination category.
_TYPE_TO_DEST = {
    "CHECKPOINT": DestType.CHECKPOINT,
    "LORA": DestType.LORA,
    "VAE": DestType.VAE,
    "EMBEDDING": DestType.EMBEDDING,
    "SEGMENTATION": DestType.OTHER,
    "OTHER": DestType.OTHER,
}


def _dest_from_type(type_str: Optional[str]) -> DestType:
    """Map an Arc model ``type`` string to a normalized destination category."""
    if not type_str:
        return DestType.OTHER
    return _TYPE_TO_DEST.get(str(type_str).upper(), DestType.OTHER)


def _dest_from_filepath(file_path: Optional[str]) -> DestType:
    """Best-effort destination from a version's ``filePath`` (e.g. models/Lora/...).

    Used when only version data is available (no model ``type``). Keyword match so
    it survives folder-name variations.
    """
    if not file_path:
        return DestType.OTHER
    lowered = file_path.lower()
    if "lora" in lowered:
        return DestType.LORA
    if "checkpoint" in lowered or "ckpt" in lowered:
        return DestType.CHECKPOINT
    if "vae" in lowered:
        return DestType.VAE
    if "embed" in lowered:
        return DestType.EMBEDDING
    return DestType.OTHER


def _sanitize_filename(name: str) -> str:
    """Reduce a supplied name to a bare, traversal-safe filename."""
    base = os.path.basename(name.replace("\\", "/")).strip()
    base = base.lstrip(".")  # no leading dots / ".." traversal
    return base or "download"


class ArcEnCielAdapter:
    """Adapter for arcenciel.io. Satisfies the :class:`SourceAdapter` Protocol."""

    name = "arcenciel"
    credential_kind = "none"  # v1 pull needs no API key

    # ── HTTP ────────────────────────────────────────────────────────────────
    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        resp = requests.get(
            f"{ARC_BASE}{path}",
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _as_list(data: Any, *keys: str) -> list:
        """Return a list from ``data`` — either data itself, or data[key]."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, list):
                    return value
        return []

    @staticmethod
    def _cover_url(node: dict) -> Optional[str]:
        """Best-effort cover image for a model/version node.

        Arc images are ``{filePath, fileName, variants:[{path,label,...}]}`` with
        uploads-relative paths. Prefer a ``w512`` webp thumbnail (card-sized),
        falling back to any variant, then the full ``filePath``. Resolved against
        ``/uploads``. Probed defensively so an odd shape yields None, not a crash.
        """
        versions = node.get("versions")
        candidates = versions if isinstance(versions, list) else [node]
        for ver in candidates:
            if not isinstance(ver, dict):
                continue
            images = ver.get("images")
            if not (isinstance(images, list) and images and isinstance(images[0], dict)):
                continue
            img = images[0]
            path: Optional[str] = None
            variants = img.get("variants")
            if isinstance(variants, list) and variants:
                by_label = {
                    v.get("label"): v.get("path")
                    for v in variants if isinstance(v, dict) and v.get("path")
                }
                path = by_label.get("w512") or by_label.get("w256") or next(
                    (v.get("path") for v in variants
                     if isinstance(v, dict) and v.get("path")),
                    None,
                )
            path = path or img.get("filePath") or img.get("fileName")
            if path:
                return path if path.startswith("http") else f"{ARC_UPLOADS}/{path.lstrip('/')}"
        return None

    # ── Adapter API ─────────────────────────────────────────────────────────
    def base_model_classes(self) -> list[str]:
        data = self._get("/models/classes")
        classes = self._as_list(data, "classes")
        return [c["name"] for c in classes if isinstance(c, dict) and c.get("name")]

    def search(self, query: SearchQuery) -> SearchResult:
        params: dict[str, Any] = {
            "search": query.term or "",
            "page": query.page,
            "limit": query.limit,
            "status": "available",
            # compact grids omit file metadata — fine for a summary list; we never
            # resolve a download from compact data (spec §4.2).
            "compact": "true",
            "versionLimit": 1,
        }
        if query.sort:
            params["sort"] = query.sort
        if query.base_model:
            params["baseModel"] = query.base_model
        if query.model_type:
            params["modelType"] = query.model_type

        data = self._get("/models/search", params)
        raw = self._as_list(data, "models", "items", "data")
        items = [self._to_summary(m) for m in raw if isinstance(m, dict)]
        if query.nsfw is not None:
            items = [m for m in items if m.nsfw == query.nsfw]
        # Arc's compact search doesn't return a total; a full page implies more.
        has_more = len(items) >= query.limit
        return SearchResult(source=self.name, items=items, page=query.page, has_more=has_more)

    def get_model(self, model_id: str) -> ModelDetail:
        data = self._get(f"/models/{model_id}")
        versions = self.get_versions(model_id)
        # The model's own type is authoritative — stamp it onto every version.
        dest = _dest_from_type(data.get("type"))
        for ver in versions:
            ver.dest_type = dest
        return ModelDetail(
            source=self.name,
            model_id=str(data.get("id", model_id)),
            title=data.get("title") or "",
            type=data.get("type"),
            base_model=versions[0].base_model if versions else None,
            # The /models/{id} node's versions may be trimmed; use the fully
            # fetched versions, whose covers were resolved from real image data.
            cover_url=next((v.cover_url for v in versions if v.cover_url), None),
            nsfw=bool(data.get("nsfw")),
            uploader=(
                data["uploader"]["username"]
                if isinstance(data.get("uploader"), dict)
                else data.get("uploader")
            ),
            description=data.get("description"),
            versions=versions,
        )

    def get_versions(self, model_id: str) -> list[ModelVersion]:
        data = self._get(f"/models/{model_id}/versions")
        raw = self._as_list(data, "versions")
        return [self._to_version(model_id, v) for v in raw if isinstance(v, dict)]

    def resolve_download(
        self, version: ModelVersion, credential: Optional[str] = None
    ) -> DownloadSpec:
        filename = _sanitize_filename(
            version.original_name or version.file_name or f"version-{version.version_id}"
        )
        allowed = list(ARC_REDIRECT_HOSTS)

        url = self._canonical_download(version)
        # Prefer an external URL only when its host is explicitly allowlisted.
        ext = version.external_download_url
        if ext:
            host = (urlparse(ext).hostname or "").lower()
            if any(host == h or host.endswith("." + h) for h in HF_REDIRECT_HOSTS):
                url = ext
                allowed = allowed + HF_REDIRECT_HOSTS

        size = int(version.file_size_kb * 1024) if version.file_size_kb else None
        return DownloadSpec(
            url=url,
            filename=filename,
            sha256=version.sha256,
            size=size,
            allowed_redirect_hosts=allowed,
            dest_type=version.dest_type,
        )

    # ── Mapping helpers ──────────────────────────────────────────────────────
    def _to_summary(self, m: dict) -> ModelSummary:
        versions = m.get("versions")
        base_model = None
        if isinstance(versions, list) and versions and isinstance(versions[0], dict):
            base_model = versions[0].get("baseModel")
        return ModelSummary(
            source=self.name,
            model_id=str(m.get("id")),
            title=m.get("title") or "",
            type=m.get("type"),
            base_model=base_model,
            cover_url=self._cover_url(m),
            nsfw=bool(m.get("nsfw")),
            uploader=(
                m["uploader"]["username"]
                if isinstance(m.get("uploader"), dict)
                else m.get("uploader")
            ),
        )

    def _to_version(self, model_id: str, v: dict) -> ModelVersion:
        return ModelVersion(
            source=self.name,
            model_id=str(model_id),
            version_id=str(v.get("id")),
            name=v.get("versionName"),
            base_model=v.get("baseModel"),
            model_type=v.get("modelType"),
            status=v.get("status"),
            file_name=v.get("fileName"),
            original_name=v.get("originalName"),
            file_size_kb=v.get("fileSizeKb"),
            sha256=v.get("sha256"),
            sha256_webui=v.get("sha256webui"),
            external_download_url=v.get("externalDownloadUrl"),
            file_scan_status=v.get("fileScanStatus"),
            cover_url=self._cover_url(v),
            dest_type=_dest_from_filepath(v.get("filePath")),
        )

    def _canonical_download(self, version: ModelVersion) -> str:
        return f"{ARC_BASE}/models/{version.model_id}/versions/{version.version_id}/download"
