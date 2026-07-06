"""LM-compatible .metadata.json sidecar writer.

Writes a sidecar file alongside downloaded safetensors so that ComfyUI
LoRA Manager (willmiao/ComfyUI-Lora-Manager) can pick up metadata even
for models sourced from Arc En Ciel (which LM doesn't natively support).
"""

from __future__ import annotations

import json
import os
import time


def write_metadata_json(file_path: str, metadata: dict) -> str:
    """Write a .metadata.json sidecar alongside ``file_path``.

    Args:
        file_path: Absolute path to the downloaded model file.
        metadata: Dict of metadata fields to write (see schema below).

    Returns:
        The path to the written sidecar file.
    """
    stem, _ = os.path.splitext(file_path)
    sidecar_path = f"{stem}.metadata.json"

    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return sidecar_path


def build_metadata(
    *,
    file_name: str,
    file_path: str,
    size: int,
    sha256: str | None,
    model_name: str,
    base_model: str | None,
    model_type: str | None,
    cover_url: str | None,
    uploader: str | None,
    description: str | None,
    tags: list[str] | None,
    activation_tags: list[str] | None,
) -> dict:
    """Build a metadata dict in LM's schema format.

    Result is ready for :func:`write_metadata_json`.
    """
    meta: dict = {
        "file_name": file_name,
        "model_name": model_name,
        "file_path": file_path,
        "size": size,
        "modified": time.time(),
        "base_model": base_model or "Unknown",
        "preview_url": cover_url or "",
        "preview_nsfw_level": 0,
        "notes": "",
        "from_civitai": False,
        "tags": sorted(tags or []),
        "modelDescription": description or "",
        "civitai_deleted": False,
        "favorite": False,
        "exclude": False,
        "db_checked": False,
        "skip_metadata_refresh": False,
        "metadata_source": None,
        "last_checked_at": 0,
        "hash_status": "completed" if sha256 else "pending",
        "schema_version": "arc-v1",
    }

    if sha256:
        meta["sha256"] = sha256

    civitai_section: dict = {}

    if uploader:
        civitai_section["creator"] = {"username": uploader}

    if activation_tags:
        civitai_section["trainedWords"] = activation_tags

    if model_type:
        civitai_section["type"] = model_type

    meta["civitai"] = civitai_section

    return meta
