"""
API routes for user settings management.
"""

import json
import os
import shutil
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Settings file location
SETTINGS_DIR = os.path.join(os.getcwd(), "config")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "user_settings.json")


class UserSettings(BaseModel):
    huggingface_token: Optional[str] = None
    civitai_api_key: Optional[str] = None


def ensure_settings_dir():
    """Ensure settings directory exists."""
    os.makedirs(SETTINGS_DIR, exist_ok=True)


def load_settings() -> dict:
    """Load settings from JSON file."""
    ensure_settings_dir()

    if not os.path.exists(SETTINGS_FILE):
        return {}

    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}


def save_settings(settings: dict) -> bool:
    """Save settings to JSON file."""
    ensure_settings_dir()

    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving settings to {SETTINGS_FILE}: {e}", exc_info=True)
        return False


def get_api_keys() -> dict:
    """
    Get API keys for use in other modules.
    Returns full unmasked keys.
    """
    settings = load_settings()
    return {
        "huggingface_token": settings.get("huggingface_token"),
        "civitai_api_key": settings.get("civitai_api_key")
    }


def mask_token(token: Optional[str]) -> Optional[str]:
    """Mask API token for display (show first 4 and last 4 chars)."""
    if not token or len(token) < 12:
        return None
    return f"{token[:4]}...{token[-4:]}"


@router.get("/user")
async def get_user_settings():
    """
    Get user settings with masked API keys.
    """
    try:
        settings = load_settings()

        return {
            "success": True,
            "settings": {
                "huggingface_token": mask_token(settings.get("huggingface_token")),
                "civitai_api_key": mask_token(settings.get("civitai_api_key")),
                "has_huggingface_token": bool(settings.get("huggingface_token")),
                "has_civitai_api_key": bool(settings.get("civitai_api_key"))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user")
async def update_user_settings(settings: UserSettings):
    """
    Update user settings.
    Only updates provided fields, preserves others.
    """
    try:
        current_settings = load_settings()

        # Update only provided fields
        if settings.huggingface_token is not None:
            current_settings["huggingface_token"] = settings.huggingface_token

        if settings.civitai_api_key is not None:
            current_settings["civitai_api_key"] = settings.civitai_api_key

        if save_settings(current_settings):
            return {
                "success": True,
                "message": "Settings updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save settings")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/user")
async def clear_user_settings():
    """
    Clear all user settings.
    """
    try:
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)

        return {
            "success": True,
            "message": "Settings cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/user/{key}")
async def delete_setting_key(key: str):
    """
    Delete a specific setting key.

    Args:
        key: Either "huggingface_token" or "civitai_api_key"
    """
    try:
        valid_keys = ["huggingface_token", "civitai_api_key"]
        if key not in valid_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid key. Must be one of: {', '.join(valid_keys)}"
            )

        current_settings = load_settings()

        if key in current_settings:
            del current_settings[key]
            save_settings(current_settings)

        return {
            "success": True,
            "message": f"Removed {key}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/storage")
async def get_storage_info():
    """
    Get disk storage information for the current working directory.
    Returns total, used, and available storage in bytes and GB.
    """
    try:
        # Get disk usage for the current working directory
        cwd = os.getcwd()
        usage = shutil.disk_usage(cwd)

        # Convert bytes to GB for readability
        gb = 1024 ** 3

        return {
            "success": True,
            "storage": {
                "path": cwd,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "total_gb": round(usage.total / gb, 2),
                "used_gb": round(usage.used / gb, 2),
                "free_gb": round(usage.free / gb, 2),
                "used_percent": round((usage.used / usage.total) * 100, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage info: {str(e)}")
