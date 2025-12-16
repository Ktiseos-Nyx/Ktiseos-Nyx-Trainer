"""
API routes for user settings management.
"""

import json
import os
import shutil
import logging

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Settings file location
SETTINGS_DIR = os.path.join(os.getcwd(), "config")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "user_settings.json")


class UserSettings(BaseModel):
    huggingface_token: Optional[str] = None
    civitai_api_key: Optional[str] = None


def ensure_settings_dir():
    """
    Ensures the settings directory exists AND is writable.
    Only attempts to fix permissions if they are actually broken.
    This is safe for both local machines and servers.
    """
    try:
        # Step 1: Always ensure the directory exists.
        os.makedirs(SETTINGS_DIR, exist_ok=True)

        # Step 2: Check if the directory is writable by the current user.
        if not os.access(SETTINGS_DIR, os.W_OK):
            logger.warning(
                "Configuration directory '%s' is not writable. "
                "This can happen in Docker/server environments. "
                "Attempting to auto-fix permissions...",
                SETTINGS_DIR
            )
            # Step 3: If not writable, TRY to fix it.
            try:
                # 0o777 gives read/write/execute permissions to everyone.
                os.chmod(SETTINGS_DIR, 0o777)
                logger.info("✅ Successfully fixed directory permissions.")

                # Also fix the settings file itself if it exists
                if os.path.exists(SETTINGS_FILE):
                    os.chmod(SETTINGS_FILE, 0o666)  # Read/Write for everyone

            except PermissionError:
                logger.error(
                    "❌ Failed to fix permissions. The current user does not own the directory "
                    "or lacks privileges to change permissions. Please fix manually."
                )
    except Exception as e:
        logger.error(
            "An unexpected error occurred while ensuring "
            "settings directory: %s",
            e,
            exc_info=True
        )


def load_settings() -> dict:
    """Load settings from JSON file."""
    ensure_settings_dir()

    if not os.path.exists(SETTINGS_FILE):
        return {}

    try:
        # THE FIX IS HERE:
        with open(SETTINGS_FILE, 'r', encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # Use the logger to be consistent
        logger.error(
            "Error loading settings from %s: %s",
            SETTINGS_FILE,
            e,
            exc_info=True
        )
        return {}


def save_settings(settings: dict) -> bool:
    """Save settings to JSON file."""
    ensure_settings_dir()

    try:
        # Now, this 'settings' variable is correctly defined from the parameter
        with open(SETTINGS_FILE, 'w', encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.error(
            "Error saving settings to %s: %s",
            SETTINGS_FILE,
            e,
            exc_info=True
        )
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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get storage info: {str(e)}"
        ) from e
