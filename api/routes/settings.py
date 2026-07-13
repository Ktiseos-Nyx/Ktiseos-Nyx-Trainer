"""
API routes for user settings management.
"""

import gc
import json
import logging
import os
import shutil
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Settings file location — anchored on the project root (file-relative), NOT os.getcwd().
# The Next.js settings UI writes user_settings.json to <project_root>/user_config (anchored via
# process.cwd()/.. in settings-service.ts). Reading it from os.getcwd()/user_config here meant
# that when the FastAPI process's working dir wasn't the project root, the UI's saved settings
# (incl. comfyui_models_path) landed in a DIFFERENT file than this resolver read — so the merge
# tools silently fell back / saw stale config. Same getcwd class of bug as a94854f.
# parents: settings.py -> routes -> api -> <project_root>.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_DIR = os.path.join(_PROJECT_ROOT, "user_config")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "user_settings.json")


class UserSettings(BaseModel):
    huggingface_token: Optional[str] = None
    civitai_api_key: Optional[str] = None
    extra_model_dirs: Optional[List[str]] = None
    extra_vae_dirs: Optional[List[str]] = None


def _fix_settings_permissions() -> bool:
    """
    Attempt to repair settings directory/file permissions in Docker/server
    environments where the directory was created by a different user (e.g. root).

    Skipped entirely on Windows — os.chmod() is a no-op for Unix mode bits there,
    and Windows permissions are managed via ACLs in Explorer/icacls instead.

    Returns True if the directory is writable after the attempt.
    """
    if os.name == "nt":
        # Windows: chmod does nothing useful here; caller logs guidance.
        return False

    try:
        os.chmod(SETTINGS_DIR, 0o700)   # owner-only rwx — no world/group write
        if os.path.exists(SETTINGS_FILE):
            os.chmod(SETTINGS_FILE, 0o600)  # owner-only rw
        return os.access(SETTINGS_DIR, os.W_OK)
    except PermissionError:
        return False


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
            if _fix_settings_permissions():
                logger.info("✅ Successfully fixed directory permissions.")
            else:
                logger.error(
                    "❌ Could not make '%s' writable. "
                    "On Windows: right-click the folder → Properties → Security tab "
                    "and grant your user Write permission. "
                    "On Linux/Docker: run 'chown -R $USER <project_dir>'.",
                    SETTINGS_DIR,
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


def get_comfyui_models_path() -> str:
    """
    Return the filesystem path to ComfyUI's models directory.

    Resolution order — the trainer's OWN bundled ComfyUI wins first, so a stale env
    var or saved setting can never shadow the models the installer actually put on
    disk:
      1. {project_root}/ComfyUI/models — used whenever it exists on disk. The installer
         clones ComfyUI directly inside the project root, so on a normal box this IS the
         answer, and it is authoritative.
      2. COMFYUI_MODELS_PATH environment variable (if it is a real directory) — fallback
         for setups that have no bundled ComfyUI and run one elsewhere.
      3. comfyui_models_path field in user_settings.json (if it is a real directory) —
         same escape hatch, from the settings UI.
      4. {project_root}/ComfyUI/models — bare anchored string when nothing above exists
         on disk; callers still validate that the directory is present.

    Why bundled-first: the previous order consulted the env/setting BEFORE the anchored
    path. A stale value that pointed at an existing-but-wrong directory (e.g. a leftover
    "/workspace/ComfyUI/models" from an old template) passed the is_dir() guard and got
    returned, so _comfyui_model_dirs() scanned a foreign/empty tree — the bug where the
    merge/bake pickers listed only pretrained_model/ even though checkpoints and
    diffusion_models sat right there in the trainer's ComfyUI. Anchoring first is
    cwd-independent (services.core.validation.PROJECT_ROOT is Path(__file__)-based), so
    it's stable on Windows / VastAI / RunPod.
    """
    import os as _os
    from services.core.validation import PROJECT_ROOT

    anchored = str((PROJECT_ROOT / "ComfyUI" / "models").resolve())
    # The trainer manages its own ComfyUI here; when present it is authoritative and a
    # stale/wrong override must not win over the models the user actually has.
    if _os.path.isdir(anchored):
        return anchored

    env_path = _os.environ.get("COMFYUI_MODELS_PATH", "")
    if env_path and _os.path.isdir(env_path):
        return env_path

    settings = load_settings()
    settings_path = settings.get("comfyui_models_path", "")
    if settings_path and _os.path.isdir(settings_path):
        return settings_path

    return anchored


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
                "has_civitai_api_key": bool(settings.get("civitai_api_key")),
                "extra_model_dirs": settings.get("extra_model_dirs", []),
                "extra_vae_dirs": settings.get("extra_vae_dirs", []),
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

        if settings.extra_model_dirs is not None:
            current_settings["extra_model_dirs"] = settings.extra_model_dirs

        if settings.extra_vae_dirs is not None:
            current_settings["extra_vae_dirs"] = settings.extra_vae_dirs

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
        valid_keys = ["huggingface_token", "civitai_api_key", "extra_model_dirs", "extra_vae_dirs"]
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


def _dir_size(path: str) -> int:
    """Recursively compute the total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += _dir_size(entry.path)
            except OSError:
                continue
    except OSError:
        pass
    return total


@router.get("/cache/info")
async def get_cache_info():
    """Report approximate cache sizes and locations so the user can decide whether to clear."""
    import gc as _gc

    info = {
        "success": True,
        "caches": {},
        "python_gc": {"enabled": _gc.isenabled(), "tracked_objects": len(_gc.get_objects())},
    }

    # HuggingFace hub cache
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hf_hub = os.path.join(hf_home, "hub")
    if os.path.isdir(hf_hub):
        size_bytes = _dir_size(hf_hub)
        info["caches"]["huggingface_hub"] = {
            "path": hf_hub,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "size_gb": round(size_bytes / (1024 ** 3), 2),
        }
    else:
        info["caches"]["huggingface_hub"] = {"path": hf_hub, "exists": False}

    # huggingface_hub download cache (hf_hub_download local cache)
    hf_cache = os.path.join(hf_home, ".locks") if os.path.isdir(os.path.join(hf_home, ".locks")) else None

    # Torch hub cache
    torch_home = os.environ.get("TORCH_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "torch")
    if os.path.isdir(torch_home):
        size_bytes = _dir_size(torch_home)
        info["caches"]["torch_hub"] = {
            "path": torch_home,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "size_gb": round(size_bytes / (1024 ** 3), 2),
        }
    else:
        info["caches"]["torch_hub"] = {"path": torch_home, "exists": False}

    # GPU memory info (if available)
    try:
        import torch
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info["cuda"] = {
                "device_count": torch.cuda.device_count(),
                "devices": [],
            }
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_mem
                mem_allocated = torch.cuda.memory_allocated(i)
                mem_reserved = torch.cuda.memory_reserved(i)
                gpu_info["cuda"]["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_mb": round(mem_total / (1024 * 1024), 0),
                    "allocated_mb": round(mem_allocated / (1024 * 1024), 2),
                    "reserved_mb": round(mem_reserved / (1024 * 1024), 2),
                })
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            gpu_info["xpu"] = {"device_count": torch.xpu.device_count()}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info["mps"] = {"available": True}
        if gpu_info:
            info["gpu"] = gpu_info
    except (ImportError, Exception):
        info["gpu"] = {"available": False}

    return info


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear in-process memory caches (Python GC, PyTorch CUDA/XPU/MPS cache).

    This frees memory that Python and PyTorch are holding but not actively
    using — "cached" memory that the allocator hasn't returned to the OS.
    Does NOT delete any files from disk.
    """
    result = {
        "success": True,
        "actions": [],
        "note": "On-disk caches (HuggingFace hub, torch hub) are NOT deleted. "
                "Use external tools to clear those if needed.",
    }

    before_objects = len(gc.get_objects())
    collected = gc.collect()
    after_objects = len(gc.get_objects())
    result["actions"].append({
        "type": "python_gc",
        "collected": collected,
        "objects_before": before_objects,
        "objects_after": after_objects,
    })

    try:
        import torch

        cuda_freed = 0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                reserved_before = torch.cuda.memory_reserved(i)
                torch.cuda.empty_cache()
                reserved_after = torch.cuda.memory_reserved(i)
                freed = reserved_before - reserved_after
                cuda_freed += freed
            result["actions"].append({
                "type": "cuda_empty_cache",
                "freed_bytes": cuda_freed,
                "freed_mb": round(cuda_freed / (1024 * 1024), 2),
            })

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                torch.xpu.empty_cache()
                result["actions"].append({"type": "xpu_empty_cache"})
            except Exception:
                pass

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                result["actions"].append({"type": "mps_empty_cache"})
            except Exception:
                pass
    except ImportError:
        result["actions"].append({"type": "torch", "note": "PyTorch not available — skipped GPU cache clear"})
    except Exception as e:
        logger.warning("GPU cache clear error: %s", e)
        result["actions"].append({"type": "torch_error", "error": str(e)})

    if collected == 0 and cuda_freed == 0:
        result["message"] = "No memory to free — caches were already clear."
    else:
        result["message"] = f"Cleared {collected} Python objects"
        if cuda_freed > 0:
            result["message"] += f" and {round(cuda_freed / (1024 * 1024), 2)} MB GPU memory."

    return result
