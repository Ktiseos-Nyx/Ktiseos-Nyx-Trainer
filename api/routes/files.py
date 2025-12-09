"""
File Management API Routes
Handles file browsing, upload, download, delete operations.
Replaces Jupyter Lab file manager!
"""
import logging
import mimetypes
import shutil
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Project root detection
# Find where the API code is running (works anywhere the repo is cloned)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()  # api/routes/files.py -> project root

# Default workspace is the project root
DEFAULT_WORKSPACE = PROJECT_ROOT

# Allowed directories for security (project root + user home)
ALLOWED_DIRS = [
    PROJECT_ROOT,
    Path.home()
]


class FileInfo(BaseModel):
    """File or directory information"""
    name: str
    path: str
    type: str  # "file" or "dir"
    size: int
    modified: float
    is_image: bool = False
    mime_type: Optional[str] = None


class DirectoryListing(BaseModel):
    """Directory contents"""
    path: str
    parent: Optional[str]
    files: List[FileInfo]


def is_safe_path(base_path: Path, user_path: str) -> bool:
    """Check if path is within allowed directories"""
    try:
        full_path = (base_path / user_path).resolve()
        return any(
            str(full_path).startswith(str(allowed_dir.resolve()))
            for allowed_dir in ALLOWED_DIRS
        )
    except Exception:
        return False


def get_file_info(path: Path) -> FileInfo:
    """Get file information"""
    stat = path.stat()
    mime_type, _ = mimetypes.guess_type(str(path))

    return FileInfo(
        name=path.name,
        path=str(path),
        type="dir" if path.is_dir() else "file",
        size=stat.st_size if path.is_file() else 0,
        modified=stat.st_mtime,
        is_image=mime_type and mime_type.startswith("image/") if mime_type else False,
        mime_type=mime_type
    )


@router.get("/default-workspace")
async def get_default_workspace():
    """
    Get the default workspace path for the current environment.
    Returns /workspace on VastAI/cloud, or home directory for local.
    """
    return {
        "path": str(DEFAULT_WORKSPACE),
        "allowed_dirs": [str(d) for d in ALLOWED_DIRS]
    }


@router.get("/list", response_model=DirectoryListing)
async def list_directory(path: str = Query(default=None)):
    """
    List files and directories at the specified path.
    Returns sorted list with directories first.
    """
    try:
        # Use default workspace if no path provided
        if path is None:
            path = str(DEFAULT_WORKSPACE)

        target_path = Path(path).resolve()

        # Security check
        if not any(str(target_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        if not target_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        # Get parent directory
        parent = str(target_path.parent) if target_path.parent != target_path else None

        # List contents
        files = []
        for item in target_path.iterdir():
            try:
                files.append(get_file_info(item))
            except Exception as e:
                logger.warning(f"Failed to get info for {item}: {e}")

        # Sort: directories first, then by name
        files.sort(key=lambda x: (x.type == "file", x.name.lower()))

        return DirectoryListing(
            path=str(target_path),
            parent=parent,
            files=files
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    destination: str = Query(default=None)
):
    """
    Upload a file to the specified destination.
    Supports large files with streaming.
    """
    try:
        # Use default workspace if no destination provided
        if destination is None:
            destination = str(DEFAULT_WORKSPACE)

        dest_path = Path(destination).resolve()

        # Security check
        if not any(str(dest_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        # Create destination directory if needed
        dest_path.mkdir(parents=True, exist_ok=True)

        # Full file path
        file_path = dest_path / file.filename

        # Check if file exists
        if file_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} already exists"
            )

        # Stream upload (handles large files)
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)

        logger.info(f"Uploaded file: {file_path}")

        return {
            "success": True,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image/{path:path}")
async def serve_image(path: str):
    """Serve an image file (for thumbnails/previews)"""
    try:
        # Construct path relative to datasets directory
        file_path = (PROJECT_ROOT / "datasets" / path).resolve()

        # Security check
        if not any(str(file_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Get proper MIME type for images
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Not an image file")

        return FileResponse(
            path=file_path,
            media_type=mime_type
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{path:path}")
async def download_file(path: str):
    """Download a file"""
    try:
        file_path = Path("/" + path).resolve()

        # Security check
        if not any(str(file_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        return FileResponse(
            path=file_path,
            filename=file_path.name,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def delete_file(path: str = Query(...)):
    """Delete a file or directory"""
    try:
        target_path = Path(path).resolve()

        # Security check
        if not any(str(target_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        # Delete
        if target_path.is_file():
            target_path.unlink()
            logger.info(f"Deleted file: {target_path}")
        elif target_path.is_dir():
            shutil.rmtree(target_path)
            logger.info(f"Deleted directory: {target_path}")

        return {"success": True, "message": f"Deleted {target_path.name}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rename")
async def rename_file(old_path: str, new_name: str):
    """Rename a file or directory"""
    try:
        old = Path(old_path).resolve()
        new = old.parent / new_name

        # Security checks
        if not any(str(old).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        if not old.exists():
            raise HTTPException(status_code=404, detail="Path not found")

        if new.exists():
            raise HTTPException(status_code=400, detail="Target name already exists")

        # Rename
        old.rename(new)
        logger.info(f"Renamed {old} to {new}")

        return {
            "success": True,
            "old_path": str(old),
            "new_path": str(new)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rename: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mkdir")
async def create_directory(path: str, name: str):
    """Create a new directory"""
    try:
        parent_path = Path(path).resolve()
        new_dir = parent_path / name

        # Security check
        if not any(str(parent_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        if new_dir.exists():
            raise HTTPException(status_code=400, detail="Directory already exists")

        # Create directory
        new_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Created directory: {new_dir}")

        return {
            "success": True,
            "path": str(new_dir)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/read/{path:path}")
async def read_file(path: str):
    """Read text file contents (for editor)"""
    try:
        file_path = Path("/" + path).resolve()

        # Security check
        if not any(str(file_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")

        # Read file
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        return {
            "path": str(file_path),
            "content": content,
            "size": file_path.stat().st_size
        }

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a text file")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/write")
async def write_file(path: str, content: str):
    """Write text file contents (for editor)"""
    try:
        file_path = Path(path).resolve()

        # Security check
        if not any(str(file_path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        logger.info(f"Wrote file: {file_path}")

        return {
            "success": True,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to write file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
