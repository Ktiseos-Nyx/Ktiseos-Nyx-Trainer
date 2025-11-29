"""
Model download service for HuggingFace and Civitai.

Handles downloading models, VAEs, and LoRAs with multiple fallback methods:
- aria2c (fastest, multi-connection)
- hf-transfer (HuggingFace specific, fast)
- wget (reliable fallback)
- requests (Python fallback)
"""

import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import datetime
import glob

from services.models.model_download import (
    DownloadConfig,
    DownloadResponse,
    DownloadMethod,
    DownloadSource,
    ModelInfo,
    ListModelsResponse,
    DeleteModelResponse,
    ModelType,
)
from services.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for downloading and managing models, VAEs, and LoRAs.

    Supports multiple download methods with automatic fallback:
    1. aria2c - Fastest, multi-connection (preferred)
    2. hf-transfer - HuggingFace optimized
    3. wget - Reliable fallback
    4. requests - Python fallback (slowest)
    """

    def __init__(self):
        self.project_root = Path.cwd()
        self.pretrained_model_dir = self.project_root / "pretrained_model"
        self.vae_dir = self.project_root / "vae"
        self.lora_dir = self.project_root / "output"  # LoRAs stored in output

        # Ensure directories exist
        self.pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        self.vae_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)

    async def download_model_or_vae(self, config: DownloadConfig) -> DownloadResponse:
        """
        Download a model or VAE with automatic fallback chain.

        Args:
            config: Download configuration

        Returns:
            DownloadResponse with download result
        """
        try:
            # Validate and normalize URL
            validated_url = self._validate_url(config.url)
            if not validated_url:
                return DownloadResponse(
                    success=False,
                    message="Invalid URL provided",
                    error="URL validation failed"
                )

            # Determine filename
            filename = config.filename
            if not filename:
                filename = os.path.basename(validated_url.split('?')[0])

            # Ensure proper extension
            if not filename.endswith(('.safetensors', '.ckpt', '.pt')):
                # Default to .safetensors if no extension
                filename = f"{filename}.safetensors"

            destination_path = Path(config.download_dir) / filename

            # Check if file already exists
            if destination_path.exists() and destination_path.stat().st_size > 0:
                file_size_mb = destination_path.stat().st_size / (1024 * 1024)
                logger.info(f"File already exists: {destination_path} ({file_size_mb:.1f} MB)")
                return DownloadResponse(
                    success=True,
                    message="File already exists",
                    file_path=str(destination_path),
                    file_name=filename,
                    size_mb=round(file_size_mb, 2),
                    download_method=None
                )

            logger.info(f"Starting download: {validated_url}")

            # Try download methods in order
            result_path, method = self._download_with_fallback(
                validated_url,
                destination_path,
                config.api_token
            )

            if result_path and result_path.exists():
                file_size_mb = result_path.stat().st_size / (1024 * 1024)
                logger.info(f"Download complete: {result_path} ({file_size_mb:.1f} MB) via {method}")
                return DownloadResponse(
                    success=True,
                    message=f"Download complete via {method}",
                    file_path=str(result_path),
                    file_name=filename,
                    size_mb=round(file_size_mb, 2),
                    download_method=method
                )
            else:
                return DownloadResponse(
                    success=False,
                    message="All download methods failed",
                    error="Download failed after trying all methods"
                )

        except Exception as e:
            logger.exception(f"Download error: {e}")
            return DownloadResponse(
                success=False,
                message="Download failed",
                error=str(e)
            )

    async def list_models(self) -> ListModelsResponse:
        """List all downloaded models, VAEs, and LoRAs."""
        try:
            def get_files_info(directory: Path, file_type: ModelType) -> list[ModelInfo]:
                files = []
                if directory.exists():
                    patterns = ["*.safetensors", "*.ckpt", "*.pt"]
                    for pattern in patterns:
                        for file_path in directory.glob(pattern):
                            size_bytes = file_path.stat().st_size
                            size_mb = size_bytes / (1024 * 1024)

                            files.append(ModelInfo(
                                name=file_path.name,
                                path=str(file_path),
                                size_mb=round(size_mb, 2),
                                type=file_type
                            ))

                # Sort by name
                files.sort(key=lambda x: x.name.lower())
                return files

            models = get_files_info(self.pretrained_model_dir, ModelType.MODEL)
            vaes = get_files_info(self.vae_dir, ModelType.VAE)
            loras = get_files_info(self.lora_dir, ModelType.LORA)

            return ListModelsResponse(
                success=True,
                models=models,
                vaes=vaes,
                loras=loras,
                model_dir=str(self.pretrained_model_dir),
                vae_dir=str(self.vae_dir),
                lora_dir=str(self.lora_dir)
            )

        except Exception as e:
            logger.exception(f"Failed to list models: {e}")
            return ListModelsResponse(
                success=False,
                models=[],
                vaes=[],
                loras=[],
                model_dir=str(self.pretrained_model_dir),
                vae_dir=str(self.vae_dir),
                lora_dir=str(self.lora_dir)
            )

    async def delete_model(self, file_path: str) -> DeleteModelResponse:
        """Delete a model file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            path.unlink()
            logger.info(f"Deleted model: {file_path}")

            return DeleteModelResponse(
                success=True,
                message=f"Deleted {path.name}",
                file_path=file_path
            )

        except Exception as e:
            logger.exception(f"Failed to delete model: {e}")
            return DeleteModelResponse(
                success=False,
                message=f"Delete failed: {e}",
                file_path=file_path
            )

    def _validate_url(self, url: str) -> Optional[str]:
        """Validate and normalize download URL."""
        import re

        # Normalize Civitai URLs
        if "civitai.com" in url:
            # Extract model version ID if present
            if match := re.search(r"modelVersionId=(\d+)", url):
                return f"https://civitai.com/api/download/models/{match.group(1)}"

        return url

    def _download_with_fallback(
        self,
        url: str,
        destination: Path,
        api_token: Optional[str] = None
    ) -> tuple[Optional[Path], Optional[DownloadMethod]]:
        """
        Try download methods in order until one succeeds.

        Returns:
            (destination_path, method_used) or (None, None) on failure
        """
        # For HuggingFace URLs, prioritize hf-transfer (HF-optimized)
        if "huggingface.co" in url and shutil.which("hf-transfer"):
            logger.info("HuggingFace URL detected - trying hf-transfer first")
            if self._try_hf_transfer(url, destination, api_token):
                return destination, DownloadMethod.HF_TRANSFER

        # Method 1: aria2c (fastest, multi-connection)
        if shutil.which("aria2c"):
            if self._try_aria2c(url, destination, api_token):
                return destination, DownloadMethod.ARIA2C

        # Method 2: wget (reliable fallback)
        if shutil.which("wget"):
            if self._try_wget(url, destination, api_token):
                return destination, DownloadMethod.WGET

        # Method 3: Python requests (final fallback)
        if self._try_requests(url, destination, api_token):
            return destination, DownloadMethod.REQUESTS

        return None, None

    def _try_aria2c(
        self,
        url: str,
        destination: Path,
        api_token: Optional[str] = None
    ) -> bool:
        """Try downloading with aria2c."""
        logger.info("Attempting download with aria2c...")
        try:
            header = ""
            download_url = url
            parsed = urlparse(url)
            hostname = parsed.hostname or ""

            # Handle API tokens
            if "civitai.com" in hostname and api_token and "hf" not in api_token:
                download_url = f"{url}?token={api_token}"
            elif "huggingface.co" in hostname and api_token:
                header = f"Authorization: Bearer {api_token}"

            command = [
                "aria2c", download_url,
                "--console-log-level=warn",
                "-c",  # Continue partial downloads
                "-s", "16",  # Split into 16 connections
                "-x", "16",  # Max connections
                "-k", "10M",  # Min split size
                "-d", str(destination.parent),  # Directory
                "-o", destination.name  # Output filename
            ]

            if header:
                command.extend(["--header", header])

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"✅ Download complete with aria2c: {destination}")
                return True
            else:
                logger.warning(f"aria2c failed with exit code {result.returncode}")
                return False

        except Exception as e:
            logger.warning(f"aria2c error: {e}")
            return False

    def _try_hf_transfer(
        self,
        url: str,
        destination: Path,
        api_token: Optional[str] = None
    ) -> bool:
        """Try downloading with hf-transfer (HuggingFace specific)."""
        logger.info("Attempting download with hf-transfer...")
        try:
            env = os.environ.copy()
            if api_token:
                env["HF_HUB_TOKEN"] = api_token

            result = subprocess.run(
                ["hf-transfer", "download", url, "--local-dir", str(destination.parent)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )

            if result.returncode == 0:
                logger.info(f"✅ Download complete with hf-transfer: {destination}")
                return True
            else:
                logger.warning(f"hf-transfer failed with exit code {result.returncode}")
                return False

        except Exception as e:
            logger.warning(f"hf-transfer error: {e}")
            return False

    def _try_wget(
        self,
        url: str,
        destination: Path,
        api_token: Optional[str] = None
    ) -> bool:
        """Try downloading with wget."""
        logger.info("Attempting download with wget...")
        try:
            download_url = url
            wget_args = ["wget", "-O", str(destination)]

            # Handle API tokens
            if "civitai.com" in url and api_token and "hf" not in api_token:
                download_url = f"{url}?token={api_token}"
            elif "huggingface.co" in url and api_token:
                wget_args.extend(["--header", f"Authorization: Bearer {api_token}"])

            wget_args.append(download_url)

            result = subprocess.run(
                wget_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"✅ Download complete with wget: {destination}")
                return True
            else:
                logger.warning(f"wget failed with exit code {result.returncode}")
                return False

        except Exception as e:
            logger.warning(f"wget error: {e}")
            return False

    def _try_requests(
        self,
        url: str,
        destination: Path,
        api_token: Optional[str] = None
    ) -> bool:
        """Try downloading with Python requests (final fallback)."""
        logger.info("Attempting download with Python requests (final fallback)...")
        try:
            import requests

            headers = {}
            download_url = url

            # Handle API tokens
            if "civitai.com" in url and api_token and "hf" not in api_token:
                download_url = f"{url}?token={api_token}"
            elif "huggingface.co" in url and api_token:
                headers["Authorization"] = f"Bearer {api_token}"

            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.debug(f"Progress: {percent:.1f}%")

            logger.info(f"✅ Download complete with Python requests: {destination}")
            return True

        except Exception as e:
            logger.warning(f"requests error: {e}")
            return False


# Global service instance
model_service = ModelService()
