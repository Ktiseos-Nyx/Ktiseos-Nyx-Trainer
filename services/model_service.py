"""
Model download service for HuggingFace and Civitai.

Handles downloading models, VAEs, and LoRAs with multiple fallback methods:
- hf_hub_download via hf_xet (HuggingFace URLs, auto-enabled in huggingface_hub>=0.32)
- aria2c (Civitai and other direct URLs)
- wget (reliable fallback)
- requests (Python fallback)
"""

import asyncio
import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse, urljoin
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
from services.core.validation import PROJECT_ROOT

logger = logging.getLogger(__name__)


# Browser-like User-Agent to avoid geo-blocking from CDNs
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


class ModelService:
    """
    Service for downloading and managing models, VAEs, and LoRAs.

    Supports multiple download methods with automatic fallback:
    1. hf_hub_download - HuggingFace URLs only; uses hf_xet (Rust/Xet backend,
       adaptive concurrency) automatically when huggingface_hub>=0.32 is installed
    2. aria2c - Civitai and other direct URLs, multi-connection
    3. wget - Reliable fallback
    4. requests - Python fallback (slowest)
    """

    def __init__(self):
        # Anchor to the source file (via PROJECT_ROOT), NOT the process CWD — the backend
        # starts from /root or similar on VastAI/RunPod, so downloads landed in the wrong
        # place relative to where the trainer/merge tools look for them.
        self.project_root = PROJECT_ROOT
        self.pretrained_model_dir = self.project_root / "pretrained_model"
        self.vae_dir = self.project_root / "vae"
        self.lora_dir = self.project_root / "output"  # LoRAs stored in output

        # Ensure directories exist
        self.pretrained_model_dir.mkdir(parents=True, exist_ok=True)
        self.vae_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)

    async def download_model_or_vae(
        self,
        config: DownloadConfig,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> DownloadResponse:
        """
        Download a model or VAE with automatic fallback chain.

        Args:
            config: Download configuration
            progress_callback: Optional callable invoked with an integer percent
                (0-99) as the download grows. Best-effort: it relies on a HEAD
                request for the total size, so when that can't be determined
                (some gated/xet cases) no progress is reported and the caller
                should treat progress as indeterminate.

        Returns:
            DownloadResponse with download result
        """
        staging_path: Optional[Path] = None
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

            # When a caller supplies a redirect allowlist (Arc En Ciel), resolve
            # the redirect chain up front — rejecting any non-HTTPS hop or a host
            # outside the allowlist — and download the validated final URL.
            # Civitai/HuggingFace (no allowlist set) keep their prior behaviour.
            download_url = validated_url
            if config.allowed_redirect_hosts:
                resolved = await asyncio.to_thread(
                    self._resolve_download_url,
                    validated_url,
                    config.allowed_redirect_hosts,
                )
                if not resolved:
                    return DownloadResponse(
                        success=False,
                        message="Download blocked: redirect to a disallowed host or scheme",
                        error="redirect allowlist violation",
                    )
                download_url = resolved

            logger.info(f"Starting download: {download_url}")

            # Stage to a ".part" file in the same directory so a partial, corrupt,
            # or hash-mismatched download never appears under its real name. On
            # success it is SHA-verified (when a hash is supplied) then atomically
            # renamed into place; on any failure it is removed.
            staging_path = destination_path.with_name(destination_path.name + ".part")
            if staging_path.exists():
                staging_path.unlink()  # clear a stale partial from a prior run

            # Best-effort progress: fetch total size, then poll the growing staging
            # file on disk while the download runs. Decoupled from the download
            # method so it works for aria2c, hf_hub_download, and requests alike.
            stop_event: Optional[asyncio.Event] = None
            watcher_task: Optional[asyncio.Task] = None
            if progress_callback is not None:
                total_size = await asyncio.to_thread(
                    self._get_total_size, download_url, config.api_token
                )
                if total_size:
                    stop_event = asyncio.Event()
                    watcher_task = asyncio.create_task(
                        self._watch_progress(
                            staging_path, total_size, progress_callback, stop_event
                        )
                    )

            # Try download methods in order. The fallback chain runs blocking work
            # (subprocess.run for aria2c/wget, synchronous hf_hub_download), so it is
            # offloaded to a worker thread — otherwise it would freeze the asyncio event
            # loop for the entire multi-GB download and starve every other request
            # (including health checks), which the gateway reads as a dead origin (502).
            try:
                result_path, method = await asyncio.to_thread(
                    self._download_with_fallback,
                    download_url,
                    staging_path,
                    config.api_token,
                )
            finally:
                if stop_event is not None:
                    stop_event.set()
                if watcher_task is not None:
                    await watcher_task

            if not (result_path and result_path.exists()):
                if staging_path.exists():
                    staging_path.unlink()
                return DownloadResponse(
                    success=False,
                    message="All download methods failed",
                    error="Download failed after trying all methods"
                )

            # Verify the content hash before the file may take its final name. Arc
            # En Ciel supplies sha256; skipped when no hash is given, so Civitai/HF
            # downloads are unaffected.
            if config.expected_sha256:
                actual_sha = await asyncio.to_thread(self._compute_sha256, staging_path)
                if actual_sha.lower() != config.expected_sha256.lower():
                    staging_path.unlink()
                    logger.error(
                        "SHA-256 mismatch for %s (expected %s, got %s) - file rejected",
                        filename, config.expected_sha256, actual_sha,
                    )
                    return DownloadResponse(
                        success=False,
                        message="Downloaded file failed SHA-256 verification",
                        error="sha256 mismatch",
                    )
                logger.info("SHA-256 verified for %s", filename)

            # Atomic move into place: staging is in the same directory as the
            # destination (same filesystem), so os.replace can't leave a half file.
            os.replace(str(staging_path), str(destination_path))

            file_size_mb = destination_path.stat().st_size / (1024 * 1024)
            logger.info(f"Download complete: {destination_path} ({file_size_mb:.1f} MB) via {method}")
            return DownloadResponse(
                success=True,
                message=f"Download complete via {method}",
                file_path=str(destination_path),
                file_name=filename,
                size_mb=round(file_size_mb, 2),
                download_method=method
            )

        except Exception as e:
            logger.exception(f"Download error: {e}")
            if staging_path is not None:
                try:
                    if staging_path.exists():
                        staging_path.unlink()
                except OSError:
                    pass
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

    def _get_total_size(self, url: str, api_token: Optional[str] = None) -> Optional[int]:
        """
        Resolve the total download size via a HEAD request (following redirects).

        Mirrors the auth/url handling of the download methods so gated HF and
        token-gated Civitai files report a size. Returns None when the size can't
        be determined — the caller then treats progress as indeterminate.
        """
        try:
            import requests

            headers = {}
            download_url = url
            hostname = (urlparse(url).hostname or "")
            is_hf_host = hostname == "huggingface.co" or hostname.endswith(".huggingface.co")
            if "civitai" in hostname and api_token and "hf" not in api_token:
                download_url = f"{url}?token={api_token}"
            elif is_hf_host and api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            if is_hf_host and "/blob/" in download_url:
                download_url = download_url.replace("/blob/", "/resolve/", 1)

            response = requests.head(
                download_url, headers=headers, allow_redirects=True, timeout=15
            )
            content_length = response.headers.get("content-length")
            if content_length and content_length.isdigit():
                return int(content_length)
            return None
        except Exception as e:
            logger.debug(f"Could not determine total size for progress: {e}")
            return None

    def _downloaded_bytes(self, destination: Path) -> int:
        """
        Bytes written so far for this download.

        Uses the larger of the final destination file and any in-progress
        ``*.incomplete`` file under ``.cache`` (where hf_hub_download stages
        before moving). Taking the max — not a sum — avoids double-counting
        across the move and never counts unrelated pre-existing models in the
        directory.
        """
        sizes: list[int] = []
        try:
            if destination.exists():
                sizes.append(destination.stat().st_size)
        except OSError:
            pass

        cache_dir = destination.parent / ".cache"
        if cache_dir.exists():
            try:
                for partial in cache_dir.rglob("*.incomplete"):
                    try:
                        sizes.append(partial.stat().st_size)
                    except OSError:
                        pass
            except OSError:
                pass

        return max(sizes) if sizes else 0

    async def _watch_progress(
        self,
        destination: Path,
        total_size: int,
        callback: Callable[[int], None],
        stop_event: asyncio.Event,
    ) -> None:
        """Poll the growing file every second and report percent (capped at 99)."""
        while not stop_event.is_set():
            downloaded = await asyncio.to_thread(self._downloaded_bytes, destination)
            if total_size > 0:
                callback(min(99, int(downloaded * 100 / total_size)))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    def _resolve_download_url(
        self,
        url: str,
        allowed_hosts: list[str],
        max_hops: int = 5,
    ) -> Optional[str]:
        """Follow a download's redirect chain, enforcing HTTPS + a host allowlist.

        Every hop — the initial URL and each redirect target — must use https and
        a hostname that is (or is a subdomain of) an entry in ``allowed_hosts``;
        otherwise the download is refused. Returns the final resolved URL, or None
        on any violation or resolution failure.

        Only used when a caller supplies ``allowed_redirect_hosts`` (Arc En Ciel).
        Civitai/HuggingFace downloads keep their previous redirect behaviour.
        """
        import requests
        allowed = {h.lower() for h in allowed_hosts}
        current = url
        headers = {"User-Agent": BROWSER_UA}
        for _ in range(max_hops):
            parsed = urlparse(current)
            if parsed.scheme != "https":
                logger.error("Redirect allowlist: refusing non-HTTPS URL: %s", current)
                return None
            host = (parsed.hostname or "").lower()
            if not any(host == a or host.endswith("." + a) for a in allowed):
                logger.error("Redirect allowlist: host '%s' not permitted (%s)", host, current)
                return None
            try:
                resp = requests.head(
                    current, headers=headers, allow_redirects=False, timeout=15
                )
            except requests.RequestException as exc:
                logger.warning("Redirect allowlist: HEAD failed for %s: %s", current, exc)
                return None
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    logger.error("Redirect allowlist: %s redirected with no Location", current)
                    return None
                current = urljoin(current, location)
                continue
            # Non-redirect response reached; its host was validated at loop top.
            return current
        logger.error("Redirect allowlist: too many redirects from %s", url)
        return None

    def _compute_sha256(self, path: Path) -> str:
        """Return the lowercase hex SHA-256 of a file.

        Read in 1 MB chunks so multi-GB model files are never loaded into memory
        all at once.
        """
        import hashlib
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

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
        # For HuggingFace URLs, use hf_hub_download which automatically uses
        # hf_xet (Rust Xet backend, adaptive concurrency) when available
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        is_huggingface = hostname == "huggingface.co" or hostname.endswith(".huggingface.co")

        if is_huggingface:
            logger.info("HuggingFace URL detected - using hf_hub_download (hf_xet)")
            if self._try_hf_hub_download(url, destination, api_token):
                return destination, DownloadMethod.HF_XET
            # Normalize /blob/ page URLs to /resolve/ artifact URLs before falling
            # back to generic download methods, so they receive the actual file URL
            # rather than the HTML page, which would be silently treated as success.
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/", 1)
                logger.info(f"Normalized HF blob URL to resolve URL for generic fallback: {url}")

        # aria2c for Civitai and other direct URLs
        if shutil.which("aria2c"):
            if self._try_aria2c(url, destination, api_token):
                return destination, DownloadMethod.ARIA2C

        # wget fallback
        if shutil.which("wget"):
            if self._try_wget(url, destination, api_token):
                return destination, DownloadMethod.WGET

        # Python requests (final fallback)
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
            is_hf_host = hostname == "huggingface.co" or hostname.endswith(".huggingface.co")
            if "civitai" in hostname and api_token and "hf" not in api_token:
                download_url = f"{url}?token={api_token}"
            elif is_hf_host and api_token:
                header = f"Authorization: Bearer {api_token}"

            command = [
                "aria2c", download_url,
                "--console-log-level=warn",
                "-c",  # Continue partial downloads
                "-s", "8",  # Split into 8 connections
                "-x", "8",  # Max connections per server
                "-k", "1M",  # Min split size (smaller = more effective splitting)
                "-d", str(destination.parent),  # Directory
                "-o", destination.name,  # Output filename
                "--user-agent", BROWSER_UA,
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

    def _try_hf_hub_download(
        self,
        url: str,
        destination: Path,
        api_token: Optional[str] = None
    ) -> bool:
        """
        Download via huggingface_hub.hf_hub_download().

        hf_xet (Rust Xet backend with adaptive concurrency) is used automatically
        when huggingface_hub>=0.32 is installed — no extra configuration needed.
        hf_transfer is deprecated and no longer used.
        """
        import re
        logger.info("Attempting download with hf_hub_download (hf_xet)...")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            logger.warning("huggingface_hub not available: %s", e)
            return False

        # Import specific exception types — try huggingface_hub.errors (>=0.26)
        # then huggingface_hub.utils (older releases), then fall back to OSError.
        try:
            from huggingface_hub.errors import HfHubHTTPError
        except ImportError:
            try:
                from huggingface_hub.utils import HfHubHTTPError  # type: ignore[no-redef]
            except ImportError:
                HfHubHTTPError = OSError  # type: ignore[misc,assignment]
        try:
            from huggingface_hub.errors import RepositoryNotFoundError
        except ImportError:
            try:
                from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore[no-redef]
            except ImportError:
                RepositoryNotFoundError = OSError  # type: ignore[misc,assignment]
        try:
            from huggingface_hub.errors import RevisionNotFoundError
        except ImportError:
            try:
                from huggingface_hub.utils import RevisionNotFoundError  # type: ignore[no-redef]
            except ImportError:
                RevisionNotFoundError = OSError  # type: ignore[misc,assignment]
        try:
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError:
            try:
                from huggingface_hub.utils import LocalEntryNotFoundError  # type: ignore[no-redef]
            except ImportError:
                LocalEntryNotFoundError = OSError  # type: ignore[misc,assignment]

        try:
            # Parse https://huggingface.co/{repo_id}/resolve/{revision}/{filepath}
            # or   https://huggingface.co/{repo_id}/blob/{revision}/{filepath}
            match = re.match(
                r'https://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+)',
                url
            )
            if not match:
                logger.warning("Could not parse HuggingFace URL — skipping hf_hub_download")
                return False

            repo_id = match.group(1)
            revision = match.group(2)
            hf_filename = match.group(3).split('?')[0]  # strip any query params

            downloaded_path = Path(hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                revision=revision,
                token=api_token or None,
                local_dir=str(destination.parent),
            ))

            # hf_hub_download may save with a subdirectory-prefixed path
            # (e.g. local_dir/subfolder/model.safetensors); flatten to the
            # caller-supplied destination.  Intentional: the destination name
            # is chosen by the caller and we always want a single flat file.
            if downloaded_path != destination and downloaded_path.exists():
                if destination.exists():
                    logger.info("Replacing existing file at destination: %s", destination)
                    destination.unlink()
                shutil.move(str(downloaded_path), str(destination))

            if destination.exists() and destination.stat().st_size > 0:
                logger.info(f"✅ Download complete with hf_hub_download: {destination}")
                return True

            logger.warning("hf_hub_download returned but file missing or empty")
            return False

        except (
            HfHubHTTPError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
            LocalEntryNotFoundError,
            EnvironmentError,
            OSError,
            ValueError,
        ) as e:
            logger.warning(f"hf_hub_download error: {e}")
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
            wget_hostname = (urlparse(url).hostname or "")
            is_hf_host = wget_hostname == "huggingface.co" or wget_hostname.endswith(".huggingface.co")
            if "civitai" in wget_hostname and api_token and "hf" not in api_token:
                download_url = f"{url}?token={api_token}"
            elif is_hf_host and api_token:
                wget_args.extend(["--header", f"Authorization: Bearer {api_token}"])

            wget_args.extend(["-U", BROWSER_UA])
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

            headers = {"User-Agent": BROWSER_UA}
            download_url = url

            # Handle API tokens
            req_hostname = (urlparse(url).hostname or "")
            is_hf_host = req_hostname == "huggingface.co" or req_hostname.endswith(".huggingface.co")
            if "civitai" in req_hostname and api_token and "hf" not in api_token:
                download_url = f"{url}?token={api_token}"
            elif is_hf_host and api_token:
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
