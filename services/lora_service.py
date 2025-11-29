"""
LoRA service for utilities (resize, upload).

Handles:
- LoRA resizing to different ranks
- HuggingFace Hub upload
- File validation
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

from services.models.lora import (
    LoRAResizeRequest,
    LoRAResizeResponse,
    HuggingFaceUploadRequest,
    HuggingFaceUploadResponse,
)
from services.core.exceptions import ValidationError, ProcessError, NotFoundError

logger = logging.getLogger(__name__)


class LoRAService:
    """
    High-level service for LoRA utilities.

    Responsibilities:
    - Resize LoRA models to different ranks
    - Upload LoRA models to HuggingFace Hub
    - Validate LoRA files
    """

    def __init__(self):
        self.project_root = Path.cwd()
        self.resize_script = (
            self.project_root / "trainer" / "derrian_backend" / "sd_scripts" /
            "networks" / "resize_lora.py"
        )

    async def resize_lora(self, request: LoRAResizeRequest) -> LoRAResizeResponse:
        """
        Resize a LoRA model to a different rank.

        Args:
            request: LoRA resize request

        Returns:
            LoRAResizeResponse with result
        """
        try:
            # Validate input file
            input_path = Path(request.input_path)
            if not input_path.exists():
                raise NotFoundError(f"Input LoRA not found: {request.input_path}")

            if input_path.suffix.lower() not in ['.safetensors', '.pt', '.ckpt']:
                raise ValidationError(f"Invalid LoRA file format: {input_path.suffix}")

            # Validate output path
            output_path = Path(request.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check resize script exists
            if not self.resize_script.exists():
                raise NotFoundError(
                    "LoRA resize script not found. "
                    "Please ensure the training backend is installed."
                )

            # Build command
            command = [
                sys.executable,
                str(self.resize_script),
                "--model", str(input_path),
                "--save_to", str(output_path),
                "--new_rank", str(request.target_dim),
                "--device", request.device,
                "--save_precision", request.save_precision,
            ]

            if request.target_alpha is not None:
                command.extend(["--new_alpha", str(request.target_alpha)])

            logger.info(f"Resizing LoRA from {input_path.name} to rank {request.target_dim}")

            # Execute resize
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                raise ProcessError(f"LoRA resize failed: {error_msg}")

            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            logger.info(f"LoRA resized successfully: {output_path.name} ({file_size_mb:.2f} MB)")

            return LoRAResizeResponse(
                success=True,
                message=f"LoRA resized to rank {request.target_dim}",
                input_path=str(input_path),
                output_path=str(output_path),
                new_dim=request.target_dim,
                file_size_mb=round(file_size_mb, 2)
            )

        except (ValidationError, NotFoundError, ProcessError) as e:
            logger.error(f"LoRA resize failed: {e}")
            return LoRAResizeResponse(
                success=False,
                message=str(e),
                input_path=request.input_path,
                output_path=request.output_path
            )

        except Exception as e:
            logger.exception(f"Unexpected error during LoRA resize: {e}")
            return LoRAResizeResponse(
                success=False,
                message=f"Internal error: {e}",
                input_path=request.input_path,
                output_path=request.output_path
            )

    async def upload_to_huggingface(
        self,
        request: HuggingFaceUploadRequest
    ) -> HuggingFaceUploadResponse:
        """
        Upload a LoRA model to HuggingFace Hub.

        Args:
            request: HuggingFace upload request

        Returns:
            HuggingFaceUploadResponse with result
        """
        try:
            # Lazy import to avoid dependency issues
            try:
                from huggingface_hub import HfApi, create_repo, upload_file
            except ImportError:
                raise ProcessError(
                    "huggingface_hub not installed. "
                    "Install with: pip install huggingface_hub"
                )

            # Validate LoRA file
            lora_path = Path(request.lora_path)
            if not lora_path.exists():
                raise NotFoundError(f"LoRA file not found: {request.lora_path}")

            if lora_path.suffix.lower() not in ['.safetensors', '.pt', '.ckpt']:
                raise ValidationError(f"Invalid LoRA file format: {lora_path.suffix}")

            # Initialize HF API
            api = HfApi(token=request.token)

            # Create repo if requested
            if request.create_repo:
                try:
                    repo_url = create_repo(
                        repo_id=request.repo_id,
                        token=request.token,
                        private=request.private,
                        exist_ok=True,
                        repo_type="model"
                    )
                    logger.info(f"Repository ready: {repo_url}")
                except Exception as e:
                    logger.warning(f"Repo creation warning: {e}")

            # Prepare metadata for README
            readme_content = self._generate_readme(request)

            # Upload README if metadata provided
            if readme_content:
                readme_path = lora_path.parent / "README.md"
                readme_path.write_text(readme_content, encoding='utf-8')

                try:
                    upload_file(
                        path_or_fileobj=str(readme_path),
                        path_in_repo="README.md",
                        repo_id=request.repo_id,
                        token=request.token,
                        commit_message=f"{request.commit_message} (README)"
                    )
                    logger.info("Uploaded README.md")
                except Exception as e:
                    logger.warning(f"README upload warning: {e}")

            # Upload LoRA file
            logger.info(f"Uploading {lora_path.name} to {request.repo_id}...")

            commit_info = upload_file(
                path_or_fileobj=str(lora_path),
                path_in_repo=lora_path.name,
                repo_id=request.repo_id,
                token=request.token,
                commit_message=request.commit_message
            )

            repo_url = f"https://huggingface.co/{request.repo_id}"

            logger.info(f"Successfully uploaded to {repo_url}")

            return HuggingFaceUploadResponse(
                success=True,
                message=f"LoRA uploaded successfully to {request.repo_id}",
                repo_url=repo_url,
                commit_hash=commit_info.commit_url.split('/')[-1] if hasattr(commit_info, 'commit_url') else None,
                errors=[]
            )

        except (ValidationError, NotFoundError, ProcessError) as e:
            logger.error(f"HuggingFace upload failed: {e}")
            return HuggingFaceUploadResponse(
                success=False,
                message=str(e),
                errors=[str(e)]
            )

        except Exception as e:
            logger.exception(f"Unexpected error during HuggingFace upload: {e}")
            return HuggingFaceUploadResponse(
                success=False,
                message=f"Internal error: {e}",
                errors=[str(e)]
            )

    def _generate_readme(self, request: HuggingFaceUploadRequest) -> Optional[str]:
        """Generate README.md content from metadata."""
        if not any([request.model_type, request.base_model, request.description, request.trigger_word]):
            return None

        lines = [f"# {request.repo_id.split('/')[-1]}", ""]

        if request.description:
            lines.extend([request.description, ""])

        lines.append("## Model Details")
        if request.model_type:
            lines.append(f"- **Architecture**: {request.model_type}")
        if request.base_model:
            lines.append(f"- **Base Model**: {request.base_model}")
        if request.trigger_word:
            lines.append(f"- **Trigger Word**: `{request.trigger_word}`")

        if request.tags:
            lines.extend(["", "## Tags", request.tags])

        lines.extend([
            "",
            "## Usage",
            "Download and use this LoRA with your favorite Stable Diffusion interface.",
            "",
            f"Generated with [Ktiseos-Nyx-Trainer](https://github.com/yourusername/Ktiseos-Nyx-Trainer)"
        ])

        return '\n'.join(lines)


# Global service instance
lora_service = LoRAService()
