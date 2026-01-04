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
    LoRAMergeRequest,
    LoRAMergeResponse,
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
        self.scripts_path = (
            self.project_root / "trainer" / "derrian_backend" / "sd_scripts" / "networks"
        )
        self.resize_script = self.scripts_path / "resize_lora.py"
        self.merge_scripts = {
            "sd": self.scripts_path / "merge_lora.py",
            "sdxl": self.scripts_path / "sdxl_merge_lora.py",
            "flux": self.scripts_path / "flux_merge_lora.py",
            "svd": self.scripts_path / "svd_merge_lora.py",
        }

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

        # FIX: resize_lora.py does not support --new_alpha.
        # It calculates alpha automatically based on SVD rank.
        # if request.target_alpha is not None:
        #     command.extend(["--new_alpha", str(request.target_alpha)])

            logger.info(
                "Resizing LoRA from %s to rank %s",
                input_path.name,
                request.target_dim
            )

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

            logger.info(
                "LoRA resized successfully: %s (%.2f MB)",
                output_path.name,
                file_size_mb
            )

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

    async def merge_lora(self, request: LoRAMergeRequest) -> LoRAMergeResponse:
        """
        Merge multiple LoRA models into one.

        Args:
            request: LoRA merge request

        Returns:
            LoRAMergeResponse with result
        """
        try:
            # Validate all input LoRAs exist
            for lora_input in request.lora_inputs:
                input_path = Path(lora_input.path)
                if not input_path.exists():
                    raise NotFoundError(f"Input LoRA not found: {lora_input.path}")

                if input_path.suffix.lower() not in ['.safetensors', '.pt', '.ckpt']:
                    raise ValidationError(f"Invalid LoRA file format: {input_path.suffix}")

            # Validate output path
            output_path = Path(request.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Select appropriate merge script
            model_type = request.model_type.lower()
            if model_type not in self.merge_scripts:
                raise ValidationError(
                    f"Unsupported model type: {model_type}. "
                    f"Supported types: {list(self.merge_scripts.keys())}"
                )

            merge_script = self.merge_scripts[model_type]
            if not merge_script.exists():
                raise NotFoundError(
                    f"Merge script not found for {model_type}. "
                    "Please ensure the training backend is installed."
                )

            # Build command
            command = [
                sys.executable,
                str(merge_script),
                "--save_to", str(output_path),
                "--save_precision", request.save_precision,
                "--precision", request.precision,
            ]

            # Collect all model paths and ratios (Kohya expects them as separate lists)
            model_paths = [lora_input.path for lora_input in request.lora_inputs]
            model_ratios = [str(lora_input.ratio) for lora_input in request.lora_inputs]

            # Add all models and ratios as single arguments with multiple values
            command.extend(["--models"] + model_paths)
            command.extend(["--ratios"] + model_ratios)

            # Add device
            if request.device != "cpu":
                command.extend(["--device", request.device])

            logger.info(
                f"Merging {len(request.lora_inputs)} LoRAs "
                f"using {model_type} merge script"
            )

            # Execute merge
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                raise ProcessError(f"LoRA merge failed: {error_msg}")

            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            logger.info(
                f"LoRAs merged successfully: {output_path.name} "
                f"({file_size_mb:.2f} MB)"
            )

            return LoRAMergeResponse(
                success=True,
                message=f"Successfully merged {len(request.lora_inputs)} LoRAs",
                output_path=str(output_path),
                merged_count=len(request.lora_inputs),
                file_size_mb=round(file_size_mb, 2)
            )

        except (ValidationError, NotFoundError, ProcessError) as e:
            logger.error(f"LoRA merge failed: {e}")
            return LoRAMergeResponse(
                success=False,
                message=str(e)
            )

        except Exception as e:
            logger.exception(f"Unexpected error during LoRA merge: {e}")
            return LoRAMergeResponse(
                success=False,
                message=f"Internal error: {e}"
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
            "Generated with [Ktiseos-Nyx-Trainer]"
            "(https://github.com/yourusername/Ktiseos-Nyx-Trainer)"
        ])

        return '\n'.join(lines)


# Global service instance
lora_service = LoRAService()
