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

from services.models.lora import (
    LoRAResizeRequest,
    LoRAResizeResponse,
    HuggingFaceUploadRequest,
    HuggingFaceUploadResponse,
    LoRAMergeRequest,
    LoRAMergeResponse,
    CheckpointMergeRequest,
    CheckpointMergeResponse,
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
        self.tools_path = (
            self.project_root / "trainer" / "derrian_backend" / "sd_scripts" / "tools"
        )
        self.resize_script = self.scripts_path / "resize_lora.py"
        self.merge_scripts = {
            "sd": self.scripts_path / "merge_lora.py",
            "sdxl": self.scripts_path / "sdxl_merge_lora.py",
            "flux": self.scripts_path / "flux_merge_lora.py",
            "svd": self.scripts_path / "svd_merge_lora.py",
        }
        self.checkpoint_merge_script = self.tools_path / "merge_models.py"

    def _validate_device(self, device: str) -> None:
        """Raise ValidationError if CUDA is requested but unavailable."""
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValidationError(
                        "CUDA device requested but no GPU is available. Use 'cpu' instead."
                    )
            except ImportError:
                raise ValidationError(
                    "CUDA device requested but PyTorch is not installed."
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

            self._validate_device(request.device)

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

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1800)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ProcessError("LoRA resize timed out after 30 minutes")

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
        Upload files to HuggingFace Hub.

        Supports uploading multiple files to a repo, with optional
        remote folder prefix and pull request creation.

        Args:
            request: HuggingFace upload request

        Returns:
            HuggingFaceUploadResponse with per-file results
        """
        try:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise ProcessError(
                    "huggingface_hub not installed. "
                    "Install with: pip install huggingface_hub"
                )

            # Validate all files exist
            for file_path_str in request.file_paths:
                fp = Path(file_path_str)
                if not fp.exists():
                    raise NotFoundError(f"File not found: {file_path_str}")

            api = HfApi(token=request.token)

            # Ensure repo exists (create if needed)
            try:
                api.create_repo(
                    repo_id=request.repo_id,
                    repo_type=request.repo_type,
                    exist_ok=True,
                )
                logger.info(f"Repository ready: {request.repo_id}")
            except Exception as e:
                logger.warning(f"Repo creation warning: {e}")

            # Upload each file
            uploaded_files: list[str] = []
            failed_files: list[str] = []

            for file_path_str in request.file_paths:
                fp = Path(file_path_str)
                # Build the remote path: remote_folder/filename
                path_in_repo = fp.name
                if request.remote_folder:
                    path_in_repo = f"{request.remote_folder.strip('/')}/{fp.name}"

                try:
                    api.upload_file(
                        path_or_fileobj=str(fp),
                        path_in_repo=path_in_repo,
                        repo_id=request.repo_id,
                        repo_type=request.repo_type,
                        commit_message=request.commit_message,
                        create_pr=request.create_pr,
                    )
                    uploaded_files.append(fp.name)
                    logger.info(f"Uploaded {fp.name} to {request.repo_id}/{path_in_repo}")
                except Exception as e:
                    failed_files.append(fp.name)
                    logger.error(f"Failed to upload {fp.name}: {e}")

            success = len(uploaded_files) > 0
            return HuggingFaceUploadResponse(
                success=success,
                repo_id=request.repo_id,
                uploaded_files=uploaded_files,
                failed_files=failed_files,
                error=None if success else "All files failed to upload",
            )

        except (ValidationError, NotFoundError, ProcessError) as e:
            logger.error(f"HuggingFace upload failed: {e}")
            return HuggingFaceUploadResponse(
                success=False,
                repo_id=request.repo_id,
                error=str(e),
            )

        except Exception as e:
            logger.exception(f"Unexpected error during HuggingFace upload: {e}")
            return HuggingFaceUploadResponse(
                success=False,
                repo_id=request.repo_id,
                error=f"Internal error: {e}",
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

            self._validate_device(request.device)

            # Build command
            command = [
                sys.executable,
                str(merge_script),
                "--save_to", str(output_path),
                "--save_precision", request.save_precision,  # merge_lora.py uses --save_precision
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

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3600)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ProcessError("LoRA merge timed out after 1 hour")

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

    async def merge_checkpoint(self, request: CheckpointMergeRequest) -> CheckpointMergeResponse:
        """
        Merge multiple checkpoint models into one.

        Args:
            request: Checkpoint merge request

        Returns:
            CheckpointMergeResponse with result
        """
        try:
            # Validate all input checkpoints exist
            for checkpoint_input in request.checkpoint_inputs:
                input_path = Path(checkpoint_input.path)
                if not input_path.exists():
                    raise NotFoundError(f"Input checkpoint not found: {checkpoint_input.path}")

                if input_path.suffix.lower() not in ['.safetensors', '.ckpt']:
                    raise ValidationError(f"Invalid checkpoint file format: {input_path.suffix}")

            # Validate output path
            output_path = Path(request.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check merge script exists
            if not self.checkpoint_merge_script.exists():
                raise NotFoundError(
                    "Checkpoint merge script not found. "
                    "Please ensure the training backend is installed."
                )

            self._validate_device(request.device)

            # Build command
            command = [
                sys.executable,
                str(self.checkpoint_merge_script),
                "--output", str(output_path),
                "--precision", request.precision,
                "--saving_precision", request.save_precision,  # merge_models.py uses --saving_precision (different from merge_lora.py's --save_precision)
            ]

            # Collect all model paths and ratios (Kohya expects them as separate lists)
            model_paths = [cp_input.path for cp_input in request.checkpoint_inputs]
            model_ratios = [str(cp_input.ratio) for cp_input in request.checkpoint_inputs]

            # Add all models and ratios as single arguments with multiple values
            command.extend(["--models"] + model_paths)
            command.extend(["--ratios"] + model_ratios)

            # Add optional flags
            if request.unet_only:
                command.append("--unet_only")

            if request.show_skipped:
                command.append("--show_skipped")

            # Add device
            if request.device != "cpu":
                command.extend(["--device", request.device])

            logger.info(
                f"Merging {len(request.checkpoint_inputs)} checkpoints "
                f"(UNet only: {request.unet_only})"
            )

            # Execute merge
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3600)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ProcessError("Checkpoint merge timed out after 1 hour")

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                raise ProcessError(f"Checkpoint merge failed: {error_msg}")

            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            logger.info(
                f"Checkpoints merged successfully: {output_path.name} "
                f"({file_size_mb:.2f} MB)"
            )

            return CheckpointMergeResponse(
                success=True,
                message=f"Successfully merged {len(request.checkpoint_inputs)} checkpoints",
                output_path=str(output_path),
                merged_count=len(request.checkpoint_inputs),
                file_size_mb=round(file_size_mb, 2)
            )

        except (ValidationError, NotFoundError, ProcessError) as e:
            logger.error(f"Checkpoint merge failed: {e}")
            return CheckpointMergeResponse(
                success=False,
                message=str(e)
            )

        except Exception as e:
            logger.exception(f"Unexpected error during checkpoint merge: {e}")
            return CheckpointMergeResponse(
                success=False,
                message=f"Internal error: {e}"
            )


# Global service instance
lora_service = LoRAService()
