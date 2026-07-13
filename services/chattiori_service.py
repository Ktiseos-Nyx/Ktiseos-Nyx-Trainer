"""
Chattiori service — wraps Chattiori's merge.py and lora_bake.py as async
subprocesses with job-manager integration for WebSocket log streaming.

Responsibilities:
  - Advanced checkpoint merging via Chattiori's merge.py
  - LoRA baking (multi-LoRA → checkpoint) via Chattiori's lora_bake.py
  - Input path validation before subprocess launch
  - Device availability checks (CUDA guard)
"""

import sys
import asyncio
import logging
import re
from pathlib import Path

from services.models.chattiori import (
    CheckpointAdvancedMergeRequest,
    CheckpointAdvancedMergeResponse,
    BakeRequest,
    BakeResponse,
)
from services.core.exceptions import ValidationError, NotFoundError, ProcessError
from services.core.validation import PROJECT_ROOT
from services.models.job import JobType
from services.jobs.job_manager import job_manager

logger = logging.getLogger(__name__)

# tqdm progress bar regex: matches "  45%|" or "100%|" at the start of a line
_TQDM_PROGRESS_RE = re.compile(r"^\s*(\d+)%\s*\|")


class ChattioriService:
    """
    High-level service for Chattiori checkpoint merging and LoRA baking.

    Both operations run as async subprocesses managed by the global JobManager.
    The service validates all inputs, builds the CLI command array, spawns the
    subprocess, and returns a job_id immediately.  Callers poll job status via
    the existing GET /jobs/{job_id} endpoint and stream logs over WebSocket.
    """

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.chattiori_path = self.project_root / "trainer" / "chattiori"
        self.merge_script = self.chattiori_path / "merge.py"
        self.bake_script = self.chattiori_path / "lora_bake.py"

    # ── device validation ────────────────────────────────────────────────

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

    # ── path helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_or_raise(path: Path, label: str) -> None:
        """Ensure *path* exists and is a file (or directory, depending)."""
        if not path.exists():
            raise NotFoundError(f"{label} not found: {path}")

    # ── merge command builder ────────────────────────────────────────────

    def _build_merge_command(
        self, request: CheckpointAdvancedMergeRequest
    ) -> list[str]:
        """Build the CLI argument list for merge.py from the request model."""
        command: list[str] = [
            sys.executable,
            str(self.merge_script),
            request.mode,
            request.model_path,
            request.model_0,
            request.model_1,
        ]

        # -- optional: model C (three-model modes)
        if request.model_2 is not None:
            command.append(request.model_2)

        # -- alpha / beta
        if request.alpha is not None:
            command.extend(["--alpha", request.alpha])
        if request.beta is not None:
            command.extend(["--beta", request.beta])

        # -- cosine structure-preference flags
        if request.cosine0:
            command.append("--cosine0")
        if request.cosine1:
            command.append("--cosine1")
        if request.cosine2:
            command.append("--cosine2")

        # -- VAE path
        if request.vae is not None:
            command.extend(["--vae", request.vae])

        # -- pruning / EMA
        if request.prune:
            command.append("--prune")
        if request.keep_ema:
            command.append("--keep_ema")

        # -- ReBasin iterations
        if request.rebasin is not None:
            command.extend(["--rebasin", str(request.rebasin)])

        # -- finetune key pattern
        if request.fine is not None:
            command.extend(["--fine", request.fine])

        # -- deterministic seed for stochastic modes
        if request.seed is not None:
            command.extend(["--seed", str(request.seed)])

        # -- metadata note
        if request.memo is not None:
            command.extend(["--memo", request.memo])

        # -- output format
        if request.save_safetensors:
            command.append("--save_safetensors")
        if request.save_half:
            command.append("--save_half")

        # -- device
        command.extend(["--device", request.device])

        # -- output name (filename without extension)
        command.extend(["--output", request.output])

        return command

    # ── bake command builder ─────────────────────────────────────────────

    def _build_bake_command(self, request: BakeRequest) -> list[str]:
        """
        Build the CLI argument list for lora_bake.py.

        lora_bake.py expects:
          python lora_bake.py <model_path> [checkpoint] [loras] [flags...]

        Where:
          - model_path   = parent directory of the checkpoint
          - checkpoint   = basename inside model_path (or None to auto-detect)
          - loras        = "path1:alpha1,path2:alpha2,..." (comma-separated)

        We pass the absolute directory as model_path and the basename of the
        checkpoint so the script can resolve sibling LoRA paths relative to it.
        """
        base_dir = Path(request.base_model_path).resolve().parent
        checkpoint_name = Path(request.base_model_path).name

        # Format LoRA paths with ratios: "path1:ratio1,path2:ratio2,..."
        ratios = request.lora_ratios if request.lora_ratios else [1.0] * len(request.lora_paths)
        loras_str = ",".join(
            f"{p}:{r}" for p, r in zip(request.lora_paths, ratios)
        )

        command: list[str] = [
            sys.executable,
            str(self.bake_script),
            str(base_dir),
            checkpoint_name,
            loras_str,
        ]

        # -- output format
        if request.save_safetensors:
            command.append("--save_safetensors")
        if request.save_half:
            command.append("--save_half")
        if request.prune:
            command.append("--prune")
        if request.keep_ema:
            command.append("--keep_ema")
        if request.memo is not None:
            command.extend(["--memo", request.memo])

        # -- bake-specific flags
        if request.bake_scale is not None:
            command.extend(["--bake_scale", str(request.bake_scale)])
        if request.bake_unet_only:
            command.append("--bake_unet_only")
        if request.bake_clip_scale is not None:
            command.extend(["--bake_clip_scale", str(request.bake_clip_scale)])

        command.extend(["--device", request.device])

        # Output: the script appends .safetensors / .ckpt automatically so we
        # strip the extension to match its --output convention.
        out_stem = Path(request.output_path).stem
        command.extend(["--output", out_stem])

        return command

    # ── public API: merge_checkpoints ────────────────────────────────────

    async def merge_checkpoints(
        self, request: CheckpointAdvancedMergeRequest
    ) -> str:
        """
        Start a Chattiori advanced checkpoint merge job.

        Validates all input paths and device, then launches merge.py as an
        async subprocess registered with the JobManager. Returns the job_id
        immediately.

        Raises:
            NotFoundError:  If any model file or the model directory is missing.
            ValidationError: If CUDA is requested but unavailable.

        Returns:
            str: The job_id that can be polled for status/logs.
        """
        # ── Early Exit: validate model directory exists ──────────────
        model_dir = Path(request.model_path)
        if not model_dir.is_dir():
            raise NotFoundError(
                f"Model directory not found: {request.model_path}"
            )

        # ── Early Exit: validate all model files exist in directory ──
        for field_name, filename in [
            ("model_0", request.model_0),
            ("model_1", request.model_1),
            ("model_2", request.model_2),
        ]:
            if filename is None:
                continue
            model_file = model_dir / filename
            if not model_file.is_file():
                raise NotFoundError(
                    f"Model file for {field_name} not found: "
                    f"{filename} in {request.model_path}"
                )

        # ── Fail Fast: guard clause for device availability ───────────
        self._validate_device(request.device)

        # ── Build command & launch subprocess ──────────────────────────
        command = self._build_merge_command(request)

        logger.info(
            "Starting Chattiori merge: mode=%s, models=%s/%s%s",
            request.mode,
            request.model_0,
            request.model_1,
            f"/{request.model_2}" if request.model_2 else "",
        )

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root,
        )

        job_id = job_manager.create_job(JobType.MERGE, process)
        logger.info("Chattiori merge job created: %s", job_id)
        return job_id

    # ── public API: bake_lora ──────────────────────────────────────────────

    async def bake_lora(self, request: BakeRequest) -> str:
        """
        Start a Chattiori LoRA bake job (bake LoRA(s) into a checkpoint).

        Validates input paths and device, then launches lora_bake.py as an
        async subprocess registered with the JobManager. Returns the job_id
        immediately.

        Args:
            request: Bake request with base checkpoint path and LoRA paths.

        Raises:
            NotFoundError:   If the base checkpoint or any LoRA file is missing.
            ValidationError: If CUDA is requested but unavailable.

        Returns:
            str(job_id): The job_id for status/log polling.
        """
        # ── Early Exit: validate base checkpoint ─────────────────────
        base_path = Path(request.base_model_path)
        if not base_path.is_file():
            raise NotFoundError(
                f"Base checkpoint not found: {request.base_model_path}"
            )

        # ── Early Exit: validate every LoRA file exists ──────────────
        for lora_path_str in request.lora_paths:
            lora_path = Path(lora_path_str)
            if not lora_path.is_file():
                raise NotFoundError(
                    f"LoRA file not found: {lora_path_str}"
                )

        # ── Fail Fast: device guard ───────────────────────────────────
        self._validate_device(request.device)

        # ── Build command & launch subprocess ──────────────────────────
        command = self._build_bake_command(request)

        logger.info(
            "Starting Chattiori bake: %d LoRA(s) → %s",
            len(request.lora_paths),
            request.base_model_path,
        )

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root,
        )

        job_id = job_manager.create_job(JobType.BAKE, process)
        logger.info("Chattiori bake job created: %s", job_id)
        return job_id

    # ── async helpers: wait & collect result ──────────────────────────────

    async def _monitor_and_collect_result(
        self, job_id: str
    ) -> CheckpointAdvancedMergeResponse:
        """
        Wait for a merge job to complete and collect the result.

        This polls the job store directly rather than going through an HTTP
        endpoint.  It is intended for callers that want a synchronous-style
        result instead of polling over WebSocket.

        NOTE: Typically the caller streams logs over WebSocket and reads the
        final job status via GET /jobs/{job_id}, making this helper optional.
        It is provided for convenience when the caller needs the response
        model directly.
        """
        # The job manager's _monitor_job handles completion; we just poll
        # until the job is no longer running, then build a response.
        raise NotImplementedError(
            "Direct result collection is not yet implemented. "
            "Use the job_id with GET /jobs/{job_id} or stream_logs() instead."
        )


# Global singleton instance
chattiori_service = ChattioriService()