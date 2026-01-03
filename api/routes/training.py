"""
Training API Routes
Handles training start/stop, status monitoring via new service layer.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import new service layer
from services import training_service
from services.models.training import TrainingConfig

logger = logging.getLogger(__name__)
router = APIRouter()

# Project root detection
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_WORKSPACE = PROJECT_ROOT


class ValidationError(BaseModel):
    """Structured validation error with field reference"""

    field: str
    message: str
    severity: str = "warning"


def validate_training_config_extended(config: TrainingConfig) -> list[ValidationError]:
    """
    Additional validation beyond Pydantic.
    Checks for conflicts like in original Jupyter notebook.
    """
    errors = []

    # Text encoder caching vs shuffle caption conflict
    if config.cache_text_encoder_outputs and config.shuffle_caption:
        errors.append(
            ValidationError(
                field="shuffle_caption",
                message="Cannot use Caption Shuffling with Text Encoder Caching enabled.",
                severity="error",
            )
        )

    # Text encoder caching vs text encoder training conflict
    if config.cache_text_encoder_outputs and config.text_encoder_lr > 0:
        errors.append(
            ValidationError(
                field="text_encoder_lr",
                message="Cannot cache Text Encoder outputs while training it. Set Text Encoder LR to 0 or disable caching.",
                severity="error",
            )
        )

    # Random crop vs latent caching conflict
    if config.random_crop and config.cache_latents:
        errors.append(
            ValidationError(
                field="random_crop",
                message="Cannot use Random Crop with Latent Caching.",
                severity="error",
            )
        )

    # Check for zero learning rates
    if config.unet_lr <= 0 and config.text_encoder_lr <= 0:
        errors.append(
            ValidationError(
                field="unet_lr",
                message="Both UNet LR and Text Encoder LR are 0 - nothing will be trained!",
                severity="error",
            )
        )

    # Helpful warnings for common mistakes
    if config.network_dim > 128:
        errors.append(
            ValidationError(
                field="network_dim",
                message=f"Network dim {config.network_dim} is very high. Typical values are 4-64.",
                severity="info",
            )
        )

    if config.unet_lr > 1e-3:
        errors.append(
            ValidationError(
                field="unet_lr",
                message=f"UNet learning rate {config.unet_lr} is very high. Typical SDXL values are 1e-4 to 5e-4.",
                severity="warning",
            )
        )

    # Flux-specific validation
    if config.model_type == "Flux":
        if not config.clip_l_path:
            errors.append(
                ValidationError(
                    field="clip_l_path",
                    message="CLIP-L path is required for Flux training",
                    severity="error",
                )
            )

        if not config.t5xxl_path:
            errors.append(
                ValidationError(
                    field="t5xxl_path",
                    message="T5-XXL path is required for Flux training",
                    severity="error",
                )
            )

        if config.train_batch_size > 2:
            errors.append(
                ValidationError(
                    field="train_batch_size",
                    message="Flux requires significant VRAM. Batch size > 2 may cause OOM errors.",
                    severity="warning",
                )
            )

        if config.blocks_to_swap is None and config.train_batch_size > 1:
            errors.append(
                ValidationError(
                    field="blocks_to_swap",
                    message="Consider setting blocks_to_swap (e.g., 18) to reduce VRAM usage.",
                    severity="info",
                )
            )

    # Lumina-specific validation
    if config.model_type == "Lumina":
        if not config.gemma2:
            errors.append(
                ValidationError(
                    field="gemma2",
                    message="Gemma2 model path is required for Lumina training",
                    severity="error",
                )
            )

        if not config.ae_path:
            errors.append(
                ValidationError(
                    field="ae_path",
                    message="AutoEncoder path is required for Lumina training",
                    severity="error",
                )
            )

    return errors


class TrainingStartResponse(BaseModel):
    """Response when training starts"""

    success: bool
    message: str
    job_id: Optional[str] = None  # Changed from training_id to job_id
    validation_errors: list[ValidationError] = []


class TrainingStatusResponse(BaseModel):
    """Training status response"""

    job_id: str
    status: str
    progress: int = 0
    current_step: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    error: Optional[str] = None


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(config: TrainingConfig):
    """
    Start a new training session.
    Returns immediately with job_id for monitoring via WebSocket.
    """
    try:
        # Additional validation beyond Pydantic
        validation_errors = validate_training_config_extended(config)

        # Check for critical errors
        critical_errors = [e for e in validation_errors if e.severity == "error"]
        if critical_errors:
            return TrainingStartResponse(
                success=False,
                message=f"Configuration has {len(critical_errors)} error(s).",
                validation_errors=validation_errors,
            )

        logger.info("Starting training: %s", config.project_name)

        # Use new service layer
        response = await training_service.start_training(config)

        # Add our validation warnings/info to the response
        if validation_errors and response.success:
            response.validation_errors.extend(
                [
                    {"field": e.field, "message": e.message, "severity": e.severity}
                    for e in validation_errors
                ]
            )

        return TrainingStartResponse(
            success=response.success,
            message=response.message,
            job_id=response.job_id,
            validation_errors=validation_errors,
        )

    except Exception as e:
        logger.error("Failed to start training: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_config(config: TrainingConfig):
    """
    Validate training configuration without starting training.
    Returns list of warnings/conflicts.
    """
    errors = validate_training_config_extended(config)
    return {
        "valid": len([e for e in errors if e.severity == "error"]) == 0,
        "warnings": errors,
    }


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """Get current training status and progress"""
    try:
        status = await training_service.get_status(job_id)

        if not status:
            raise HTTPException(status_code=404, detail="Training job not found")

        return TrainingStatusResponse(
            job_id=status.job_id,
            status=status.status.value,
            progress=status.progress,
            current_step=status.current_step,
            current_epoch=status.current_epoch,
            total_epochs=status.total_epochs,
            error=status.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get training status: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{job_id}")
async def stop_training(job_id: str):
    """Stop a running training job"""
    try:
        logger.info("Stopping training: %s", job_id)
        stopped = await training_service.stop_training(job_id)

        if not stopped:
            raise HTTPException(
                status_code=404, detail="Training job not found or already stopped"
            )

        return {"success": True, "message": f"Training {job_id} stopped"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to stop training: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_training_history():
    """Get list of past training sessions from output directory"""
    try:
        output_dir = DEFAULT_WORKSPACE / "output"

        if not output_dir.exists():
            return {"trainings": []}

        trainings = []
        for item in output_dir.iterdir():
            if item.is_dir():
                trainings.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "created": item.stat().st_ctime,
                    }
                )

        trainings.sort(key=lambda x: x["created"], reverse=True)
        return {"trainings": trainings}

    except Exception as e:
        logger.error("Failed to get training history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: WebSocket routes are now in services/websocket.py
# Use: ws://host/ws/jobs/{job_id}/logs for real-time log streaming
# Use: ws://host/ws/jobs/{job_id}/status for real-time status updates
