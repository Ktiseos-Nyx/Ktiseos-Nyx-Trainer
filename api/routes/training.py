"""
Training API Routes
Handles training start/stop, status monitoring, and log streaming.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import logging
from pathlib import Path

from shared_managers import get_training_manager
from core.log_streamer import get_training_log_streamer

logger = logging.getLogger(__name__)
router = APIRouter()

# Get global log streamer
log_streamer = get_training_log_streamer()


class TrainingConfig(BaseModel):
    """
    Complete training configuration parameters.
    Matches Jupyter notebook training_widget.py functionality.
    """

    # ========== PROJECT & MODEL SETUP ==========
    project_name: str = "my_lora"
    model_type: str = "SDXL"  # SD1.5, SDXL, Flux, SD3
    pretrained_model_name_or_path: str
    vae_path: Optional[str] = None

    # Conditional paths for Flux/SD3
    clip_l_path: Optional[str] = None
    clip_g_path: Optional[str] = None
    t5xxl_path: Optional[str] = None

    continue_from_lora: Optional[str] = None
    wandb_key: Optional[str] = None

    # ========== DATASET & BASIC TRAINING ==========
    train_data_dir: str
    output_dir: str
    resolution: int = 1024
    num_repeats: int = 10  # CRITICAL for Kohya!
    max_train_epochs: int = 10
    max_train_steps: int = 0  # 0 = use epochs instead
    train_batch_size: int = 4
    seed: int = 42

    # Data augmentation
    flip_aug: bool = False
    random_crop: bool = False
    color_aug: bool = False  # Color augmentation
    shuffle_caption: bool = True

    # ========== LEARNING RATES ==========
    unet_lr: float = 5e-4
    text_encoder_lr: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_scheduler_number: int = 3  # For restarts/polynomial
    lr_warmup_ratio: float = 0.05
    lr_warmup_steps: int = 0  # 0 = use ratio
    lr_power: float = 1.0  # For polynomial scheduler

    # ========== LORA STRUCTURE ==========
    lora_type: str = "LoRA"  # LoRA, LoCon, LoKR, DyLoRA, DoRA, LoHa, (IA)³, GLoRA, Native Fine-Tuning, Diag-OFT, BOFT
    network_module: str = "networks.lora"
    network_dim: int = 16
    network_alpha: int = 8
    conv_dim: int = 16  # For textures/details
    conv_alpha: int = 8
    network_dropout: float = 0.0
    dim_from_weights: bool = False
    factor: int = -1  # LoKR decomposition, -1=auto
    train_norm: bool = False  # LyCORIS

    # Advanced LyCORIS parameters
    rank_dropout: float = 0.0  # LyCORIS rank dropout
    module_dropout: float = 0.0  # LyCORIS module dropout

    # Block-wise learning rates (advanced)
    down_lr_weight: Optional[str] = None  # e.g., "1,1,1,1,1,1,1,1,1,1,1,1"
    mid_lr_weight: Optional[str] = None  # e.g., "1"
    up_lr_weight: Optional[str] = None  # e.g., "1,1,1,1,1,1,1,1,1,1,1,1"
    block_lr_zero_threshold: Optional[str] = None  # e.g., "0.1"
    block_dims: Optional[str] = None  # Per-block dimensions
    block_alphas: Optional[str] = None  # Per-block alphas
    conv_block_dims: Optional[str] = None  # Per-block conv dimensions
    conv_block_alphas: Optional[str] = None  # Per-block conv alphas

    # ========== OPTIMIZER ==========
    optimizer_type: str = "AdamW8bit"
    weight_decay: float = 0.01  # Weight decay for optimizer
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer_args: Optional[str] = None  # JSON string for custom args

    # ========== CAPTION & TOKEN CONTROL ==========
    keep_tokens: int = 0
    clip_skip: int = 2
    max_token_length: int = 75  # Maximum token length
    caption_dropout_rate: float = 0.0
    caption_tag_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    keep_tokens_separator: str = ""
    secondary_separator: str = ""
    enable_wildcard: bool = False
    weighted_captions: bool = False  # Enable weighted captions

    # ========== BUCKETING ==========
    enable_bucket: bool = True
    sdxl_bucket_optimization: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 2048
    bucket_no_upscale: bool = False

    # ========== ADVANCED TRAINING ==========
    # SNR & Noise
    min_snr_gamma_enabled: bool = True
    min_snr_gamma: float = 5.0
    ip_noise_gamma_enabled: bool = False
    ip_noise_gamma: float = 0.05
    multinoise: bool = False
    multires_noise_discount: float = 0.25
    noise_offset: float = 0.0
    adaptive_noise_scale: float = 0.0
    zero_terminal_snr: bool = False

    # ========== MEMORY & PERFORMANCE ==========
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    full_fp16: bool = False
    fp8_base: bool = False  # FP8 base (experimental)
    vae_batch_size: int = 1
    no_half_vae: bool = False
    cache_latents: bool = True  # Changed to True to match Jupyter default
    cache_latents_to_disk: bool = True  # Changed to True to match Jupyter default
    cache_text_encoder_outputs: bool = False
    cross_attention: str = "sdpa"  # sdpa or xformers
    persistent_data_loader_workers: int = 0  # 0=auto
    no_token_padding: bool = False  # Memory optimization

    # ========== SAVING & CHECKPOINTS ==========
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 0
    save_last_n_epochs: int = 0
    save_last_n_epochs_state: int = 0
    save_state: bool = False  # Save training state for resuming
    save_last_n_steps_state: int = 0  # Save last N steps state
    save_model_as: str = "safetensors"  # safetensors, ckpt, pt, diffusers
    save_precision: str = "fp16"
    output_name: str = ""
    no_metadata: bool = False

    # ========== SAMPLE GENERATION ==========
    sample_every_n_epochs: int = 0
    sample_every_n_steps: int = 0
    sample_prompts: Optional[str] = None
    sample_sampler: str = "euler_a"

    # ========== LOGGING ==========
    logging_dir: Optional[str] = None
    log_with: Optional[str] = None  # tensorboard, wandb
    log_prefix: Optional[str] = None

    # ========== SD 2.x & ADVANCED ==========
    v2: bool = False  # SD 2.x base model flag
    v_parameterization: bool = False  # For SDXL v-pred or SD 2.x 768px
    network_train_unet_only: bool = False  # Train U-Net only (recommended for SDXL)
    prior_loss_weight: float = 1.0  # Prior loss weight

    # ========== FLUX-SPECIFIC PARAMETERS ==========
    # Flux model paths (required when model_type="Flux")
    ae_path: Optional[str] = None  # Flux AutoEncoder path (*.safetensors)
    t5xxl_max_token_length: Optional[int] = None  # Max tokens for T5-XXL (256 for schnell, 512 for dev)
    apply_t5_attn_mask: bool = False  # Apply attention mask to T5-XXL
    guidance_scale: float = 3.5  # Guidance scale for Flux.1 dev
    timestep_sampling: str = "sigma"  # sigma, uniform, sigmoid, shift, flux_shift
    sigmoid_scale: float = 1.0  # Scale for sigmoid timestep sampling
    model_prediction_type: str = "raw"  # raw or additive for dev model
    blocks_to_swap: Optional[int] = None  # Number of blocks to swap (memory optimization)

    # ========== LUMINA-SPECIFIC PARAMETERS ==========
    # Lumina model paths (required when model_type="Lumina")
    gemma2: Optional[str] = None  # Path to Gemma2 model (*.sft or *.safetensors), should be float16
    gemma2_max_token_length: Optional[int] = None  # Maximum token length for Gemma2. Default: 256
    # ae_path is shared with Flux (AutoEncoder path)
    # timestep_sampling is shared with Flux (sigma, uniform, sigmoid, shift, nextdit_shift)
    # sigmoid_scale is shared with Flux
    # blocks_to_swap is shared with Flux


class ValidationError(BaseModel):
    """Structured validation error with field reference"""
    field: str  # Field name that has the issue
    message: str  # Human-readable error message
    severity: str = "warning"  # warning, error, info


def validate_training_config(config: TrainingConfig) -> List[ValidationError]:
    """
    Validate training configuration and return list of validation errors.
    Checks for conflicts like in Jupyter notebook.
    Returns structured errors with field references for better UX.
    """
    errors = []

    # Text encoder caching vs shuffle caption conflict
    if config.cache_text_encoder_outputs and config.shuffle_caption:
        errors.append(ValidationError(
            field="shuffle_caption",
            message="Cannot use Caption Shuffling with Text Encoder Caching enabled. Disable one of these options.",
            severity="error"
        ))

    # Text encoder caching vs text encoder training conflict
    if config.cache_text_encoder_outputs and config.text_encoder_lr > 0:
        errors.append(ValidationError(
            field="text_encoder_lr",
            message="Cannot cache Text Encoder outputs while training it. Set Text Encoder LR to 0 or disable caching.",
            severity="error"
        ))

    # Random crop vs latent caching conflict
    if config.random_crop and config.cache_latents:
        errors.append(ValidationError(
            field="random_crop",
            message="Cannot use Random Crop with Latent Caching. Choose one or the other.",
            severity="error"
        ))

    # Check for zero learning rates
    if config.unet_lr <= 0 and config.text_encoder_lr <= 0:
        errors.append(ValidationError(
            field="unet_lr",
            message="Both UNet LR and Text Encoder LR are 0 - nothing will be trained! Set at least one to a positive value.",
            severity="error"
        ))

    # Check batch size
    if config.train_batch_size <= 0:
        errors.append(ValidationError(
            field="train_batch_size",
            message="Batch size must be greater than 0",
            severity="error"
        ))

    # Check resolution
    if config.resolution < 256:
        errors.append(ValidationError(
            field="resolution",
            message=f"Resolution {config.resolution} is too low. Minimum recommended is 256.",
            severity="warning"
        ))
    elif config.resolution > 4096:
        errors.append(ValidationError(
            field="resolution",
            message=f"Resolution {config.resolution} is very high. Maximum recommended is 4096.",
            severity="warning"
        ))

    # Check epochs/steps
    if config.max_train_epochs <= 0 and config.max_train_steps <= 0:
        errors.append(ValidationError(
            field="max_train_epochs",
            message="Must specify either max_train_epochs or max_train_steps (or both)",
            severity="error"
        ))

    # Check dataset directory
    if not config.train_data_dir or config.train_data_dir.strip() == "":
        errors.append(ValidationError(
            field="train_data_dir",
            message="Dataset directory is required",
            severity="error"
        ))

    # Check output directory
    if not config.output_dir or config.output_dir.strip() == "":
        errors.append(ValidationError(
            field="output_dir",
            message="Output directory is required",
            severity="error"
        ))

    # Check project name
    if not config.project_name or config.project_name.strip() == "":
        errors.append(ValidationError(
            field="project_name",
            message="Project name is required",
            severity="error"
        ))

    # Helpful warnings for common mistakes
    if config.network_dim > 128:
        errors.append(ValidationError(
            field="network_dim",
            message=f"Network dim {config.network_dim} is very high. Typical values are 4-64. High values increase training time and file size.",
            severity="info"
        ))

    if config.unet_lr > 1e-3:
        errors.append(ValidationError(
            field="unet_lr",
            message=f"UNet learning rate {config.unet_lr} is very high. Typical SDXL values are 1e-4 to 5e-4.",
            severity="warning"
        ))

    # Flux-specific validation
    if config.model_type == "Flux":
        # Check required Flux paths
        if not config.clip_l_path or config.clip_l_path.strip() == "":
            errors.append(ValidationError(
                field="clip_l_path",
                message="CLIP-L path is required for Flux training",
                severity="error"
            ))

        if not config.t5xxl_path or config.t5xxl_path.strip() == "":
            errors.append(ValidationError(
                field="t5xxl_path",
                message="T5-XXL path is required for Flux training",
                severity="error"
            ))

        # VRAM warning for Flux
        if config.train_batch_size > 2:
            errors.append(ValidationError(
                field="train_batch_size",
                message="Flux requires significant VRAM. Batch size > 2 may cause OOM errors. Recommended: 1-2 with 24GB VRAM.",
                severity="warning"
            ))

        # Guidance for blocks_to_swap
        if config.blocks_to_swap is None and config.train_batch_size > 1:
            errors.append(ValidationError(
                field="blocks_to_swap",
                message="Consider setting blocks_to_swap (e.g., 18) to reduce VRAM usage. Recommended for GPUs with <48GB VRAM.",
                severity="info"
            ))

    # Lumina-specific validation
    if config.model_type == "Lumina":
        # Check required Lumina paths
        if not config.gemma2 or config.gemma2.strip() == "":
            errors.append(ValidationError(
                field="gemma2",
                message="Gemma2 model path is required for Lumina training (*.sft or *.safetensors)",
                severity="error"
            ))

        if not config.ae_path or config.ae_path.strip() == "":
            errors.append(ValidationError(
                field="ae_path",
                message="AutoEncoder path is required for Lumina training",
                severity="error"
            ))

        # VRAM warning for Lumina (similar to Flux)
        if config.train_batch_size > 2:
            errors.append(ValidationError(
                field="train_batch_size",
                message="Lumina requires significant VRAM. Batch size > 2 may cause OOM errors. Recommended: 1-2 with 24GB VRAM.",
                severity="warning"
            ))

        # Guidance for blocks_to_swap
        if config.blocks_to_swap is None and config.train_batch_size > 1:
            errors.append(ValidationError(
                field="blocks_to_swap",
                message="Consider setting blocks_to_swap to reduce VRAM usage. Recommended for GPUs with <48GB VRAM.",
                severity="info"
            ))

    return errors


class TrainingStartResponse(BaseModel):
    """Response when training starts"""
    success: bool
    message: str
    training_id: Optional[str] = None
    warnings: List[str] = []  # Deprecated, kept for backwards compatibility
    validation_errors: List[ValidationError] = []  # New structured errors


class TrainingStatusResponse(BaseModel):
    """Training status response"""
    is_training: bool
    progress: Optional[Dict[str, Any]] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(config: TrainingConfig):
    """
    Start a new training session.
    Returns immediately with training ID for monitoring via WebSocket.
    Includes config validation warnings.
    """
    try:
        # Validate configuration
        validation_errors = validate_training_config(config)

        # Check for critical errors (severity="error")
        critical_errors = [e for e in validation_errors if e.severity == "error"]
        if critical_errors:
            # Return validation errors without starting training
            error_messages = [f"{e.field}: {e.message}" for e in critical_errors]
            return TrainingStartResponse(
                success=False,
                message=f"Configuration has {len(critical_errors)} error(s). Please fix them before starting training.",
                validation_errors=validation_errors,
                warnings=[e.message for e in validation_errors]  # Backwards compat
            )

        training_manager = get_training_manager()

        # Convert config to dict for training manager
        config_dict = config.dict()

        # Start training (launches subprocess, returns immediately)
        logger.info(f"Starting training: {config.project_name}")

        success = training_manager.start_training(config_dict, monitor_widget=None)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Training failed to start"
            )

        training_id = f"train_{config.project_name}"

        # Return with warnings/info messages (non-critical)
        return TrainingStartResponse(
            success=True,
            message="Training started successfully",
            training_id=training_id,
            validation_errors=validation_errors,
            warnings=[e.message for e in validation_errors]  # Backwards compat
        )

    except Exception as e:
        logger.error(f"Failed to start training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_config(config: TrainingConfig):
    """
    Validate training configuration without starting training.
    Returns list of warnings/conflicts.
    """
    warnings = validate_training_config(config)
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings
    }


@router.post("/stop")
async def stop_training():
    """Stop the current training session"""
    try:
        training_manager = get_training_manager()

        logger.info("Stopping training...")
        training_manager.stop_training()

        return {"success": True, "message": "Training stopped"}

    except Exception as e:
        logger.error(f"Failed to stop training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get current training status and progress"""
    try:
        from shared_managers import get_config_manager
        config_manager = get_config_manager()
        status = config_manager.get_training_status()

        # Check if training process is running by checking config files
        is_training = status.get("ready", False)

        return TrainingStatusResponse(
            is_training=is_training,
            progress=None,  # TODO: Parse from logs if needed
            current_step=None,
            total_steps=None,
            current_epoch=None,
            total_epochs=None
        )

    except Exception as e:
        logger.error(f"Failed to get training status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/logs")
async def training_logs_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training logs.
    Streams logs from training subprocess via log_streamer.
    """
    await websocket.accept()
    log_streamer.add_connection(websocket)

    try:
        logger.info("Client connected to training logs WebSocket")

        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "✓ Connected to training logs"
        })

        # Broadcast pending logs every 0.1 seconds
        while True:
            await log_streamer.broadcast_pending()
            await asyncio.sleep(0.1)  # 10 updates per second

    except WebSocketDisconnect:
        logger.info("Client disconnected from training logs WebSocket")
        log_streamer.remove_connection(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        log_streamer.remove_connection(websocket)


async def broadcast_log(message: Dict[str, Any]):
    """
    Broadcast log message to all connected WebSocket clients.
    Call this from training manager when new logs are available.
    """
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            disconnected.append(connection)

    # Clean up disconnected clients
    for conn in disconnected:
        active_connections.remove(conn)


@router.get("/history")
async def get_training_history():
    """Get list of past training sessions"""
    try:
        # TODO: Implement training history from output directory
        output_dir = Path("/workspace/output")

        if not output_dir.exists():
            return {"trainings": []}

        trainings = []
        for item in output_dir.iterdir():
            if item.is_dir():
                trainings.append({
                    "name": item.name,
                    "path": str(item),
                    "created": item.stat().st_ctime
                })

        trainings.sort(key=lambda x: x["created"], reverse=True)
        return {"trainings": trainings}

    except Exception as e:
        logger.error(f"Failed to get training history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
