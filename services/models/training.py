"""
Pydantic models for LoRA training configuration.

Matches frontend TrainingConfig interface from api.ts.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ModelType(str, Enum):
    """Supported model architectures."""
    SD15 = "SD1.5"
    SDXL = "SDXL"
    FLUX = "Flux"
    SD3 = "SD3"
    LUMINA = "Lumina"


class LoRAType(str, Enum):
    """LoRA network types (LyCORIS algorithms)."""
    # Standard LoRA variants
    LORA = "LoRA"
    LOCON = "LoCon"
    LOHA = "LoHa"
    LOKR = "LoKr"
    DORA = "DoRA"  # LoRA with weight decomposition

    # Advanced LyCORIS algorithms
    FULL = "Full"  # Native fine-tuning (DreamBooth)
    IA3 = "IA3"  # (IA)^3 - Infused Adapter by Inhibiting and Amplifying Inner Activations
    DYLORA = "DyLoRA"  # Dynamic LoRA with resizable rank
    GLORA = "GLoRA"  # Generalized LoRA
    DIAG_OFT = "Diag-OFT"  # Diagonal Orthogonal Finetuning
    BOFT = "BOFT"  # Butterfly OFT


class OptimizerType(str, Enum):
    """Available optimizers."""
    ADAMW = "AdamW"
    ADAMW8BIT = "AdamW8bit"
    LION = "Lion"
    LION8BIT = "Lion8bit"
    ADAFACTOR = "Adafactor"
    ADAFACTOR_CAMEL = "AdaFactor"
    DADAPTATION = "DAdaptation"
    DADAPTADAM = "DAdaptAdam"
    DADAPTADAGRAD = "DAdaptAdaGrad"
    DADAPTADAN = "DAdaptAdan"
    DADAPTSGD = "DAdaptSGD"
    PRODIGY = "Prodigy"
    CAME = "CAME"
    SGDNESTEROV = "SGDNesterov"
    SGDNESTEROV8BIT = "SGDNesterov8bit"


class LRScheduler(str, Enum):
    """Learning rate schedulers."""
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"


class MixedPrecision(str, Enum):
    """Mixed precision modes."""
    FP16 = "fp16"
    BF16 = "bf16"
    NO = "no"


class CrossAttention(str, Enum):
    """Cross attention implementations."""
    SDPA = "sdpa"
    XFORMERS = "xformers"
    NONE = "none"


class SaveModelAs(str, Enum):
    """Model save formats."""
    SAFETENSORS = "safetensors"
    CKPT = "ckpt"


class TrainingMode(str, Enum):
    """Training modes."""
    LORA = "lora"
    CHECKPOINT = "checkpoint"


class TrainingConfig(BaseModel):
    """
    Complete training configuration.

    Supports SD1.5, SDXL, Flux, SD3.5, and Lumina model types.
    Supports both LoRA and full checkpoint training.
    Matches frontend TrainingConfig interface.
    """

    # ========== PROJECT & MODEL SETUP ==========
    project_name: str = Field(..., description="Name of the training project")
    model_type: ModelType = Field(..., description="Model architecture type")
    training_mode: TrainingMode = Field(TrainingMode.LORA, description="Training mode: LoRA or full checkpoint")
    pretrained_model_name_or_path: str = Field(..., description="Path to pretrained model checkpoint")
    vae_path: Optional[str] = Field(None, description="Path to VAE model")

    # Conditional paths for Flux/SD3
    clip_l_path: Optional[str] = Field(None, description="Path to CLIP-L model (Flux/SD3)")
    clip_g_path: Optional[str] = Field(None, description="Path to CLIP-G model (Flux/SD3)")
    t5xxl_path: Optional[str] = Field(None, description="Path to T5-XXL model (Flux/SD3)")

    continue_from_lora: Optional[str] = Field(None, description="Path to LoRA to continue from")
    resume_from_state: Optional[str] = Field(None, description="Path to training state to resume from")
    wandb_key: Optional[str] = Field(None, description="W&B API key for logging")

    # ========== DATASET & BASIC TRAINING ==========
    train_data_dir: str = Field(..., description="Directory containing training data")
    output_dir: str = Field(..., description="Directory to save outputs")
    resolution: int = Field(..., ge=64, le=4096, description="Training resolution")
    num_repeats: int = Field(10, ge=1, description="Dataset repeat count")
    max_train_epochs: int = Field(10, ge=1, description="Maximum training epochs")
    max_train_steps: int = Field(0, ge=0, description="Max steps (0 = use epochs)")
    train_batch_size: int = Field(1, ge=1, description="Training batch size")
    seed: int = Field(42, description="Random seed for reproducibility")

    # Data augmentation
    flip_aug: bool = Field(False, description="Enable horizontal flipping")
    random_crop: bool = Field(False, description="Enable random cropping")
    color_aug: bool = Field(False, description="Enable color augmentation")
    shuffle_caption: bool = Field(False, description="Shuffle caption tokens")

    # ========== LEARNING RATES ==========
    unet_lr: float = Field(1e-4, gt=0, description="UNet learning rate")
    text_encoder_lr: float = Field(1e-5, gt=0, description="Text encoder learning rate")
    lr_scheduler: LRScheduler = Field(LRScheduler.COSINE, description="LR scheduler type")
    lr_scheduler_number: int = Field(1, ge=0, description="Scheduler-specific parameter")
    lr_warmup_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Warmup ratio")
    lr_warmup_steps: int = Field(0, ge=0, description="Warmup steps (0 = use ratio)")
    lr_power: float = Field(1.0, description="Power for polynomial scheduler")

    # ========== LORA STRUCTURE ==========
    lora_type: LoRAType = Field(LoRAType.LORA, description="LoRA type")
    network_module: str = Field("networks.lora", description="Network module")
    network_dim: int = Field(32, ge=1, le=1024, description="LoRA rank")
    network_alpha: int = Field(32, ge=1, description="LoRA alpha")
    conv_dim: int = Field(0, ge=0, description="Conv layer dimension (LoCon/LoHa)")
    conv_alpha: int = Field(0, ge=0, description="Conv alpha (LoCon/LoHa)")
    network_dropout: float = Field(0.0, ge=0.0, le=1.0, description="Network dropout")
    dim_from_weights: bool = Field(False, description="Derive dim from weights")
    factor: int = Field(-1, description="LoKR decomposition factor")
    train_norm: bool = Field(False, description="Train norm layers (LyCORIS)")

    # Advanced LyCORIS
    rank_dropout: float = Field(0.0, ge=0.0, le=1.0, description="LyCORIS rank dropout")
    module_dropout: float = Field(0.0, ge=0.0, le=1.0, description="LyCORIS module dropout")

    # Block-wise learning rates
    down_lr_weight: Optional[str] = Field(None, description="Down block LR weights")
    mid_lr_weight: Optional[str] = Field(None, description="Mid block LR weight")
    up_lr_weight: Optional[str] = Field(None, description="Up block LR weights")
    block_lr_zero_threshold: Optional[str] = Field(None, description="Block LR zero threshold")
    block_dims: Optional[str] = Field(None, description="Per-block dimensions")
    block_alphas: Optional[str] = Field(None, description="Per-block alphas")
    conv_block_dims: Optional[str] = Field(None, description="Per-block conv dims")
    conv_block_alphas: Optional[str] = Field(None, description="Per-block conv alphas")

    # ========== OPTIMIZER ==========
    optimizer_type: OptimizerType = Field(OptimizerType.ADAMW8BIT, description="Optimizer")
    weight_decay: float = Field(0.01, ge=0.0, description="Weight decay")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Gradient accumulation")
    max_grad_norm: float = Field(1.0, gt=0, description="Max gradient norm")
    optimizer_args: Optional[str] = Field(None, description="Custom optimizer args JSON")

    # ========== CAPTION & TOKEN CONTROL ==========
    keep_tokens: int = Field(0, ge=0, description="Tokens to keep (no dropout)")
    clip_skip: int = Field(2, ge=1, le=12, description="CLIP layers to skip")
    max_token_length: int = Field(225, ge=75, description="Max token length")
    caption_dropout_rate: float = Field(0.0, ge=0.0, le=1.0, description="Caption dropout")
    caption_tag_dropout_rate: float = Field(0.0, ge=0.0, le=1.0, description="Tag dropout")
    caption_dropout_every_n_epochs: int = Field(0, ge=0, description="Dropout frequency")
    keep_tokens_separator: str = Field("|||", description="Keep tokens separator")
    secondary_separator: str = Field("", description="Secondary separator")
    enable_wildcard: bool = Field(False, description="Enable wildcard expansion")
    weighted_captions: bool = Field(False, description="Enable weighted captions")

    # ========== BUCKETING ==========
    enable_bucket: bool = Field(True, description="Enable bucketing")
    sdxl_bucket_optimization: bool = Field(False, description="SDXL bucket optimization")
    min_bucket_reso: int = Field(256, ge=64, description="Min bucket resolution")
    max_bucket_reso: int = Field(2048, le=4096, description="Max bucket resolution")
    bucket_no_upscale: bool = Field(False, description="No upscaling in buckets")
    bucket_reso_steps: Optional[int] = Field(None, ge=1, description="Bucket resolution steps")

    # ========== ADVANCED TRAINING - SNR & NOISE ==========
    min_snr_gamma_enabled: bool = Field(False, description="Enable min SNR gamma")
    min_snr_gamma: float = Field(5.0, ge=0, description="Min SNR gamma value")
    ip_noise_gamma_enabled: bool = Field(False, description="Enable input perturbation")
    ip_noise_gamma: float = Field(0.0, ge=0, description="Input perturbation noise")
    ip_noise_gamma_random_strength: Optional[bool] = Field(None, description="Randomize IP noise strength")
    multinoise: bool = Field(False, description="Enable multi-res noise")
    multires_noise_discount: float = Field(0.0, ge=0, description="Multi-res noise discount")
    multires_noise_iterations: Optional[int] = Field(None, ge=0, description="Multi-res noise iterations")
    noise_offset: float = Field(0.0, description="Noise offset")
    adaptive_noise_scale: float = Field(0.0, ge=0, description="Adaptive noise scale")
    zero_terminal_snr: bool = Field(False, description="Zero terminal SNR")
    min_timestep: Optional[int] = Field(None, ge=0, le=1000, description="Minimum timestep for noise schedule")
    max_timestep: Optional[int] = Field(None, ge=0, le=1000, description="Maximum timestep for noise schedule")

    # ========== MEMORY & PERFORMANCE ==========
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    mixed_precision: MixedPrecision = Field(MixedPrecision.FP16, description="Mixed precision")
    full_fp16: bool = Field(False, description="Full FP16 training")
    full_bf16: Optional[bool] = Field(None, description="Full BF16 training")
    fp8_base: bool = Field(False, description="FP8 base model (experimental)")
    vae_batch_size: int = Field(1, ge=1, description="VAE batch size")
    no_half_vae: bool = Field(False, description="Disable half VAE")
    cache_latents: bool = Field(True, description="Cache latents")
    cache_latents_to_disk: bool = Field(False, description="Cache latents to disk")
    cache_text_encoder_outputs: bool = Field(False, description="Cache text encoder outputs")
    cache_text_encoder_outputs_to_disk: Optional[bool] = Field(None, description="Cache text encoder outputs to disk")
    cross_attention: CrossAttention = Field(CrossAttention.SDPA, description="Cross attention")
    persistent_data_loader_workers: int = Field(0, ge=0, description="Persistent workers")
    max_data_loader_n_workers: Optional[int] = Field(None, ge=0, description="Max data loader workers")
    no_token_padding: bool = Field(False, description="Disable token padding")
    lowram: bool = Field(False, description="Low RAM mode")

    # ========== SAVING & CHECKPOINTS ==========
    save_every_n_epochs: int = Field(1, ge=0, description="Save every N epochs")
    save_every_n_steps: int = Field(0, ge=0, description="Save every N steps")
    save_last_n_epochs: int = Field(0, ge=0, description="Keep last N epochs")
    save_last_n_epochs_state: int = Field(0, ge=0, description="Keep last N epoch states")
    save_state: bool = Field(False, description="Save training state")
    save_state_on_train_end: Optional[bool] = Field(None, description="Save state when training completes")
    save_last_n_steps_state: int = Field(0, ge=0, description="Keep last N step states")
    save_model_as: SaveModelAs = Field(SaveModelAs.SAFETENSORS, description="Model format")
    save_precision: str = Field("fp16", description="Save precision")
    output_name: str = Field("lora", description="Output file name")
    no_metadata: bool = Field(False, description="Skip metadata")

    # ========== SAMPLE GENERATION ==========
    sample_every_n_epochs: int = Field(0, ge=0, description="Generate samples every N epochs")
    sample_every_n_steps: int = Field(0, ge=0, description="Generate samples every N steps")
    sample_prompts: Optional[str] = Field(None, description="Path to sample prompts file")
    sample_sampler: str = Field("euler_a", description="Sample generation sampler")

    # ========== LOGGING ==========
    logging_dir: Optional[str] = Field(None, description="Logging directory")
    log_with: Optional[str] = Field(None, description="Logging backend (tensorboard/wandb)")
    log_prefix: Optional[str] = Field(None, description="Log file prefix")
    log_tracker_name: Optional[str] = Field(None, description="Tracker name for logging")
    log_tracker_config: Optional[str] = Field(None, description="Tracker config (JSON string)")
    wandb_run_name: Optional[str] = Field(None, description="W&B run name")

    # ========== SD 2.x & ADVANCED ==========
    v2: bool = Field(False, description="SD 2.x base model flag")
    v_parameterization: bool = Field(False, description="V-parameterization")
    scale_v_pred_loss_like_noise_pred: Optional[bool] = Field(None, description="Scale v-pred loss like noise pred")
    v_pred_like_loss: Optional[float] = Field(None, ge=0, description="V-prediction-like loss weight")
    debiased_estimation_loss: Optional[bool] = Field(None, description="Use debiased estimation loss")
    network_train_unet_only: bool = Field(True, description="Train UNet only")
    prior_loss_weight: float = Field(1.0, ge=0, description="Prior loss weight")

    # ========== FLUX-SPECIFIC ==========
    ae_path: Optional[str] = Field(None, description="Flux AutoEncoder path")
    t5xxl_max_token_length: Optional[int] = Field(None, ge=75, description="T5-XXL max tokens")
    apply_t5_attn_mask: bool = Field(False, description="Apply T5-XXL attention mask")
    guidance_scale: float = Field(1.0, ge=0, description="Guidance scale (Flux.1 dev)")
    timestep_sampling: str = Field("sigma", description="Timestep sampling method")
    sigmoid_scale: float = Field(1.0, description="Sigmoid timestep scale")
    model_prediction_type: str = Field("raw", description="Model prediction type")
    blocks_to_swap: Optional[int] = Field(None, ge=0, description="Blocks to swap (memory)")

    # ========== LUMINA-SPECIFIC ==========
    gemma2: Optional[str] = Field(None, description="Path to Gemma2 model")
    gemma2_max_token_length: Optional[int] = Field(256, ge=75, description="Gemma2 max tokens")

    @field_validator('model_type', mode='before')
    @classmethod
    def normalize_model_type(cls, v):
        """Allow flexible model type input with better error handling."""
        # If already a ModelType enum, return as-is
        if isinstance(v, ModelType):
            return v

        # Handle string inputs
        if isinstance(v, str):
            # Normalize common variations
            v_upper = v.upper().strip()

            # SD1.5 variations
            if v_upper in {"SD1.5", "SD15", "SD_1.5", "SD-1.5", "SD 1.5"}:
                return "SD1.5"  # Return the actual enum value
            # SDXL variations
            elif v_upper in {"SDXL", "SD_XL", "SD-XL", "SD XL"}:
                return "SDXL"
            # FLUX variations
            elif v_upper in {"FLUX", "FLUX.1", "FLUX-1"}:
                return "Flux"
            # SD3 variations
            elif v_upper in {"SD3", "SD3.5", "SD_3", "SD-3", "SD 3"}:
                return "SD3"
            # LUMINA variations
            elif v_upper == "LUMINA":
                return "Lumina"

        # If we can't normalize, return the original value
        # Pydantic will handle validation errors appropriately
        return v

    class Config:
        use_enum_values = True  # Store enum values as strings in dict


class TrainingStartRequest(BaseModel):
    """Request to start training."""
    config: TrainingConfig


class TrainingStartResponse(BaseModel):
    """Response from starting training."""
    success: bool
    message: str
    job_id: Optional[str] = None
    validation_errors: list[dict] = Field(default_factory=list)


class TrainingStatusResponse(BaseModel):
    """Training status response."""
    job_id: str
    status: str
    progress: int = Field(ge=0, le=100)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_step: Optional[str] = None
    error: Optional[str] = None
