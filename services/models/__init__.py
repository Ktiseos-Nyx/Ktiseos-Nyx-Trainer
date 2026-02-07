"""
Pydantic models for API request/response validation.

Models are organized by domain:
- common: Shared models (pagination, errors)
- job: Job status and tracking
- training: Training configuration
- tagging: WD14 tagging requests
- dataset: Dataset operations
- caption: Caption editing
- lora: LoRA utilities (resize, upload)
"""

from .common import PaginatedResponse, ErrorResponse
from .job import JobStatus, JobType, JobStatusEnum
from .training import (
    TrainingConfig,
    TrainingStartRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
    ModelType,
    LoRAType,
    OptimizerType,
    LRScheduler,
    MixedPrecision,
    CrossAttention,
    SaveModelAs,
)
from .tagging import (
    TaggingConfig,
    TaggingStartResponse,
    TaggingStatusResponse,
    TaggerModel,
)
from .dataset import (
    DatasetInfo,
    FileInfo,
    CreateDatasetRequest,
    UploadImageRequest,
    DatasetListResponse,
    DatasetFilesResponse,
)
from .caption import (
    CaptionFile,
    AddTriggerWordRequest,
    RemoveTagsRequest,
    ReplaceTextRequest,
    ReadCaptionRequest,
    WriteCaptionRequest,
    CaptionOperationResponse,
    CaptionReadResponse,
)
from .lora import (
    LoRAResizeRequest,
    LoRAResizeResponse,
    HuggingFaceUploadRequest,
    HuggingFaceUploadResponse,
)

__all__ = [
    # Common
    "PaginatedResponse",
    "ErrorResponse",
    # Job
    "JobStatus",
    "JobType",
    "JobStatusEnum",
    # Training
    "TrainingConfig",
    "TrainingStartRequest",
    "TrainingStartResponse",
    "TrainingStatusResponse",
    "ModelType",
    "LoRAType",
    "OptimizerType",
    "LRScheduler",
    "MixedPrecision",
    "CrossAttention",
    "SaveModelAs",
    # Tagging
    "TaggingConfig",
    "TaggingStartResponse",
    "TaggingStatusResponse",
    "TaggerModel",
    # Dataset
    "DatasetInfo",
    "FileInfo",
    "CreateDatasetRequest",
    "UploadImageRequest",
    "DatasetListResponse",
    "DatasetFilesResponse",
    # Caption
    "CaptionFile",
    "AddTriggerWordRequest",
    "RemoveTagsRequest",
    "ReplaceTextRequest",
    "ReadCaptionRequest",
    "WriteCaptionRequest",
    "CaptionOperationResponse",
    "CaptionReadResponse",
    # LoRA
    "LoRAResizeRequest",
    "LoRAResizeResponse",
    "HuggingFaceUploadRequest",
    "HuggingFaceUploadResponse",
]
