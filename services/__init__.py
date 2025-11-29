"""
Services layer for Ktiseos-Nyx-Trainer backend.

Clean, modular service layer replacing old Jupyter managers.

Quick Start:
    from services import training_service, tagging_service

    # Start training
    response = await training_service.start_training(config)
    job_id = response.job_id

    # Monitor via WebSocket
    # ws://localhost:8000/ws/jobs/{job_id}/logs

Services:
    - training_service: LoRA training orchestration
    - tagging_service: WD14 auto-tagging
    - dataset_service: Dataset management
    - caption_service: Caption editing
    - lora_service: LoRA utilities (resize, HF upload)
    - job_manager: Job tracking (used internally by services)
"""

# Service instances (singletons)
from .training_service import training_service
from .tagging_service import tagging_service
from .dataset_service import dataset_service
from .caption_service import caption_service
from .lora_service import lora_service
from .jobs import job_manager

# Service classes (if you need to instantiate manually)
from .training_service import TrainingService
from .tagging_service import TaggingService
from .dataset_service import DatasetService
from .caption_service import CaptionService
from .lora_service import LoRAService
from .jobs import JobManager

__all__ = [
    # Singleton instances (use these in most cases)
    "training_service",
    "tagging_service",
    "dataset_service",
    "caption_service",
    "lora_service",
    "job_manager",
    # Classes (if you need custom instances)
    "TrainingService",
    "TaggingService",
    "DatasetService",
    "CaptionService",
    "LoRAService",
    "JobManager",
]
