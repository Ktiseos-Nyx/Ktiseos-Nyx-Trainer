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
    - model_service: Model/VAE downloads with aria2c support
    - job_manager: Job tracking (used internally by services)
"""

# Service instances (singletons)
from .training_service import training_service
from .tagging_service import tagging_service
from .dataset_service import dataset_service
from .caption_service import caption_service
from .captioning_service import captioning_service
from .lora_service import lora_service
from .model_service import model_service
from .jobs import job_manager

# Service classes (if you need to instantiate manually)
from .training_service import TrainingService
from .tagging_service import TaggingService
from .dataset_service import DatasetService
from .caption_service import CaptionService
from .captioning_service import CaptioningService
from .lora_service import LoRAService
from .model_service import ModelService
from .jobs import JobManager

__all__ = [
    # Singleton instances (use these in most cases)
    "training_service",
    "tagging_service",
    "dataset_service",
    "caption_service",
    "captioning_service",
    "lora_service",
    "model_service",
    "job_manager",
    # Classes (if you need custom instances)
    "TrainingService",
    "TaggingService",
    "DatasetService",
    "CaptionService",
    "CaptioningService",
    "LoRAService",
    "ModelService",
    "JobManager",
]
