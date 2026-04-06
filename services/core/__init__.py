"""
Core utilities for the service layer.

This package contains shared utilities used across all services:
- Exceptions: Custom exception hierarchy
- Validation: Path and input validation
- Log parsing: Extract progress from training logs
"""

from .exceptions import (
    ServiceError,
    ValidationError,
    NotFoundError,
    ProcessError,
    ConfigError,
)
from .validation import (
    validate_dataset_path,
    validate_model_path,
    validate_image_filename,
    DATASETS_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
)

__all__ = [
    # Exceptions
    "ServiceError",
    "ValidationError",
    "NotFoundError",
    "ProcessError",
    "ConfigError",
    # Validation
    "validate_dataset_path",
    "validate_model_path",
    "validate_image_filename",
    "DATASETS_DIR",
    "MODELS_DIR",
    "OUTPUT_DIR",
]
