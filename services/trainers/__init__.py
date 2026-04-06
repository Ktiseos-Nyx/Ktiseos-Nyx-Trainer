"""
Training backend abstraction layer.

Provides a flexible BaseTrainer pattern for different training backends:
- KohyaTrainer: Current Kohya-ss implementation (v1)
- Future: Multi-GPU trainers, ROCm support, other frameworks
"""

from .base import BaseTrainer
from .kohya import KohyaTrainer

__all__ = [
    "BaseTrainer",
    "KohyaTrainer",
]
