"""
Parse training and tagging logs for progress extraction.

Kohya scripts output logs with epoch/step information.
This module extracts structured progress data from log strings.
"""

import re
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingProgress:
    """
    Parsed training progress information.
    """
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    lr: Optional[float] = None  # Learning rate
    progress_percent: int = 0  # 0-100


@dataclass
class TaggingProgress:
    """
    Parsed tagging progress information.
    """
    current_image: Optional[int] = None
    total_images: Optional[int] = None
    current_file: Optional[str] = None
    progress_percent: int = 0  # 0-100


class LogParser:
    """
    Parse Kohya training and WD14 tagging logs for progress.

    Examples of Kohya log formats:
        "epoch 3/10, step 150/500"
        "steps: 150, loss: 0.0234, lr: 0.0001"
        "epoch 5, loss=0.0145"
    """

    # Regex patterns for Kohya training logs
    EPOCH_PATTERN = re.compile(r'epoch[:\s]+(\d+)(?:/(\d+))?', re.I)
    STEP_PATTERN = re.compile(r'step[:\s]+(\d+)(?:/(\d+))?', re.I)
    LOSS_PATTERN = re.compile(r'loss[:\s=]+([0-9.]+)', re.I)
    LR_PATTERN = re.compile(r'lr[:\s=]+([0-9.e-]+)', re.I)

    # WD14 tagger patterns
    TAGGING_IMAGE_PATTERN = re.compile(r'(\d+)/(\d+)', re.I)
    TAGGING_FILE_PATTERN = re.compile(r'tagging:\s+(.+?)(?:\s|$)', re.I)

    @classmethod
    def parse_training_log(cls, log_line: str) -> Optional[TrainingProgress]:
        """
        Parse a training log line for progress information.

        Args:
            log_line: Single line from training logs

        Returns:
            TrainingProgress if progress info found, None otherwise

        Example:
            >>> parser = LogParser()
            >>> progress = parser.parse_training_log("epoch 3/10, step 150/500, loss: 0.0234")
            >>> progress.epoch
            3
            >>> progress.total_epochs
            10
            >>> progress.progress_percent
            30
        """
        if not log_line:
            return None

        progress = TrainingProgress()
        found_anything = False

        # Extract epoch
        epoch_match = cls.EPOCH_PATTERN.search(log_line)
        if epoch_match:
            progress.epoch = int(epoch_match.group(1))
            if epoch_match.group(2):
                progress.total_epochs = int(epoch_match.group(2))
            found_anything = True

        # Extract step
        step_match = cls.STEP_PATTERN.search(log_line)
        if step_match:
            progress.step = int(step_match.group(1))
            if step_match.group(2):
                progress.total_steps = int(step_match.group(2))
            found_anything = True

        # Extract loss
        loss_match = cls.LOSS_PATTERN.search(log_line)
        if loss_match:
            try:
                progress.loss = float(loss_match.group(1))
                found_anything = True
            except ValueError:
                pass

        # Extract learning rate
        lr_match = cls.LR_PATTERN.search(log_line)
        if lr_match:
            try:
                progress.lr = float(lr_match.group(1))
                found_anything = True
            except ValueError:
                pass

        # Calculate progress percentage
        if progress.epoch and progress.total_epochs:
            progress.progress_percent = int((progress.epoch / progress.total_epochs) * 100)
        elif progress.step and progress.total_steps:
            progress.progress_percent = int((progress.step / progress.total_steps) * 100)

        return progress if found_anything else None

    @classmethod
    def parse_tagging_log(cls, log_line: str) -> Optional[TaggingProgress]:
        """
        Parse a WD14 tagging log line for progress.

        Args:
            log_line: Single line from tagging logs

        Returns:
            TaggingProgress if progress info found, None otherwise

        Example:
            >>> progress = LogParser.parse_tagging_log("Processing: 45/100")
            >>> progress.current_image
            45
            >>> progress.progress_percent
            45
        """
        if not log_line:
            return None

        progress = TaggingProgress()
        found_anything = False

        # Extract image count (e.g., "45/100")
        image_match = cls.TAGGING_IMAGE_PATTERN.search(log_line)
        if image_match:
            progress.current_image = int(image_match.group(1))
            progress.total_images = int(image_match.group(2))

            # Calculate percentage
            if progress.total_images > 0:
                progress.progress_percent = int((progress.current_image / progress.total_images) * 100)

            found_anything = True

        # Extract current filename
        file_match = cls.TAGGING_FILE_PATTERN.search(log_line)
        if file_match:
            progress.current_file = file_match.group(1).strip()
            found_anything = True

        return progress if found_anything else None

    @classmethod
    def extract_error(cls, log_line: str) -> Optional[str]:
        """
        Check if log line contains an error.

        Args:
            log_line: Single line from logs

        Returns:
            Error message if found, None otherwise
        """
        error_indicators = [
            'error',
            'exception',
            'failed',
            'traceback',
            'fatal',
        ]

        lower_line = log_line.lower()
        for indicator in error_indicators:
            if indicator in lower_line:
                return log_line.strip()

        return None
