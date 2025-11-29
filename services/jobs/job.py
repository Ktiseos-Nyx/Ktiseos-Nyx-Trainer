"""
Job dataclass for tracking background operations.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import deque

from services.models.job import JobType, JobStatusEnum


@dataclass
class Job:
    """
    Represents a background job (training, tagging, etc.)

    Tracks the job's lifecycle, progress, and logs.
    """
    job_id: str
    job_type: JobType
    status: JobStatusEnum = JobStatusEnum.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    # Process management
    process: Optional[asyncio.subprocess.Process] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Progress tracking
    progress: int = 0  # 0-100
    current_step: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_image: Optional[str] = None
    total_images: Optional[int] = None

    # Log buffer (keep last max_logs lines)
    logs: deque = field(default_factory=lambda: deque(maxlen=1000))
    max_logs: int = 1000

    # Error tracking
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    def add_log(self, log_line: str):
        """Add log line to buffer"""
        self.logs.append(log_line)

    def get_logs(self, start: int = 0) -> list[str]:
        """
        Get logs starting from line number.

        Args:
            start: Line number to start from (0-indexed)

        Returns:
            List of log lines from start onwards
        """
        if start >= len(self.logs):
            return []
        return list(self.logs)[start:]

    @property
    def is_running(self) -> bool:
        """Check if job is currently running"""
        return self.status == JobStatusEnum.RUNNING

    @property
    def is_complete(self) -> bool:
        """Check if job is in a terminal state"""
        return self.status in {
            JobStatusEnum.COMPLETED,
            JobStatusEnum.FAILED,
            JobStatusEnum.CANCELLED
        }

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds (if completed)"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None
