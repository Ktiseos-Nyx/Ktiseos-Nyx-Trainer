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
    current_step: Optional[str] = None  # human-readable label, e.g. "Epoch 1/10"
    step_num: Optional[int] = None      # numeric step index from tqdm
    total_steps: Optional[int] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    lr: Optional[float] = None
    eta_seconds: Optional[int] = None
    current_image: Optional[str] = None
    total_images: Optional[int] = None

    # Log buffer (keep last max_logs lines)
    max_logs: int = 2000
    logs: deque = field(init=False)
    # Absolute count of lines ever written — survives deque eviction.
    # 'since' cursors from the frontend are absolute offsets into this counter,
    # not relative indices into the deque. This prevents the deque wrap-around
    # bug where get_logs(N) returns [] forever once N == maxlen.
    total_lines_written: int = 0

    # Error tracking
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    def __post_init__(self):
        self.logs = deque(maxlen=self.max_logs)

    def add_log(self, log_line: str):
        """Add log line to buffer"""
        self.logs.append(log_line)
        self.total_lines_written += 1

    def get_logs(self, since: int = 0) -> list[str]:
        """
        Get logs since an absolute line number.

        Args:
            since: Absolute line count offset (matches total_lines_written
                   at the time the caller last received logs). NOT a deque index.

        Returns:
            List of log lines from since onwards (capped to buffered window).
        """
        # Oldest absolute line number still in the deque
        oldest_absolute = self.total_lines_written - len(self.logs)
        if since <= oldest_absolute:
            # Caller is behind the buffer window — return everything buffered
            return list(self.logs)
        relative_start = since - oldest_absolute
        if relative_start >= len(self.logs):
            return []
        return list(self.logs)[relative_start:]

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
