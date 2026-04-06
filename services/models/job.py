"""
Job-related Pydantic models.

Used for job status tracking and WebSocket responses.
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class JobType(str, Enum):
    """Type of background job"""
    TRAINING = "training"
    TAGGING = "tagging"
    DOWNLOAD = "download"


class JobStatusEnum(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(BaseModel):
    """
    Job status response.

    Returned by status endpoints and WebSocket for progress tracking.
    """
    job_id: str
    job_type: JobType
    status: JobStatusEnum
    progress: int = Field(ge=0, le=100, description="Progress percentage (0-100)")

    # Optional progress details
    current_step: Optional[str] = Field(None, description="Current operation (e.g., 'Epoch 3/10')")
    current_epoch: Optional[int] = Field(None, description="Current epoch number")
    total_epochs: Optional[int] = None
    current_image: Optional[str] = Field(None, description="Currently processing image")
    total_images: Optional[int] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error info
    error: Optional[str] = Field(None, description="Error message if failed")
    error_traceback: Optional[str] = Field(None, description="Full traceback for debugging")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-abc123",
                "job_type": "training",
                "status": "running",
                "progress": 45,
                "current_step": "Epoch 3/10",
                "current_epoch": 3,
                "total_epochs": 10,
                "created_at": "2025-11-29T10:00:00Z",
                "started_at": "2025-11-29T10:00:05Z"
            }
        }


class JobCreateResponse(BaseModel):
    """Response when creating a new job"""
    job_id: str
    status: str = "started"
    message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-abc123",
                "status": "started",
                "message": "Training job started successfully"
            }
        }
