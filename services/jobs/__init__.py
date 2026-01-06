"""
Job tracking system for long-running operations.

Tracks background jobs (training, tagging, downloads) and provides:
- Job status tracking
- Progress monitoring
- Log streaming via WebSocket
- Process management

Usage:
    from services.jobs import job_manager

    # Create job
    job_id = job_manager.create_job("training", process)

    # Get status
    status = await job_manager.get_job_status(job_id)

    # Stream logs
    async for log_line in job_manager.stream_logs(job_id):
        print(log_line)
"""

from .job_manager import job_manager, JobManager
from .job import Job

__all__ = [
    "job_manager",  # Global singleton
    "JobManager",
    "Job",
]
