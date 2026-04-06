"""
In-memory job storage.

Simple dict-based storage for v1. Can be replaced with SQLite later if needed.
"""

from typing import Optional, Dict
from .job import Job


class JobStore:
    """
    In-memory storage for jobs.

    Thread-safe for async operations.
    Jobs are lost on server restart (fine for v1).
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def add(self, job: Job) -> None:
        """Add job to store"""
        self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self._jobs.get(job_id)

    def exists(self, job_id: str) -> bool:
        """Check if job exists"""
        return job_id in self._jobs

    def remove(self, job_id: str) -> bool:
        """
        Remove job from store.

        Returns:
            True if job was removed, False if not found
        """
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    def get_all(self) -> list[Job]:
        """Get all jobs"""
        return list(self._jobs.values())

    def get_by_type(self, job_type: str) -> list[Job]:
        """Get all jobs of a specific type"""
        return [job for job in self._jobs.values() if job.job_type == job_type]

    def get_running(self) -> list[Job]:
        """Get all currently running jobs"""
        return [job for job in self._jobs.values() if job.is_running]

    def count(self) -> int:
        """Get total number of jobs"""
        return len(self._jobs)

    def clear(self) -> None:
        """Clear all jobs (use with caution!)"""
        self._jobs.clear()
