"""
Job Manager - Central job tracking and monitoring.

Manages lifecycle of background jobs (training, tagging, downloads).
"""

import asyncio
import logging
from uuid import uuid4
from datetime import datetime
from typing import Optional, AsyncIterator

from services.models.job import JobType, JobStatus, JobStatusEnum
from services.core.log_parser import LogParser
from .job import Job
from .job_store import JobStore

logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages background jobs and log streaming.

    Responsibilities:
    - Create and track jobs
    - Monitor subprocess execution
    - Parse logs for progress
    - Stream logs via WebSocket
    """

    def __init__(self):
        self.store = JobStore()
        self.log_parser = LogParser()

    def create_job(
        self,
        job_type: JobType,
        process: asyncio.subprocess.Process
    ) -> str:
        """
        Create new job and start monitoring.

        Args:
            job_type: Type of job (training, tagging, etc.)
            process: Async subprocess

        Returns:
            job_id: Unique identifier for this job

        Example:
            proc = await asyncio.create_subprocess_exec(...)
            job_id = job_manager.create_job(JobType.TRAINING, proc)
        """
        job_id = f"job-{uuid4().hex[:8]}"

        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatusEnum.RUNNING,
            process=process,
            started_at=datetime.now()
        )

        self.store.add(job)

        # Start monitoring in background
        asyncio.create_task(self._monitor_job(job_id))

        logger.info(f"Created {job_type} job: {job_id}")
        return job_id

    async def _monitor_job(self, job_id: str):
        """
        Monitor job subprocess and collect logs.

        This runs in the background and:
        - Reads stdout/stderr
        - Parses logs for progress
        - Updates job status
        - Detects completion/errors
        """
        job = self.store.get(job_id)
        if not job or not job.process:
            return

        try:
            # Read stdout line by line
            async for line in job.process.stdout:
                log_line = line.decode('utf-8', errors='replace').strip()

                # Add to log buffer
                job.add_log(log_line)

                # Parse for progress (training-specific)
                if job.job_type == JobType.TRAINING:
                    progress = self.log_parser.parse_training_log(log_line)
                    if progress:
                        job.progress = progress.progress_percent
                        if progress.epoch:
                            job.current_epoch = progress.epoch
                        if progress.total_epochs:
                            job.total_epochs = progress.total_epochs
                        if progress.epoch and progress.total_epochs:
                            job.current_step = f"Epoch {progress.epoch}/{progress.total_epochs}"

                # Parse for progress (tagging-specific)
                elif job.job_type == JobType.TAGGING:
                    progress = self.log_parser.parse_tagging_log(log_line)
                    if progress:
                        job.progress = progress.progress_percent
                        if progress.current_image:
                            job.current_image = progress.current_file or f"Image {progress.current_image}"
                        if progress.total_images:
                            job.total_images = progress.total_images

                # Check for errors
                error = self.log_parser.extract_error(log_line)
                if error and not job.error:
                    job.error = error

            # Wait for process to complete
            returncode = await job.process.wait()

            # Update job status based on exit code
            job.completed_at = datetime.now()

            if returncode == 0:
                job.status = JobStatusEnum.COMPLETED
                job.progress = 100
                logger.info(f"Job {job_id} completed successfully")
            else:
                job.status = JobStatusEnum.FAILED
                if not job.error:
                    job.error = f"Process exited with code {returncode}"
                logger.error(f"Job {job_id} failed: {job.error}")

        except asyncio.CancelledError:
            job.status = JobStatusEnum.CANCELLED
            job.completed_at = datetime.now()
            logger.info(f"Job {job_id} was cancelled")

        except Exception as e:
            job.status = JobStatusEnum.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            logger.exception(f"Job {job_id} monitoring failed: {e}")

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get current job status.

        Args:
            job_id: Job identifier

        Returns:
            JobStatus model or None if not found
        """
        job = self.store.get(job_id)
        if not job:
            return None

        return JobStatus(
            job_id=job.job_id,
            job_type=job.job_type,
            status=job.status,
            progress=job.progress,
            current_step=job.current_step,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            current_image=job.current_image,
            total_images=job.total_images,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
            error_traceback=job.error_traceback
        )

    async def stop_job(self, job_id: str) -> bool:
        """
        Stop a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if stopped, False if not found or already stopped
        """
        job = self.store.get(job_id)
        if not job or not job.is_running:
            return False

        if job.process:
            try:
                job.process.terminate()
                await asyncio.sleep(1)

                # Force kill if still running
                if job.process.returncode is None:
                    job.process.kill()

                job.status = JobStatusEnum.CANCELLED
                job.completed_at = datetime.now()
                logger.info(f"Job {job_id} stopped")
                return True

            except Exception as e:
                logger.error(f"Failed to stop job {job_id}: {e}")
                return False

        return False

    async def stream_logs(self, job_id: str, start_line: int = 0) -> AsyncIterator[str]:
        """
        Stream logs for a job (for WebSocket).

        Args:
            job_id: Job identifier
            start_line: Line number to start from

        Yields:
            Log lines as they become available

        Example:
            async for log_line in job_manager.stream_logs(job_id):
                await websocket.send_json({"type": "log", "message": log_line})
        """
        job = self.store.get(job_id)
        if not job:
            return

        # Send existing logs first
        existing_logs = job.get_logs(start_line)
        for log_line in existing_logs:
            yield log_line

        # Then stream new logs as they arrive
        last_line = len(job.logs)
        heartbeat_counter = 0

        while not job.is_complete:
            await asyncio.sleep(0.1)  # Check every 100ms
            heartbeat_counter += 1

            # Send heartbeat every 50 iterations (5 seconds)
            if heartbeat_counter >= 50:
                yield "__HEARTBEAT__"
                heartbeat_counter = 0

            # Yield new lines
            new_logs = job.get_logs(last_line)
            for log_line in new_logs:
                yield log_line

            last_line = len(job.logs)

        # Send final logs after completion
        final_logs = job.get_logs(last_line)
        for log_line in final_logs:
            yield log_line


# Global singleton instance
job_manager = JobManager()
