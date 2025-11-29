"""
WebSocket endpoint for real-time log streaming.

Provides live updates for training and tagging jobs.
"""

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter
import logging

from services.jobs import job_manager
from services.core.exceptions import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/jobs/{job_id}/logs")
async def stream_job_logs(websocket: WebSocket, job_id: str):
    """
    Stream job logs in real-time via WebSocket.

    Args:
        job_id: Job identifier
        websocket: WebSocket connection

    WebSocket Protocol:
        - Client connects to /ws/jobs/{job_id}/logs
        - Server streams log lines as text messages
        - Connection closes when job completes
        - Client can disconnect anytime

    Example (JavaScript):
        const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${jobId}/logs`);
        ws.onmessage = (event) => {
            console.log('Log:', event.data);
        };
    """
    await websocket.accept()

    try:
        # Check if job exists
        status = await job_manager.get_job_status(job_id)
        if not status:
            await websocket.send_json({
                "error": "Job not found",
                "job_id": job_id
            })
            await websocket.close()
            return

        logger.info(f"WebSocket connected for job {job_id}")

        # Stream logs to client
        async for log_line in job_manager.stream_logs(job_id):
            await websocket.send_text(log_line)

        # Send completion signal
        await websocket.send_json({
            "status": "complete",
            "job_id": job_id
        })

        logger.info(f"Job {job_id} completed, closing WebSocket")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")

    except Exception as e:
        logger.exception(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.send_json({
                "error": str(e),
                "job_id": job_id
            })
        except:
            pass  # Connection already closed

    finally:
        try:
            await websocket.close()
        except:
            pass  # Already closed


@router.websocket("/ws/jobs/{job_id}/status")
async def stream_job_status(websocket: WebSocket, job_id: str):
    """
    Stream job status updates via WebSocket.

    Sends periodic status updates including progress percentage,
    current step, epoch info, etc.

    Args:
        job_id: Job identifier
        websocket: WebSocket connection

    Example status update:
        {
            "job_id": "job-abc123",
            "status": "running",
            "progress": 45,
            "current_step": "Epoch 3/10",
            "current_epoch": 3,
            "total_epochs": 10
        }
    """
    await websocket.accept()

    try:
        import asyncio

        logger.info(f"Status WebSocket connected for job {job_id}")

        while True:
            status = await job_manager.get_job_status(job_id)

            if not status:
                await websocket.send_json({
                    "error": "Job not found",
                    "job_id": job_id
                })
                break

            # Send current status
            await websocket.send_json(status.dict())

            # Stop streaming if job is complete
            if status.status in {"completed", "failed", "cancelled"}:
                logger.info(f"Job {job_id} finished, closing status WebSocket")
                break

            # Update every 500ms
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"Status WebSocket disconnected for job {job_id}")

    except Exception as e:
        logger.exception(f"Status WebSocket error for job {job_id}: {e}")

    finally:
        try:
            await websocket.close()
        except:
            pass
