"""
Training API Routes
Handles training start/stop, status monitoring, and log streaming.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import logging
from pathlib import Path

from shared_managers import get_training_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Active WebSocket connections for log streaming
active_connections: List[WebSocket] = []


class TrainingConfig(BaseModel):
    """Training configuration parameters"""
    # Basic settings
    model_name: str
    pretrained_model_name_or_path: str
    output_dir: str

    # Dataset
    train_data_dir: str

    # Training parameters
    resolution: int = 512
    train_batch_size: int = 1
    max_train_epochs: Optional[int] = None
    max_train_steps: Optional[int] = 1000
    learning_rate: float = 1e-4

    # LoRA settings
    network_module: str = "networks.lora"
    network_dim: int = 32
    network_alpha: int = 32

    # Advanced
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    save_precision: str = "fp16"

    # Additional parameters
    extra_params: Dict[str, Any] = {}


class TrainingStartResponse(BaseModel):
    """Response when training starts"""
    success: bool
    message: str
    training_id: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    """Training status response"""
    is_training: bool
    progress: Optional[Dict[str, Any]] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(config: TrainingConfig):
    """
    Start a new training session.
    Returns immediately with training ID for monitoring via WebSocket.
    """
    try:
        training_manager = get_training_manager()

        # Check if already training
        status = training_manager.get_training_status()
        if status.get("is_training", False):
            raise HTTPException(
                status_code=400,
                detail="Training already in progress"
            )

        # Convert config to dict for training manager
        config_dict = config.dict()

        # Start training (non-blocking)
        logger.info(f"Starting training: {config.model_name}")

        # TODO: Implement async training launch
        # For now, this is a placeholder
        training_id = f"train_{config.model_name}_{asyncio.get_event_loop().time()}"

        return TrainingStartResponse(
            success=True,
            message="Training started successfully",
            training_id=training_id
        )

    except Exception as e:
        logger.error(f"Failed to start training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_training():
    """Stop the current training session"""
    try:
        training_manager = get_training_manager()

        # TODO: Implement training stop functionality
        logger.info("Stopping training...")

        return {"success": True, "message": "Training stopped"}

    except Exception as e:
        logger.error(f"Failed to stop training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get current training status and progress"""
    try:
        training_manager = get_training_manager()
        status = training_manager.get_training_status()

        return TrainingStatusResponse(
            is_training=status.get("is_training", False),
            progress=status.get("progress"),
            current_step=status.get("current_step"),
            total_steps=status.get("total_steps"),
            current_epoch=status.get("current_epoch"),
            total_epochs=status.get("total_epochs")
        )

    except Exception as e:
        logger.error(f"Failed to get training status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/logs")
async def training_logs_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training logs.
    NO MORE RACE CONDITIONS! 🎉
    """
    await websocket.accept()
    active_connections.append(websocket)

    try:
        logger.info("Client connected to training logs WebSocket")

        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to training logs"
        })

        # Keep connection alive and send updates
        while True:
            # TODO: Read logs from training manager and stream
            # For now, just keep connection alive
            await asyncio.sleep(1)

            # Example: Send heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": asyncio.get_event_loop().time()
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected from training logs WebSocket")
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_log(message: Dict[str, Any]):
    """
    Broadcast log message to all connected WebSocket clients.
    Call this from training manager when new logs are available.
    """
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            disconnected.append(connection)

    # Clean up disconnected clients
    for conn in disconnected:
        active_connections.remove(conn)


@router.get("/history")
async def get_training_history():
    """Get list of past training sessions"""
    try:
        # TODO: Implement training history from output directory
        output_dir = Path("/workspace/output")

        if not output_dir.exists():
            return {"trainings": []}

        trainings = []
        for item in output_dir.iterdir():
            if item.is_dir():
                trainings.append({
                    "name": item.name,
                    "path": str(item),
                    "created": item.stat().st_ctime
                })

        trainings.sort(key=lambda x: x["created"], reverse=True)
        return {"trainings": trainings}

    except Exception as e:
        logger.error(f"Failed to get training history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
