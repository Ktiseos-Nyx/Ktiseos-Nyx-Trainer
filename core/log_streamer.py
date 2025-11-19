# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx

"""
Log Streaming System for Real-time Training/Tagging Output
Provides thread-safe log broadcasting to WebSocket clients
"""

import logging
import queue
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LogStreamer:
    """
    Thread-safe log streamer that bridges synchronous subprocess output
    to asynchronous WebSocket broadcasting.

    Usage:
        streamer = LogStreamer()
        streamer.add_line("Training started...")

        # In async context:
        await streamer.broadcast_pending()
    """

    def __init__(self):
        self.log_queue = queue.Queue()
        self.websocket_connections: List[Any] = []
        self._lock = threading.Lock()

    def add_line(self, line: str, log_type: str = "log"):
        """
        Add a log line to the queue (thread-safe).
        Called from synchronous training subprocess.

        Args:
            line: Log line content
            log_type: Type of log (log, progress, error, warning)
        """
        self.log_queue.put({"type": log_type, "message": line.rstrip()})

    def add_progress(self, data: Dict[str, Any]):
        """
        Add progress update to the queue (thread-safe).

        Args:
            data: Progress data (current_step, total_steps, loss, lr, etc.)
        """
        self.log_queue.put({"type": "progress", "data": data})

    async def broadcast_pending(self):
        """
        Broadcast all pending logs to connected WebSocket clients (async).
        Should be called periodically from async event loop.
        """
        messages_to_send = []

        # Drain queue
        while not self.log_queue.empty():
            try:
                messages_to_send.append(self.log_queue.get_nowait())
            except queue.Empty:
                break

        if not messages_to_send:
            return

        # Broadcast to all connections
        with self._lock:
            disconnected = []
            for ws in self.websocket_connections:
                for message in messages_to_send:
                    try:
                        await ws.send_json(message)
                    except Exception as e:
                        logger.warning(f"Failed to send to WebSocket: {e}")
                        disconnected.append(ws)
                        break

            # Clean up disconnected clients
            for ws in disconnected:
                try:
                    self.websocket_connections.remove(ws)
                except ValueError:
                    pass

    def add_connection(self, websocket):
        """Add a WebSocket connection to broadcast list (thread-safe)"""
        with self._lock:
            if websocket not in self.websocket_connections:
                self.websocket_connections.append(websocket)
                logger.info(
                    f"WebSocket connected. Total connections: {len(self.websocket_connections)}"
                )

    def remove_connection(self, websocket):
        """Remove a WebSocket connection from broadcast list (thread-safe)"""
        with self._lock:
            try:
                self.websocket_connections.remove(websocket)
                logger.info(
                    f"WebSocket disconnected. Remaining connections: {len(self.websocket_connections)}"
                )
            except ValueError:
                pass

    def clear(self):
        """Clear all pending logs"""
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break


# Global singleton instance
_global_training_streamer: Optional[LogStreamer] = None
_global_tagging_streamer: Optional[LogStreamer] = None


def get_training_log_streamer() -> LogStreamer:
    """Get or create the global training log streamer"""
    global _global_training_streamer
    if _global_training_streamer is None:
        _global_training_streamer = LogStreamer()
    return _global_training_streamer


def get_tagging_log_streamer() -> LogStreamer:
    """Get or create the global tagging log streamer"""
    global _global_tagging_streamer
    if _global_tagging_streamer is None:
        _global_tagging_streamer = LogStreamer()
    return _global_tagging_streamer
