"""
Windows-safe FastAPI backend launcher.

On Windows, asyncio.create_subprocess_exec requires ProactorEventLoop.
Python 3.8+ defaults to ProactorEventLoop, but uvicorn creates its event
loop before importing the app module — so setting the policy inside
api/main.py runs too late when uvicorn is invoked with -m uvicorn.

This wrapper sets WindowsProactorEventLoopPolicy BEFORE uvicorn touches
the event loop, guaranteeing subprocesses (i.e. training) work correctly.

Usage (automatically called by start_services_local.bat):
    python run_backend.py [--host 127.0.0.1] [--port 8000]
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ktiseos-Nyx-Trainer Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    uvicorn.run("api.main:app", host=args.host, port=args.port, log_level="info")
