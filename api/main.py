"""
FastAPI Main Application
Provides REST API and WebSocket endpoints for the Next.js frontend.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import civitai, config, dataset, debug, files, models, settings, sources, training, utilities
from services import websocket

# PyTorch CUDA allocator config — must be set BEFORE any torch import.
# expandable_segments:True lets the allocator return freed memory to the OS
# instead of caching it indefinitely, which caused "phantom RAM" growth in
# VastAI's portal AIO after HuggingFace downloads and Chattiori merges/bakes.
# Safe on all platforms — ignored if CUDA isn't available.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# torchao ships compiled C extensions (_C*.so / _C*.pyd) that are built
# against a specific CPython ABI (often only 3.10). On 3.11+ those .so files
# fail to load with opaque errors. We only use torchao's pure-Python paths
# (TorchAOBaseTensor / get_available_devices from torchao.utils), never its
# quantized kernels, so skipping the compiled extension is safe.
os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")

# Windows: ensure ProactorEventLoop so asyncio.create_subprocess_exec works
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Background periodic GC to combat memory fragmentation from HuggingFace downloads
# and Chattiori merge/bake operations. These libraries allocate large tensors and
# sometimes hold cached allocator memory that never gets returned to the OS.
_AUTO_GC_INTERVAL_SECONDS = 300  # run every 5 minutes
_auto_gc_task: "asyncio.Task[None] | None" = None


async def _auto_gc_loop():
    """Periodically run gc.collect() + torch.cuda.empty_cache() in the background.

    Skips the cycle when any job is running (training, tagging, merge, bake,
    download, etc.) to avoid pausing or fragmenting the CUDA allocator mid-work.
    """
    import gc

    while True:
        try:
            await asyncio.sleep(_AUTO_GC_INTERVAL_SECONDS)

            from services.jobs import job_manager
            if job_manager.store.get_running():
                continue

            collected = gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if collected > 0:
                logger.debug("Auto-GC: collected %d objects", collected)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.debug("Auto-GC: error in background loop", exc_info=True)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging with file output (cross-platform: Windows, Linux, macOS)
# NOTE: logging.basicConfig() is a no-op when uvicorn has already set up root logger
# handlers. We attach directly to the root logger instead so the file handler
# always gets added regardless of what uvicorn configured first.
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

# Create log filename with date (one file per day)
log_file = logs_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

_log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

# Only construct and add the FileHandler if not already present (guards against
# double-attachment on reload and avoids leaking file descriptors on each reload).
if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_file)
    for h in _root_logger.handlers
):
    _file_handler = logging.FileHandler(log_file, encoding="utf-8", errors="replace")
    _file_handler.setFormatter(_log_formatter)
    _root_logger.addHandler(_file_handler)

logger = logging.getLogger(__name__)
logger.info("📝 Logging to: %s", log_file)
logger.info("📂 Project root: %s", project_root)
logger.info("🖥️  Platform: %s", sys.platform)

# Create FastAPI app
app = FastAPI(
    title="Ktiseos-Nyx-Trainer API",
    description="LoRA training API with real-time progress tracking",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Next.js frontend.
# allow_credentials=True + allow_origins=["*"] is a CORS spec violation and
# newer Starlette versions raise ValueError at startup for this combination.
# This app passes tokens in request bodies (not cookies), so credentials
# mode is not required. allow_origins=["*"] covers local dev AND VastAI/RunPod
# dynamic proxy URLs without needing to enumerate every possible hostname.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Needed for canvas image manipulation
)


# Request logging middleware — logs every request to the file handler
# so local dev (Windows) gets visibility into what's hitting FastAPI.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip health checks to reduce log noise
    if request.url.path in ("/health", "/api/health"):
        return await call_next(request)
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s → %d (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# Include routers
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])
app.include_router(config.router, prefix="/api/config", tags=["Config"])
app.include_router(utilities.router, prefix="/api/utilities", tags=["Utilities"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(sources.router, prefix="/api/sources", tags=["Sources"])
app.include_router(civitai.router, prefix="/api/civitai", tags=["Civitai"])
app.include_router(debug.router, prefix="/api/debug", tags=["Debug"])

# Include WebSocket routes (no prefix - they define their own paths)
app.include_router(websocket.router, tags=["WebSocket"])


@app.get("/")
async def root():
    """Root endpoint - API status check"""
    return {"name": "Ktiseos-Nyx-Trainer API", "version": "0.1.0", "status": "running", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/health")
async def root_health_check():
    """Root health check endpoint for load balancers"""
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.on_event("startup")
async def start_auto_gc():
    global _auto_gc_task
    if _auto_gc_task is None:
        _auto_gc_task = asyncio.create_task(_auto_gc_loop())
        logger.info("Auto-GC background task started (interval: %ds)", _AUTO_GC_INTERVAL_SECONDS)


@app.on_event("shutdown")
async def stop_auto_gc():
    global _auto_gc_task
    if _auto_gc_task is not None:
        _auto_gc_task.cancel()
        try:
            await _auto_gc_task
        except asyncio.CancelledError:
            pass
        _auto_gc_task = None
        logger.info("Auto-GC background task stopped")


if __name__ == "__main__":
    import uvicorn  # pyright: ignore[reportMissingImports]

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")