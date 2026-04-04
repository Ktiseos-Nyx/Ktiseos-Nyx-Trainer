"""
FastAPI Main Application
Provides REST API and WebSocket endpoints for the Next.js frontend.
"""
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import civitai, config, dataset, files, models, settings, training, utilities
from services import websocket

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

_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

# Only construct and add the FileHandler if not already present (guards against
# double-attachment on reload and avoids leaking file descriptors on each reload).
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file)
           for h in _root_logger.handlers):
    _file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
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
    redoc_url="/redoc"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://127.0.0.1:3000",
        "*"  # Allow all origins (VastAI port forwarding)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Needed for canvas image manipulation
)

# Request logging middleware — logs every request to the file handler
# so local dev (Windows) gets visibility into what's hitting FastAPI.
@app.middleware("http")
async def log_requests(request: Request, call_next):
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
app.include_router(civitai.router, prefix="/api/civitai", tags=["Civitai"])

# Include WebSocket routes (no prefix - they define their own paths)
app.include_router(websocket.router, tags=["WebSocket"])


@app.get("/")
async def root():
    """Root endpoint - API status check"""
    return {
        "name": "Ktiseos-Nyx-Trainer API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


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
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn  # pyright: ignore[reportMissingImports]
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
