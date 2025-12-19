"""
FastAPI Main Application
Provides REST API and WebSocket endpoints for the Next.js frontend.
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from api.routes import training, dataset, files, config, utilities, models, settings, civitai
from services import websocket

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
