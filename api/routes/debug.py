"""
API routes for client-side error reporting.
Receives browser error reports from Next.js error boundaries and logs them
to app.log so non-technical users don't need to open DevTools.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ClientError(BaseModel):
    """Browser-side error payload sent by Next.js error boundaries."""

    message: str
    digest: Optional[str] = None
    stack: Optional[str] = None
    url: Optional[str] = None
    boundary: Optional[str] = None  # "route" or "global"


@router.post("/client-error")
async def log_client_error(error: ClientError):
    """Receive and log browser-side errors from Next.js error boundaries."""
    boundary_label = error.boundary or "unknown"
    logger.error(
        "🌐 Browser error [%s boundary] at %s: %s%s",
        boundary_label,
        error.url or "unknown URL",
        error.message,
        f" (digest: {error.digest})" if error.digest else "",
    )
    if error.stack:
        logger.error("Stack trace:\n%s", error.stack)
    return JSONResponse({"logged": True})
