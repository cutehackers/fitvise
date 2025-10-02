"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1.fitvise import chat
from app.api.v1.rag import data_sources, ingestion, processing

# Create main API router
router = APIRouter()

# Include endpoint routers
router.include_router(chat.router, prefix="/fitvise")
router.include_router(data_sources.router)
router.include_router(ingestion.router)
router.include_router(processing.router)
