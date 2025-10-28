"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.fitvise import chat
from app.api.v1.rag import data_sources, ingestion, orchestration, storage

# Create main API router
router = APIRouter()

# Include endpoint routers (processing and legacy pipeline endpoints removed)
router.include_router(chat.router, prefix="/fitvise")
router.include_router(data_sources.router)
router.include_router(ingestion.router)
router.include_router(storage.router)
router.include_router(orchestration.router)
router.include_router(embeddings_router)
