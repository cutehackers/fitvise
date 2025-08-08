"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1.fitvise import chat

# Create main API router
router = APIRouter()

# Include endpoint routers
router.include_router(chat.router, prefix="/fitvise")
