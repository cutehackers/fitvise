"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import workout

# Create main API router
router = APIRouter()

# Include endpoint routers
router.include_router(workout.router, prefix="/workout")