"""Application factory for the canonical BotAdvisor API."""

from __future__ import annotations

from fastapi import FastAPI

from botadvisor.app.api.router import router
from botadvisor.app.health import RuntimeHealthService


def create_app(*, chat_service=None, health_service=None) -> FastAPI:
    """Create the canonical BotAdvisor FastAPI application."""
    app = FastAPI(title="BotAdvisor API", version="0.1.0")
    app.state.chat_service = chat_service
    app.state.health_service = health_service or RuntimeHealthService(chat_service=chat_service)
    app.include_router(router)
    return app


app = create_app()
