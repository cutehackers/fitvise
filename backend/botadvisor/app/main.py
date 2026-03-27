"""Application factory for the canonical BotAdvisor API."""

from __future__ import annotations

from fastapi import FastAPI

from botadvisor.app.api.middleware import RequestContextMiddleware
from botadvisor.app.api.router import router
from botadvisor.app.health import RuntimeHealthService


def create_app(*, chat_service=None, health_service=None, llm_service=None, vector_store_checker=None) -> FastAPI:
    """Create the canonical BotAdvisor FastAPI application."""
    app = FastAPI(title="BotAdvisor API", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)
    app.state.chat_service = chat_service
    app.state.health_service = health_service or RuntimeHealthService(
        chat_service=chat_service,
        llm_service=llm_service,
        vector_store_checker=vector_store_checker,
    )
    app.include_router(router)
    return app


app = create_app()
