"""FastAPI dependency helpers for the canonical BotAdvisor runtime."""

from __future__ import annotations

from fastapi import HTTPException, Request, status


def get_chat_service(request: Request):
    """Return the configured chat service or raise if unavailable."""
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service is not configured.")
    return service


def get_health_service(request: Request):
    """Return the configured health service."""
    return request.app.state.health_service
