"""Runtime health helpers for the canonical BotAdvisor API."""

from __future__ import annotations

from dataclasses import dataclass

from botadvisor.app.chat.schemas import HealthResponse
from botadvisor.app.core.config import get_settings


@dataclass
class RuntimeHealthService:
    """Provide a minimal health status for the canonical runtime."""

    chat_service: object | None = None

    async def get_status(self) -> HealthResponse:
        settings = get_settings()
        retrieval_available = self.chat_service is not None
        status = "healthy" if retrieval_available else "degraded"
        return HealthResponse(
            status=status,
            service="botadvisor-api",
            retrieval_available=retrieval_available,
            langfuse_enabled=settings.langfuse_enabled,
        )
