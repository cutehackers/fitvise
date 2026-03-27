"""Runtime health status assembly for the canonical BotAdvisor API."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from botadvisor.app.chat.schemas import HealthResponse
from botadvisor.app.core.config import get_settings


@dataclass
class RuntimeHealthService:
    """Provide a minimal health status for the canonical runtime."""

    chat_service: object | None = None
    llm_service: object | None = None
    vector_store_checker: Callable[[], Any] | None = None

    async def get_status(self) -> HealthResponse:
        settings = get_settings()
        retrieval_available = self.chat_service is not None
        vector_store_available = await self._check_vector_store()
        llm_available = await self._check_llm_path()
        checks = {
            "retrieval": {"status": "healthy" if retrieval_available else "degraded"},
            "vector_store": {"status": "healthy" if vector_store_available else "degraded"},
            "llm_path": {"status": "healthy" if llm_available else "degraded"},
        }
        status = "healthy" if retrieval_available and vector_store_available and llm_available else "degraded"
        return HealthResponse(
            status=status,
            service="botadvisor-api",
            retrieval_available=retrieval_available,
            langfuse_enabled=settings.langfuse_enabled,
            checks=checks,
        )

    async def _check_vector_store(self) -> bool:
        if self.vector_store_checker is None:
            return False
        result = self.vector_store_checker()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def _check_llm_path(self) -> bool:
        if self.llm_service is None:
            return False
        health_check = getattr(self.llm_service, "health_check", None)
        if health_check is None:
            return False
        result = health_check()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
