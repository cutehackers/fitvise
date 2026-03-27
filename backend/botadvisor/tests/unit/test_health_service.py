from __future__ import annotations

import pytest


class FakeLlmService:
    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_runtime_health_reports_component_checks():
    from botadvisor.app.health import RuntimeHealthService

    chat_service = object()

    service = RuntimeHealthService(
        chat_service=chat_service,
        llm_service=FakeLlmService(),
        vector_store_checker=lambda: True,
    )

    response = await service.get_status()

    assert response.status == "healthy"
    assert response.retrieval_available is True
    assert response.checks == {
        "retrieval": {"status": "healthy"},
        "vector_store": {"status": "healthy"},
        "llm_path": {"status": "healthy"},
    }
