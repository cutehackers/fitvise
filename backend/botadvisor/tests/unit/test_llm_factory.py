from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_create_llm_service_returns_ollama_service(monkeypatch):
    from botadvisor.app.llm import factory as factory_module

    captured: dict[str, object] = {}

    class FakeOllamaChatService:
        def __init__(self, *, base_url: str, model: str, temperature: float):
            captured["base_url"] = base_url
            captured["model"] = model
            captured["temperature"] = temperature

    monkeypatch.setattr(factory_module, "OllamaChatService", FakeOllamaChatService)

    settings = SimpleNamespace(
        llm_provider="ollama",
        llm_base_url="http://localhost:11434",
        llm_model="llama3.2:3b",
        llm_temperature=0.2,
    )

    service = factory_module.create_llm_service(settings)

    assert isinstance(service, FakeOllamaChatService)
    assert captured == {
        "base_url": "http://localhost:11434",
        "model": "llama3.2:3b",
        "temperature": 0.2,
    }


def test_create_llm_service_rejects_unknown_provider():
    from botadvisor.app.llm.factory import create_llm_service

    settings = SimpleNamespace(
        llm_provider="unknown",
        llm_base_url="http://localhost:11434",
        llm_model="llama3.2:3b",
        llm_temperature=0.2,
    )

    with pytest.raises(ValueError, match="Unsupported llm provider"):
        create_llm_service(settings)
