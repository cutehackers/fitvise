"""Small Ollama-backed chat service for the canonical BotAdvisor runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama


class OllamaChatService:
    """Wrap ChatOllama behind a small BotAdvisor-oriented interface."""

    def __init__(self, *, base_url: str, model: str, temperature: float) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.client = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    def generate(self, messages: list[BaseMessage]) -> str:
        """Generate a non-streaming answer from prompt messages."""
        response = self.client.invoke(messages)
        return response.content or ""

    async def generate_stream(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        """Generate a streaming answer from prompt messages."""
        async for chunk in self.client.astream(messages):
            if chunk.content:
                yield chunk.content

    async def health_check(self) -> bool:
        """Return whether the configured Ollama path is reachable."""
        try:
            await self.client.ainvoke([HumanMessage(content="ping")])
            return True
        except Exception:
            return False
