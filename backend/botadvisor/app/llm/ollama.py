"""Small Ollama-backed chat service for the canonical BotAdvisor runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from botadvisor.app.agent.schemas import AgentDecision
from botadvisor.app.tools.contracts import ToolDefinition, ToolExecutionRequest, ToolExecutionResult


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

    def decide_tool_calls(self, message: str, tool_definitions: list[ToolDefinition]) -> AgentDecision:
        """Ask the model for a single tool decision in a JSON envelope."""
        if not tool_definitions:
            return AgentDecision(tool_calls=())

        tool_names = ", ".join(tool_definition.name for tool_definition in tool_definitions)
        decision_messages = [
            SystemMessage(
                content=(
                    "Decide whether one tool call is needed. "
                    "Return JSON only with shape "
                    '{"tool_name": <string or null>, "arguments": <object>}. '
                    f"Allowed tools: {tool_names}."
                )
            ),
            HumanMessage(content=message),
        ]
        raw_decision = self.generate(decision_messages)

        try:
            payload = json.loads(raw_decision)
        except json.JSONDecodeError:
            return AgentDecision(tool_calls=())

        tool_name = payload.get("tool_name")
        if not tool_name:
            return AgentDecision(tool_calls=())

        if tool_name not in {tool_definition.name for tool_definition in tool_definitions}:
            return AgentDecision(tool_calls=())

        return AgentDecision(
            tool_calls=(
                ToolExecutionRequest(
                    tool_name=tool_name,
                    arguments=payload.get("arguments") or {"query": message},
                ),
            )
        )

    def generate_direct_answer(self, message: str) -> str:
        """Generate a direct answer when no tool call is used."""
        return self.generate([HumanMessage(content=message)])

    def generate_tool_answer(self, message: str, tool_result: ToolExecutionResult) -> str:
        """Generate a final answer grounded in one tool result."""
        return self.generate(self._build_tool_answer_messages(message, tool_result))

    async def generate_tool_answer_stream(
        self,
        message: str,
        tool_result: ToolExecutionResult,
    ) -> AsyncIterator[str]:
        """Stream a final answer grounded in one tool result."""
        async for token in self.generate_stream(self._build_tool_answer_messages(message, tool_result)):
            yield token

    @staticmethod
    def _build_tool_answer_messages(
        message: str,
        tool_result: ToolExecutionResult,
    ) -> list[BaseMessage]:
        tool_context = json.dumps(tool_result.payload, ensure_ascii=False)
        return [
            SystemMessage(
                content=(
                    "Use the tool result to answer the user question. "
                    "Cite the tool-provided evidence when possible.\n\n"
                    f"Tool: {tool_result.tool_name}\n"
                    f"Tool Result: {tool_context}"
                )
            ),
            HumanMessage(content=message),
        ]
