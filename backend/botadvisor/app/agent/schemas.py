"""Schemas for the canonical single-step agent runtime."""

from __future__ import annotations

from dataclasses import dataclass

from botadvisor.app.tools.contracts import ToolExecutionRequest, ToolExecutionResult


@dataclass(frozen=True)
class AgentDecision:
    """LLM decision for one agent turn."""

    tool_calls: tuple[ToolExecutionRequest, ...] = ()


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of one single-step agent turn."""

    answer: str
    used_tool_name: str | None = None
    tool_result: ToolExecutionResult | None = None
