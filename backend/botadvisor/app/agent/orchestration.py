"""Single-step orchestration for the canonical retrieval-first agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from botadvisor.app.agent.schemas import AgentTurnResult


@dataclass
class SingleStepAgentOrchestrator:
    """Coordinate at most one tool call before final answer generation."""

    llm_service: Any
    tool_executor: Any

    def run(self, *, message: str) -> AgentTurnResult:
        """Run one agent turn with zero or one tool call."""
        decision = self.llm_service.decide_tool_calls(message)

        if len(decision.tool_calls) > 1:
            raise ValueError("Single-step orchestration supports at most one tool call per turn.")

        if not decision.tool_calls:
            answer = self.llm_service.generate_direct_answer(message)
            return AgentTurnResult(answer=answer)

        tool_request = decision.tool_calls[0]
        tool_result = self.tool_executor.execute(tool_request)
        answer = self.llm_service.generate_tool_answer(message, tool_result)
        return AgentTurnResult(
            answer=answer,
            used_tool_name=tool_result.tool_name,
            tool_result=tool_result,
        )
