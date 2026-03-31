"""Runtime-facing service wrapper for the canonical agent path."""

from __future__ import annotations

from dataclasses import dataclass

from botadvisor.app.agent.orchestration import SingleStepAgentOrchestrator
from botadvisor.app.agent.schemas import AgentTurnResult


@dataclass
class AgentRuntimeService:
    """Thin service facade around the single-step orchestrator."""

    orchestrator: SingleStepAgentOrchestrator

    def run_turn(self, *, message: str) -> AgentTurnResult:
        """Run one chat turn through the canonical single-step agent runtime."""
        return self.orchestrator.run(message=message)
