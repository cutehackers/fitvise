"""Execution boundary for registered tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from botadvisor.app.tools.contracts import ToolExecutionRequest, ToolExecutionResult
from botadvisor.app.tools.registry import ToolRegistry


@dataclass
class ToolExecutor:
    """Execute named tools through the canonical registry boundary."""

    registry: ToolRegistry

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Validate arguments against one tool definition and execute it."""
        definition = self.registry.get(request.tool_name)

        try:
            parsed_arguments = definition.input_model(**request.arguments)
        except ValidationError:
            raise

        output = definition.handler(parsed_arguments)

        if isinstance(output, definition.output_model):
            payload: dict[str, Any] = output.model_dump()
        else:
            validated_output = definition.output_model(**output)
            payload = validated_output.model_dump()

        return ToolExecutionResult(
            tool_name=definition.name,
            payload=payload,
            trace_metadata={"tool_name": definition.name},
        )
