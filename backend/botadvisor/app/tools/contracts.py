"""Explicit tool runtime contracts for the canonical agent kernel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ToolExecutionRequest(BaseModel):
    """Validated request to execute a named tool with explicit arguments."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_name must not be blank")
        return normalized


class ToolExecutionResult(BaseModel):
    """Structured result returned by a named tool execution."""

    tool_name: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of one registered tool in the canonical runtime."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]

    def __post_init__(self) -> None:
        name = self.name.strip()
        description = self.description.strip()

        if not name:
            raise ValueError("tool name must not be blank")
        if not description:
            raise ValueError("tool description must not be blank")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
