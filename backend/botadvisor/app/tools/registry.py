"""Named tool registry for the canonical agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from botadvisor.app.tools.contracts import ToolDefinition


@dataclass
class ToolRegistry:
    """Registry of first-class tools available to the agent runtime."""

    _definitions: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> None:
        """Register one tool definition under its explicit name."""
        if definition.name in self._definitions:
            raise ValueError(f"Tool '{definition.name}' is already registered.")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        """Return one registered tool definition by name."""
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list_definitions(self) -> list[ToolDefinition]:
        """Return registered tool definitions in insertion order."""
        return list(self._definitions.values())
