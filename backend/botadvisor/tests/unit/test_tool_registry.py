from __future__ import annotations

import pytest
from pydantic import BaseModel


class EchoToolInput(BaseModel):
    message: str


class EchoToolOutput(BaseModel):
    echoed: str


def build_echo_definition():
    from botadvisor.app.tools.contracts import ToolDefinition

    return ToolDefinition(
        name="echo",
        description="Echo back a message for testing.",
        input_model=EchoToolInput,
        output_model=EchoToolOutput,
        handler=lambda payload: EchoToolOutput(echoed=payload.message),
    )


def test_tool_registry_registers_and_resolves_named_tools():
    from botadvisor.app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(build_echo_definition())

    definition = registry.get("echo")

    assert definition.name == "echo"
    assert [tool.name for tool in registry.list_definitions()] == ["echo"]


def test_tool_registry_rejects_duplicate_tool_names():
    from botadvisor.app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(build_echo_definition())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(build_echo_definition())


def test_tool_executor_validates_input_and_runs_registered_tool():
    from botadvisor.app.tools.contracts import ToolExecutionRequest
    from botadvisor.app.tools.executor import ToolExecutor
    from botadvisor.app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(build_echo_definition())
    executor = ToolExecutor(registry=registry)

    result = executor.execute(ToolExecutionRequest(tool_name="echo", arguments={"message": "hello"}))

    assert result.tool_name == "echo"
    assert result.payload == {"echoed": "hello"}


def test_tool_executor_rejects_unknown_tool():
    from botadvisor.app.tools.contracts import ToolExecutionRequest
    from botadvisor.app.tools.executor import ToolExecutor
    from botadvisor.app.tools.registry import ToolRegistry

    executor = ToolExecutor(registry=ToolRegistry())

    with pytest.raises(KeyError, match="Unknown tool"):
        executor.execute(ToolExecutionRequest(tool_name="missing", arguments={"message": "hello"}))
