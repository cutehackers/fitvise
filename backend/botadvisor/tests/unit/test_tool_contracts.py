from __future__ import annotations

import pytest
from pydantic import BaseModel


class FakeToolInput(BaseModel):
    query: str


class FakeToolOutput(BaseModel):
    answer: str


def test_tool_definition_requires_explicit_identity():
    from botadvisor.app.tools.contracts import ToolDefinition

    definition = ToolDefinition(
        name="retrieval",
        description="Retrieve supporting chunks for a user question.",
        input_model=FakeToolInput,
        output_model=FakeToolOutput,
        handler=lambda payload: FakeToolOutput(answer=payload.query),
    )

    assert definition.name == "retrieval"
    assert definition.description == "Retrieve supporting chunks for a user question."


def test_tool_definition_rejects_blank_name_or_description():
    from botadvisor.app.tools.contracts import ToolDefinition

    with pytest.raises(ValueError, match="tool name must not be blank"):
        ToolDefinition(
            name="",
            description="Retrieve supporting chunks for a user question.",
            input_model=FakeToolInput,
            output_model=FakeToolOutput,
            handler=lambda payload: FakeToolOutput(answer=payload.query),
        )

    with pytest.raises(ValueError, match="tool description must not be blank"):
        ToolDefinition(
            name="retrieval",
            description="",
            input_model=FakeToolInput,
            output_model=FakeToolOutput,
            handler=lambda payload: FakeToolOutput(answer=payload.query),
        )


def test_tool_execution_request_requires_named_tool_call():
    from botadvisor.app.tools.contracts import ToolExecutionRequest

    request = ToolExecutionRequest(tool_name="retrieval", arguments={"query": "protein intake"})

    assert request.tool_name == "retrieval"
    assert request.arguments == {"query": "protein intake"}

    with pytest.raises(ValueError, match="tool_name must not be blank"):
        ToolExecutionRequest(tool_name="", arguments={"query": "protein intake"})


def test_tool_execution_result_wraps_structured_output():
    from botadvisor.app.tools.contracts import ToolExecutionResult

    result = ToolExecutionResult(
        tool_name="retrieval",
        payload={"results": [{"chunk_id": "chunk-1"}]},
        trace_metadata={"top_k": 3},
    )

    assert result.tool_name == "retrieval"
    assert result.payload["results"][0]["chunk_id"] == "chunk-1"
    assert result.trace_metadata == {"top_k": 3}
