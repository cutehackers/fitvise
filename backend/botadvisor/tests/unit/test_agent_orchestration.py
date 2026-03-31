from __future__ import annotations

from unittest.mock import Mock

import pytest

from botadvisor.app.tools.contracts import ToolExecutionRequest, ToolExecutionResult


def test_single_step_orchestrator_returns_direct_answer_without_tool_call():
    from botadvisor.app.agent.orchestration import SingleStepAgentOrchestrator
    from botadvisor.app.agent.schemas import AgentDecision, AgentTurnResult

    llm_service = Mock()
    llm_service.decide_tool_calls.return_value = AgentDecision(tool_calls=())
    llm_service.generate_direct_answer.return_value = "Direct answer."

    executor = Mock()
    orchestrator = SingleStepAgentOrchestrator(llm_service=llm_service, tool_executor=executor)

    result = orchestrator.run(message="What is progressive overload?")

    assert isinstance(result, AgentTurnResult)
    assert result.answer == "Direct answer."
    assert result.used_tool_name is None
    executor.execute.assert_not_called()


def test_single_step_orchestrator_executes_one_tool_call_then_generates_answer():
    from botadvisor.app.agent.orchestration import SingleStepAgentOrchestrator
    from botadvisor.app.agent.schemas import AgentDecision

    llm_service = Mock()
    llm_service.decide_tool_calls.return_value = AgentDecision(
        tool_calls=(ToolExecutionRequest(tool_name="retrieval", arguments={"query": "protein intake"}),)
    )
    llm_service.generate_tool_answer.return_value = "Tool-based answer."

    executor = Mock()
    executor.execute.return_value = ToolExecutionResult(
        tool_name="retrieval",
        payload={"results": [{"chunk_id": "chunk-1"}]},
    )

    orchestrator = SingleStepAgentOrchestrator(llm_service=llm_service, tool_executor=executor)
    result = orchestrator.run(message="What protein intake is good?")

    executor.execute.assert_called_once()
    llm_service.generate_tool_answer.assert_called_once()
    assert result.answer == "Tool-based answer."
    assert result.used_tool_name == "retrieval"
    assert result.tool_result is not None


def test_single_step_orchestrator_rejects_multiple_tool_calls():
    from botadvisor.app.agent.orchestration import SingleStepAgentOrchestrator
    from botadvisor.app.agent.schemas import AgentDecision

    llm_service = Mock()
    llm_service.decide_tool_calls.return_value = AgentDecision(
        tool_calls=(
            ToolExecutionRequest(tool_name="retrieval", arguments={"query": "protein intake"}),
            ToolExecutionRequest(tool_name="retrieval", arguments={"query": "creatine"}),
        )
    )

    orchestrator = SingleStepAgentOrchestrator(llm_service=llm_service, tool_executor=Mock())

    with pytest.raises(ValueError, match="at most one tool call"):
        orchestrator.run(message="Compare protein and creatine.")
