from __future__ import annotations

from pydantic import BaseModel


class FakeToolInput(BaseModel):
    query: str
    top_k: int = 5


class FakeToolOutput(BaseModel):
    answer: str


def build_fake_tool_definition():
    from botadvisor.app.tools.contracts import ToolDefinition

    return ToolDefinition(
        name="retrieval",
        description="Retrieve supporting chunks for a user question.",
        input_model=FakeToolInput,
        output_model=FakeToolOutput,
        handler=lambda payload: FakeToolOutput(answer=payload.query),
    )


def test_ollama_chat_service_parses_one_tool_call_from_model_output():
    from botadvisor.app.llm.ollama import OllamaChatService

    service = object.__new__(OllamaChatService)
    service.generate = lambda _messages: (
        '{"tool_name":"retrieval","arguments":{"query":"protein intake","top_k":3}}'
    )

    decision = service.decide_tool_calls("What protein intake is good?", [build_fake_tool_definition()])

    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].tool_name == "retrieval"
    assert decision.tool_calls[0].arguments == {"query": "protein intake", "top_k": 3}


def test_ollama_chat_service_returns_no_tool_call_for_null_decision():
    from botadvisor.app.llm.ollama import OllamaChatService

    service = object.__new__(OllamaChatService)
    service.generate = lambda _messages: '{"tool_name": null, "arguments": {}}'

    decision = service.decide_tool_calls("Hello", [build_fake_tool_definition()])

    assert decision.tool_calls == ()


def test_ollama_chat_service_generates_tool_answer_from_structured_payload():
    from botadvisor.app.llm.ollama import OllamaChatService
    from botadvisor.app.tools.contracts import ToolExecutionResult

    captured: dict[str, object] = {}
    service = object.__new__(OllamaChatService)

    def fake_generate(messages):
        captured["messages"] = messages
        return "Final answer."

    service.generate = fake_generate

    answer = service.generate_tool_answer(
        "What protein intake is good?",
        ToolExecutionResult(
            tool_name="retrieval",
            payload={"results": [{"content": "Aim for 1.6 g/kg.", "chunk_id": "chunk-1"}]},
        ),
    )

    assert answer == "Final answer."
    assert "What protein intake is good?" in captured["messages"][1].content
    assert "Aim for 1.6 g/kg." in captured["messages"][0].content
