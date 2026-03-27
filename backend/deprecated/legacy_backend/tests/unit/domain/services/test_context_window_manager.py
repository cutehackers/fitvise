import pytest
from langchain_core.documents import Document

from app.domain.services.context_window_manager import (
    ContextWindow,
    ContextWindowManager,
)


def test_fit_to_window_no_documents_returns_empty():
    manager = ContextWindowManager(ContextWindow(max_tokens=100, reserve_tokens=10))

    context = manager.fit_to_window([], user_query="What is up?", system_prompt="system")

    assert context == ""


def test_fit_to_window_uses_summarizer_when_configured():
    calls = []

    def summarizer(docs, max_tokens):
        calls.append((len(docs), max_tokens))
        return "summary"

    manager = ContextWindowManager(
        ContextWindow(
            max_tokens=100,
            reserve_tokens=10,
            truncation_strategy="summarize",
        ),
        summarizer=summarizer,
    )

    context = manager.fit_to_window(
        [Document(page_content="a" * 20)], user_query="q", system_prompt="sys"
    )

    assert context == "summary"
    assert calls, "summarizer should be invoked when configured"
