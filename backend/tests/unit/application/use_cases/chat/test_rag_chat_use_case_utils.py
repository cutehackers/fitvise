from app.application.use_cases.chat.rag_chat_use_case import RagChatUseCase


def test_normalize_citation_content_truncates_and_strips_whitespace():
    text = "Line1\n   Line2 \t Line3"
    normalized = RagChatUseCase._normalize_citation_content(text, max_length=10)

    assert normalized == "Line1 Line", "should normalize whitespace and truncate to length"


def test_normalize_citation_content_handles_empty():
    assert RagChatUseCase._normalize_citation_content("") == ""
