"""Prompt assembly for retrieval-backed chat answers."""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from botadvisor.app.chat.schemas import SourceCitation


SYSTEM_PROMPT = """You are Fitvise, a fitness and wellness assistant.
Answer the user using the retrieved context when it is relevant.
Use citation markers like [1] when you rely on a retrieved source.
If the retrieved context is empty or does not answer the question, say that clearly.

Retrieved context:
{context}
"""


def build_chat_prompt_messages(*, question: str, citations: list[SourceCitation]) -> list[BaseMessage]:
    """Build prompt messages for a retrieval-backed LLM answer."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    return prompt.format_messages(
        context=format_citation_context(citations),
        question=question,
    )


def format_citation_context(citations: list[SourceCitation]) -> str:
    """Format citations into a prompt-friendly context block."""
    if not citations:
        return "No retrieved context was found."

    return "\n\n".join(f"[{citation.index}] {citation.content}" for citation in citations)
