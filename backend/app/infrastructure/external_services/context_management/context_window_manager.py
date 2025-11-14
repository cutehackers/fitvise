"""Context window management for LLM generation (Task 3.1.3).

Manages context window limits and truncation strategies to fit retrieved documents
into LLM context windows while preserving maximum information.
"""

import logging
from typing import List, Literal
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Context window configuration.

    Attributes:
        max_tokens: Maximum tokens allowed in context window
        reserve_tokens: Tokens reserved for response generation
        truncation_strategy: Strategy for handling context overflow
            - "recent": Keep most recent documents
            - "relevant": Keep highest relevance documents (assumes sorted)
            - "summarize": Summarize documents to fit (placeholder for future enhancement)
    """

    max_tokens: int = 4000  # Default for most Ollama models (llama3.2, mistral)
    reserve_tokens: int = 500  # Reserve for response generation
    truncation_strategy: Literal["recent", "relevant", "summarize"] = "relevant"


class ContextWindowManager:
    """Manages LLM context window limits.

    Handles truncation and fitting of retrieved documents into context window
    while preserving as much relevant information as possible.

    Token estimation: 1 token ≈ 4 characters (approximation for English text)
    """

    CHARS_PER_TOKEN = 4  # Rough estimate for English text

    def __init__(self, config: ContextWindow):
        """Initialize context window manager.

        Args:
            config: Context window configuration
        """
        self.config = config
        self._available_tokens = config.max_tokens - config.reserve_tokens

        logger.info(
            "ContextWindowManager initialized: max_tokens=%d, reserve=%d, strategy=%s",
            config.max_tokens,
            config.reserve_tokens,
            config.truncation_strategy,
        )

    def fit_to_window(
        self, documents: List[Document], user_query: str, system_prompt: str = ""
    ) -> str:
        """Fit documents into context window with user query.

        Args:
            documents: Retrieved documents to fit into context
            user_query: User's query text
            system_prompt: System prompt text (if any)

        Returns:
            Truncated context string that fits within window

        Raises:
            ValueError: If query + system prompt exceed available tokens
        """
        # Calculate tokens for query and system prompt
        query_tokens = self._estimate_tokens(user_query)
        system_tokens = self._estimate_tokens(system_prompt)
        overhead_tokens = query_tokens + system_tokens

        if overhead_tokens >= self._available_tokens:
            raise ValueError(
                f"Query and system prompt ({overhead_tokens} tokens) exceed "
                f"available context window ({self._available_tokens} tokens)"
            )

        available_for_context = self._available_tokens - overhead_tokens

        logger.debug(
            "Fitting %d documents into %d tokens (query=%d, system=%d, available=%d)",
            len(documents),
            self._available_tokens,
            query_tokens,
            system_tokens,
            available_for_context,
        )

        # Apply truncation strategy
        if self.config.truncation_strategy == "recent":
            context = self._truncate_recent(documents, available_for_context)
        elif self.config.truncation_strategy == "relevant":
            context = self._truncate_by_relevance(documents, available_for_context)
        else:  # summarize
            context = self._summarize(documents, available_for_context)

        actual_tokens = self._estimate_tokens(context)
        logger.debug(
            "Context fitted: strategy=%s, docs_used=%d, context_tokens=%d",
            self.config.truncation_strategy,
            self._count_documents_in_context(context),
            actual_tokens,
        )

        return context

    def _truncate_recent(self, documents: List[Document], max_tokens: int) -> str:
        """Keep most recent documents that fit in window.

        Assumes documents are ordered chronologically, keeps newest first.

        Args:
            documents: List of documents (most recent last)
            max_tokens: Maximum tokens available for context

        Returns:
            Context string with most recent documents
        """
        context_parts = []
        current_tokens = 0

        # Process documents in reverse order (most recent first)
        for doc in reversed(documents):
            doc_tokens = self._estimate_tokens(doc.page_content)

            if current_tokens + doc_tokens <= max_tokens:
                # Add document at beginning to maintain chronological order
                context_parts.insert(0, doc.page_content)
                current_tokens += doc_tokens
            else:
                # Can't fit more documents
                break

        logger.debug(
            "Truncated by recency: kept %d/%d documents, %d tokens",
            len(context_parts),
            len(documents),
            current_tokens,
        )

        return "\n\n".join(context_parts)

    def _truncate_by_relevance(
        self, documents: List[Document], max_tokens: int
    ) -> str:
        """Keep highest relevance documents that fit in window.

        Assumes documents are already sorted by relevance (highest first).

        Args:
            documents: List of documents sorted by relevance (descending)
            max_tokens: Maximum tokens available for context

        Returns:
            Context string with most relevant documents
        """
        context_parts = []
        current_tokens = 0

        # Process documents in order (highest relevance first)
        for doc in documents:
            doc_tokens = self._estimate_tokens(doc.page_content)

            if current_tokens + doc_tokens <= max_tokens:
                context_parts.append(doc.page_content)
                current_tokens += doc_tokens
            else:
                # Can't fit more documents
                break

        logger.debug(
            "Truncated by relevance: kept %d/%d documents, %d tokens",
            len(context_parts),
            len(documents),
            current_tokens,
        )

        return "\n\n".join(context_parts)

    def _summarize(self, documents: List[Document], max_tokens: int) -> str:
        """Summarize documents to fit window.

        Placeholder for future enhancement with actual summarization.
        Currently falls back to relevance-based truncation.

        Args:
            documents: List of documents to summarize
            max_tokens: Maximum tokens available for context

        Returns:
            Summarized context string
        """
        logger.warning(
            "Summarization not yet implemented, falling back to relevance truncation"
        )
        return self._truncate_by_relevance(documents, max_tokens)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Uses simple heuristic: 1 token ≈ 4 characters for English text.
        This is approximate but sufficient for context management.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // self.CHARS_PER_TOKEN

    def _count_documents_in_context(self, context: str) -> int:
        """Count number of documents in context string.

        Documents are separated by double newlines.

        Args:
            context: Context string with documents

        Returns:
            Number of documents in context
        """
        if not context:
            return 0
        # Count document separators + 1
        return context.count("\n\n") + 1

    def get_available_tokens(self) -> int:
        """Get available tokens for context after reserving for response.

        Returns:
            Available tokens for context
        """
        return self._available_tokens

    def estimate_document_tokens(self, document: Document) -> int:
        """Estimate token count for a document.

        Args:
            document: Document to estimate tokens for

        Returns:
            Estimated token count
        """
        return self._estimate_tokens(document.page_content)
