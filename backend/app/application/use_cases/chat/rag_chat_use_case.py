"""RAG chat use case with direct LangChain integration for clean architecture.

This module implements RagChatUseCase as a complete replacement for RAG orchestrators
with direct LangChain integration, document retrieval, and source citation generation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from app.domain.llm.exceptions import ChatOrchestratorError, MessageValidationError
from app.domain.llm.interfaces.llm_service import LLMService
from app.domain.entities.message_role import MessageRole
from app.domain.services.session_service import SessionService
from app.domain.services.context_window_manager import (
    ContextWindowManager,
)
from langchain_core.callbacks import BaseCallbackHandler
from app.schemas.chat import ChatRequest, RagChatResponse, SourceCitation

logger = logging.getLogger(__name__)


# Token limits for common models
MAX_TOKENS_TABLE = {
    "llama3.2:3b": 128000,
    "llama3.1:8b": 128000,
    "llama3.1:70b": 128000,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "mistral:7b": 8192,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
}

# Default max token length for models without specific limit
DEFAULT_MAX_TOKEN_LENGTH = 8192


def _get_current_timestamp() -> str:
    """Generate current timestamp in ISO 8601 format.

    Returns:
        Current timestamp in ISO 8601 format (UTC)
    """
    return datetime.now(timezone.utc).isoformat()


class RagChatUseCase:
    """Clean architecture RAG chat use case with direct LangChain integration.

    Implements all existing RAG functionality with proper clean architecture:
    - Session management through SessionService
    - Document retrieval through injected BaseRetriever
    - Context management through ContextWindowManager
    - LLM integration through LLMService
    - Full streaming support with LangChain integration
    """

    # RAG prompt template with citation instructions
    SYSTEM_PROMPT = """You are a helpful fitness assistant. Answer questions using the provided context.

IMPORTANT: When using information from the context, cite your sources using [1], [2], etc.

Context:
{context}

If the context doesn't contain relevant information, say so clearly."""

    def __init__(
        self,
        llm_service: LLMService,
        retriever: BaseRetriever,
        context_manager: ContextWindowManager,
        session_service: SessionService,
        callback_handler: Optional[BaseCallbackHandler] = None,
        turns_window: int = 10,
        max_session_age_hours: int = 24,
    ):
        """Initialize the RAG chat use case.

        Args:
            llm_service: LLM service for generating responses
            retriever: Base retriever for document retrieval
            context_manager: Context window manager for fitting documents into context
            session_service: Service for session management
            callback_handler: Optional LangChain callback handler for analytics
            turns_window: Number of conversation turns to keep in memory (default: 10)
            max_session_age_hours: Maximum age before sessions expire (default: 24)
        """
        self._llm_service = llm_service
        self._retriever = retriever
        self._context_manager = context_manager
        self._session_service = session_service
        self._callback_handler = callback_handler
        self._turns_window = turns_window
        self._max_session_age_hours = max_session_age_hours

        # RAG-specific prompt template with context placeholder
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are Fitvise, a fitness AI assistant helping users reformulate their questions "
                "based on conversation history for better context retrieval.\n\n"
                "Your task is to transform the user's latest question into an effective search query "
                "that captures their fitness-related intent, considering the full conversation context.\n\n"
                "Guidelines:\n"
                "- Include relevant fitness concepts from the conversation history\n"
                "- Maintain the core intent of the original question\n"
                "- Add specific fitness terminology when helpful\n"
                "- Keep the query concise but comprehensive\n"
                "- Focus on exercise, nutrition, wellness, or fitness goals\n"
                "- Formulate a standalone question which can be understood without the chat history\n"
                "- Do NOT answer the question, just reformulate it if needed and otherwise return it as is\n\n"
            )),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ])

        # Store LLM instance for direct access
        self._llm = llm_service.llm_instance

        # Retriever
        self._history_aware_retriever = create_history_aware_retriever(self._llm, self._retriever, self._prompt)

        # 5. RAG prompts (documents + history)
        self._qa_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are Fitvise, an expert fitness AI assistant. Provide comprehensive, "
                "accurate answers using the retrieved fitness context and conversation history.\n\n"
                "Response Guidelines:\n"
                "- Base answers primarily on the provided context when relevant and available\n"
                "- Cite sources using [1], [2] format when using contextual information\n"
                "- Include specific fitness details, exercises, or nutritional information from context\n"
                "- If no relevant context was found, state this clearly and provide general fitness guidance\n"
                "- If the retrieved context is empty or irrelevant, rely on your fitness expertise and conversation history\n"
                "- Maintain a supportive, professional, and safety-focused tone\n"
                "- Prioritize evidence-based fitness information\n\n"
                "Retrieved Context:\n{context}"
            )),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

        # RAG chain. merge documents to handle by LLM
        self._question_answer_chain = create_stuff_documents_chain(self._llm, self._qa_prompt)

        # Create chain for retrieving context and processing user input
        self._chain = create_retrieval_chain(self._history_aware_retriever, self._question_answer_chain)
        # self._chain = (
        #     RunnableParallel({
        #         "context": self._retriever,
        #         "input": RunnablePassthrough()
        #     })            
        #     | self._prompt 
        #     | self._llm
        # )
        
        # Create runnable with history using SessionService's persistent history
        self._runnable_with_history = RunnableWithMessageHistory(
            self._chain,
            lambda session_id: self._session_service.get_session_history(session_id),                
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

        logger.info(
            "RagChatUseCase initialized with retriever, context manager, and session service"
        )


    async def _retrieve(self, query: str) -> Tuple[str, List[Document]]:
        """Retrieve context documents for a query.

        Args:
            query: User query to retrieve context for

        Returns:
            Tuple of (context_string, retrieved_documents)

        Raises:
            ChatOrchestratorError: If retrieval fails
        """
        try:
            # Step 1: Retrieve relevant documents using LangChain retriever interface
            documents = await self._retriever.ainvoke(query)

            if not documents:
                logger.warning("No documents retrieved for query: %s", query)
                return "", []

            logger.debug("Retrieved %d documents for query", len(documents))

            # Step 2: Fit documents into context window
            try:
                context = self._context_manager.fit_to_window(
                    documents,
                    user_query=query,
                    system_prompt=self.SYSTEM_PROMPT,
                )
            except ValueError as e:
                logger.warning(
                    "Failed to fit documents to context window: %s. Using truncated context.",
                    str(e),
                )
                # Fallback: Use first few documents that fit
                context_parts = []
                current_tokens = 0
                max_tokens = self._context_manager.get_available_tokens()

                for doc in documents:
                    doc_tokens = self._context_manager.estimate_document_tokens(doc)
                    if current_tokens + doc_tokens <= max_tokens:
                        context_parts.append(doc.page_content)
                        current_tokens += doc_tokens
                    else:
                        break

                context = "\n\n".join(context_parts)

            return context, documents

        except Exception as e:
            logger.error("Failed to retrieve context: %s", str(e))
            raise ChatOrchestratorError(
                f"Failed to retrieve context: {str(e)}",
                original_error=e,
            )

    def _create_source_citations(
        self, documents: List[Document]
    ) -> List[SourceCitation]:
        """Create source citations from retrieved documents.

        Args:
            documents: Retrieved documents

        Returns:
            List of source citations
        """
        citations = []
        for idx, doc in enumerate(documents, 1):
            metadata = doc.metadata or {}
            citations.append(
                SourceCitation(
                    index=idx,
                    content=doc.page_content[:500],  # Truncate for display
                    similarity_score=metadata.get("_distance", 0.0),
                    document_id=metadata.get("document_id", f"doc_{idx}"),
                    chunk_id=metadata.get("chunk_id", f"chunk_{idx}"),
                    metadata=metadata,
                )
            )
        return citations

    async def chat(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[RagChatResponse, None]:
        """Process a chat message with RAG and generate streaming responses.

        Args:
            request: Chat request with message and session info

        Yields:
            Streaming RAG chat responses with source citations

        Raises:
            MessageValidationError: If request validation fails
            ChatOrchestratorError: If RAG processing fails
            Exception: For unexpected errors (wrapped as ChatOrchestratorError)
        """
        try:
            # Validate request
            self._ensure_chat_request(request)

            # Retrieve context documents
            context, documents = await self._retrieve(request.message.content)

            # Process message and stream response
            config = {"configurable": {"session_id": request.session_id}}
            full_response = ""

            async for chunk in self._runnable_with_history.astream(
                {"input": request.message.content},
                config=config,
            ):
                if isinstance(chunk, BaseMessage) and chunk.content and chunk.content.strip():
                    full_response += chunk.content

                    # Stream response chunk (without sources - will add at end)
                    yield RagChatResponse(
                        model=self._llm_service.get_model_spec().name,
                        message=request.message.model_copy(
                            update={
                                "role": MessageRole.ASSISTANT.value,
                                "content": chunk.content,
                            }
                        ),
                        done=False,
                        created_at=_get_current_timestamp(),
                    )

            # Send final response with sources
            sources = self._create_source_citations(documents)
            yield RagChatResponse(
                model=self._llm_service.get_model_spec().name,
                done=True,
                sources=sources,
                rag_metadata={
                    "documents_retrieved": len(documents),
                    "context_tokens": self._context_manager.estimate_document_tokens(
                        Document(page_content=context)
                    ),
                },
                created_at=_get_current_timestamp(),
            )

        except (MessageValidationError, ChatOrchestratorError) as e:
            # Re-raise domain exceptions for the API layer to handle
            raise e
        except Exception as e:
            logger.error(
                "Error processing RAG chat message for session %s: %s",
                request.session_id,
                str(e),
            )
            raise ChatOrchestratorError(
                f"Failed to process RAG chat message: {str(e)}",
                session_id=request.session_id,
                original_error=e,
            )

    def _ensure_chat_request(self, request: ChatRequest) -> None:
        """Validate a chat request.

        Args:
            request: Chat request to validate

        Raises:
            MessageValidationError: If request is invalid
        """
        if not request.message:
            raise MessageValidationError("Message is required")

        if not request.message.content or not request.message.content.strip():
            raise MessageValidationError(
                "Message content cannot be empty", field="content"
            )

        if not request.session_id or not request.session_id.strip():
            raise MessageValidationError("Session ID is required", field="session_id")

    async def clear_session(self, session_id: str) -> bool:
        """Clear a specific chat session.

        Args:
            session_id: The session ID to clear

        Returns:
            True if session was found and cleared, False otherwise
        """
        return self._session_service.clear_session(session_id)

    async def get_active_session_count(self) -> int:
        """Get the current number of active sessions.

        Returns:
            Number of active chat sessions
        """
        return self._session_service.get_active_session_count()

    async def clear_all_sessions(self) -> int:
        """Clear all chat sessions.

        Returns:
            Number of sessions that were cleared
        """
        return self._session_service.cleanup_expired_sessions()

    def get_turns_window(self) -> int:
        """Get the configured turns window for conversation history.

        Returns:
            Number of conversation turns kept in memory
        """
        return self._turns_window

    async def health_check(self) -> bool:
        """Check if the RAG chat use case is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check LLM provider health
            llm_service_healthy = await self._llm_service.health_check()
            if not llm_service_healthy:
                logger.warning("RagChatUseCase health check: LLM service unhealthy")
                return False

            # Check retriever availability (try a simple query)
            try:
                result = await self._retriever.ainvoke("test")
                if result is None:
                    logger.warning(
                        "RagChatUseCase health check: Retriever returned None"
                    )
                    return False
            except Exception as e:
                logger.warning(
                    "RagChatUseCase health check: Retriever error: %s", str(e)
                )
                return False

            session_count = self._session_service.get_active_session_count()
            logger.debug(
                "RagChatUseCase health: LLM OK, retriever OK, %d active sessions",
                session_count,
            )

            return True

        except Exception as e:
            logger.error("RagChatUseCase health check failed: %s", str(e))
            return False

    