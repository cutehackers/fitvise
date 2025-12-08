"""RAG chat use case with direct LangChain integration for clean architecture.

This module implements RagChatUseCase as a complete replacement for RAG orchestrators
with direct LangChain integration, document retrieval, and source citation generation.

Uses modern LCEL (LangChain Expression Language) for chain composition instead of
legacy langchain_classic helper functions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import Runnable, RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

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
    """Clean architecture RAG chat use case with modern LCEL integration.

    Uses LangChain Expression Language (LCEL) for transparent, composable chains:
    - Explicit chain composition with RunnableParallel and RunnablePassthrough
    - No legacy langchain_classic dependencies
    - Full streaming support with native async operations
    - Session management through SessionService
    - Document retrieval through injected BaseRetriever
    - Context management through ContextWindowManager
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

        # Store LLM instance for direct access
        self._llm: BaseChatModel = llm_service.llm_instance

        # Store the rephrase chain for standalone usage in chat()
        self._rephrase_chain = self._build_rephrase_chain()
        
        # Build the QA chain that outputs AIMessage directly (for history compatibility)
        self._qa_chain = self._build_qa_chain()
        self._chain = self._build_chain_with_history(self._qa_chain)

        logger.info(
            "RagChatUseCase initialized with modern LCEL chain, retriever, context manager, and session service"
        )

    def _build_rephrase_chain(self) -> Runnable:
        """Build the question rephrasing chain.
        
        This chain reformulates user questions based on conversation history
        for more effective document retrieval.
        
        Returns:
            Runnable chain that takes {"input": str, "history": List} and returns str
        """
        rephrase_prompt = ChatPromptTemplate.from_messages([
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
                "- Do NOT answer the question, just reformulate it if needed and otherwise return it as is\n"
            )),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}"),
        ])
        
        return rephrase_prompt | self._llm | StrOutputParser()

    def _build_qa_chain(self) -> Runnable:
        """Build the QA answer chain.
        
        This chain generates the final answer using context and conversation history.
        CRITICAL: This chain must output AIMessage directly (not a dict) for 
        RunnableWithMessageHistory to properly track conversation history.
        
        Returns:
            Runnable chain that takes {"input": str, "context": str, "history": List}
            and returns AIMessage
        """
        qa_prompt = ChatPromptTemplate.from_messages([
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
        
        # Output AIMessage directly - do NOT use StrOutputParser
        # This is critical for RunnableWithMessageHistory to save the response to history
        return qa_prompt | self._llm

    def _build_chain_with_history(self, chain: Runnable) -> RunnableWithMessageHistory:
        """Wrap a runnable with LangChain message history management."""

        def get_session_history(config: Dict[str, Any]) -> BaseChatMessageHistory:
            session_id = None
            configurable = config.get("configurable") if isinstance(config, dict) else None
            if configurable:
                session_id = configurable.get("session_id")
            # Fallback: create a session if none was provided in the config
            if not session_id:
                session_id, history = self._session_service.ensure_session()
                logger.debug("Generated session_id for missing history: %s", session_id)
                return history
            return self._session_service.get_session_history(session_id)

        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            history_factory_config_keys=["session_id"],
        )

    @staticmethod
    def _prepare_context_documents(documents: List[Document]) -> List[Document]:
        """Attach stable indices to documents for in-context citation."""
        return [
            Document(
                page_content=f"[{idx}] {doc.page_content}",
                metadata=doc.metadata,
            )
            for idx, doc in enumerate(documents, start=1)
        ]

    def _rephrase_query(
        self, 
        question: str, 
        history: List[BaseMessage]
    ) -> str:
        """Rephrase the question using conversation history.
        
        Args:
            question: The original user question
            history: Conversation history messages
            
        Returns:
            Rephrased question string, or original if no history
        """
        if not history:
            return question
        return self._rephrase_chain.invoke({
            "input": question,
            "history": history
        })


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
            documents = await self._retriever.ainvoke(query) or []

            if not documents:
                logger.warning("No documents retrieved for query: %s", query)
                return "No relevant context found.", []

            logger.debug("Retrieved %d documents for query", len(documents))

            # Attach indices for clear in-context citations
            indexed_documents = self._prepare_context_documents(documents)

            # Step 2: Fit documents into context window
            try:
                context = self._context_manager.fit_to_window(
                    indexed_documents,
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

                for doc in indexed_documents:
                    doc_tokens = self._context_manager.estimate_document_tokens(doc)
                    if current_tokens + doc_tokens <= max_tokens:
                        context_parts.append(doc.page_content)
                        current_tokens += doc_tokens
                    else:
                        break

                context = "\n\n".join(context_parts)

            if not context:
                context = "\n\n".join(doc.page_content for doc in indexed_documents)

            return context, indexed_documents

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
        citations: List[SourceCitation] = []
        for idx, doc in enumerate(documents, 1):
            metadata = doc.metadata or {}
            content = self._normalize_citation_content(doc.page_content)
            citations.append(
                SourceCitation(
                    index=idx,
                    content=content,
                    similarity_score=metadata.get("_distance", 0.0) or 0.0,
                    document_id=metadata.get("document_id", f"doc_{idx}"),
                    chunk_id=metadata.get("chunk_id", f"chunk_{idx}"),
                    metadata=metadata,
                )
            )
        return citations

    @staticmethod
    def _normalize_citation_content(content: str, max_length: int = 500) -> str:
        """Normalize and truncate citation content for safe display."""
        if not content:
            return ""
        normalized = " ".join(content.split())
        return normalized[:max_length]

    async def chat(
        self,
        request: ChatRequest,
    ) -> AsyncGenerator[RagChatResponse, None]:
        """Process a chat message with RAG and generate streaming responses.

        The flow:
        1. Get session history
        2. Rephrase question using history (for better retrieval)
        3. Retrieve relevant documents
        4. Stream answer using history-tracked QA chain

        Args:
            request: Chat request with message and session info

        Yields:
            Streaming RAG chat responses with source citations

        Raises:
            MessageValidationError: If request validation fails
            ChatOrchestratorError: If RAG processing fails
            Exception: For unexpected errors (wrapped as ChatOrchestratorError)
        """
        session_id = request.session_id
        try:
            # Create or retrieve a session for first-time conversations
            session_id, session_history = self._session_service.ensure_session(session_id)
            if request.session_id != session_id:
                request = request.model_copy(update={"session_id": session_id})

            # Validate request
            self._ensure_chat_request(request)

            # Step 1: Get session history for rephrasing (previous turns only)
            history_messages = list(session_history.messages)
            
            # Step 2: Rephrase question if we have history (for better retrieval)
            rephrased_query = self._rephrase_query(
                request.message.content, 
                history_messages
            )
            logger.debug(
                "Rephrased question: '%s' -> '%s'", 
                request.message.content, 
                rephrased_query
            )

            # Step 3: Retrieve documents using rephrased question (with robust fallbacks)
            context, documents = await self._retrieve(rephrased_query)
            logger.debug(
                "Retrieved %d documents for RAG (context length=%d)",
                len(documents),
                len(context),
            )

            # Step 4: Stream answer using explicit history
            # The chain receives: {"input": str, "context": str}; history is injected by RunnableWithMessageHistory.
            chain_inputs = {
                "input": request.message.content,
                "context": context,
            }
            config = {"configurable": {"session_id": session_id}}
            full_response = ""

            async for chunk in self._chain.astream(chain_inputs, config=config,):
                # Handle AIMessage/AIMessageChunk from the LLM
                chunk_content: Optional[str] = None

                if isinstance(chunk, AIMessageChunk):
                    chunk_content = chunk.content
                elif isinstance(chunk, BaseMessage):
                    chunk_content = chunk.content
                elif isinstance(chunk, str):
                    chunk_content = chunk

                if chunk_content and chunk_content.strip():
                    full_response += chunk_content
                    yield RagChatResponse(
                        model=self._llm_service.get_model_spec().name,
                        message=request.message.model_copy(
                            update={
                                "role": MessageRole.ASSISTANT.value,
                                "content": chunk_content,
                            }
                        ),
                        done=False,
                        session_id=session_id,
                        created_at=_get_current_timestamp(),
                    )

            # Send final response with sources
            sources = self._create_source_citations(documents)
            yield RagChatResponse(
                model=self._llm_service.get_model_spec().name,
                done=True,
                sources=sources,
                rag_metadata={
                    "documents": len(documents),
                    "context_available": bool(documents),
                    "context_tokens": self._context_manager.estimate_document_tokens(
                        Document(page_content=context)
                    ) if context else 0,
                    "rephrased_question": rephrased_query,
                },
                session_id=session_id,
                created_at=_get_current_timestamp(),
            )

        except (MessageValidationError, ChatOrchestratorError) as e:
            # Re-raise domain exceptions for the API layer to handle
            raise e
        except Exception as e:
            logger.error(
                "Error processing RAG chat message for session %s: %s",
                session_id,
                str(e),
            )
            raise ChatOrchestratorError(
                f"Failed to process RAG chat message: {str(e)}",
                session_id=session_id,
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

    
