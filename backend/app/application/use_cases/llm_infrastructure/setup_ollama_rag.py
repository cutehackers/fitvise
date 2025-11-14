"""RAG pipeline setup and execution with Ollama (Task 3.1.1).

Orchestrates retrieval-augmented generation using:
- Weaviate semantic search for retrieval
- Context window management for fitting documents
- Ollama LLM for generation with citations
"""

import logging
from typing import List, Tuple, AsyncGenerator, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.domain.llm.interfaces.llm_provider import LLMProvider
from app.infrastructure.adapters.weaviate_langchain_retriever import (
    WeaviateLangChainRetriever,
)
from app.infrastructure.external_services.context_management.context_window_manager import (
    ContextWindowManager,
)

logger = logging.getLogger(__name__)


class SetupOllamaRagUseCase:
    """Setup and execute RAG pipeline with LLM provider.

    Orchestrates the complete RAG workflow:
    1. Retrieve relevant documents via semantic search
    2. Fit documents into context window
    3. Generate response with LLM
    4. Return response with source citations

    Args:
        llm_provider: LLM provider for generation
        retriever: Weaviate retriever for semantic search
        context_manager: Context window manager
    """

    # RAG prompt template with citation instructions
    SYSTEM_PROMPT = """You are a helpful fitness assistant. Answer questions using the provided context.

IMPORTANT: When using information from the context, cite your sources using [1], [2], etc.

Context:
{context}

If the context doesn't contain relevant information, say so clearly."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        retriever: WeaviateLangChainRetriever,
        context_manager: ContextWindowManager,
    ):
        """Initialize RAG use case.

        Args:
            llm_provider: LLM provider for generation
            retriever: Retriever for semantic search
            context_manager: Context window manager
        """
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.context_manager = context_manager

        # Create prompt template with citations
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.SYSTEM_PROMPT), ("human", "{input}")]
        )

        # Create LangChain-compatible LLM wrapper
        self._langchain_llm = self._create_langchain_wrapper()

        logger.info("SetupOllamaRagUseCase initialized")

    async def execute_rag_query(
        self, query: str, top_k: int = 5, session_id: Optional[str] = None
    ) -> Tuple[str, List[Document]]:
        """Execute RAG query with retrieval + generation.

        Args:
            query: User query
            top_k: Number of chunks to retrieve (default: 5)
            session_id: Optional session ID for future session support

        Returns:
            Tuple of (generated_response, source_documents)

        Raises:
            ValueError: If query is empty
            Exception: If retrieval or generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        logger.info("Executing RAG query: query='%s', top_k=%d", query[:100], top_k)

        try:
            # Step 1: Retrieve relevant documents
            self.retriever.top_k = top_k
            documents = await self.retriever.aget_relevant_documents(query)

            if not documents:
                logger.warning("No documents retrieved for query")
                return (
                    "I couldn't find any relevant information to answer your question.",
                    [],
                )

            logger.debug("Retrieved %d documents", len(documents))

            # Step 2: Fit documents into context window
            context = self.context_manager.fit_to_window(
                documents=documents, user_query=query, system_prompt=self.SYSTEM_PROMPT
            )

            # Step 3: Create chain using LCEL (modern LangChain pattern)
            # The prompt template already handles formatting the context
            document_chain = self.prompt | self._langchain_llm

            # Step 4: Generate response
            response = await document_chain.ainvoke({"context": context, "input": query})

            logger.info("RAG query completed: response_length=%d", len(response))

            return response, documents

        except Exception as e:
            logger.error("RAG query failed: %s", str(e))
            raise

    async def execute_rag_stream(
        self, query: str, top_k: int = 5, session_id: Optional[str] = None
    ) -> Tuple[AsyncGenerator[str, None], List[Document]]:
        """Execute RAG query with streaming response.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            session_id: Optional session ID for future session support

        Returns:
            Tuple of (response_stream_generator, source_documents)

        Raises:
            ValueError: If query is empty
            Exception: If retrieval or generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        logger.info(
            "Executing RAG stream query: query='%s', top_k=%d", query[:100], top_k
        )

        try:
            # Step 1: Retrieve relevant documents
            self.retriever.top_k = top_k
            documents = await self.retriever.aget_relevant_documents(query)

            if not documents:
                logger.warning("No documents retrieved for streaming query")

                async def empty_stream():
                    yield "I couldn't find any relevant information to answer your question."

                return empty_stream(), []

            logger.debug("Retrieved %d documents for streaming", len(documents))

            # Step 2: Fit documents into context window
            context = self.context_manager.fit_to_window(
                documents=documents, user_query=query, system_prompt=self.SYSTEM_PROMPT
            )

            # Step 3: Create streaming chain using LCEL (modern LangChain pattern)
            document_chain = self.prompt | self._langchain_llm

            # Step 4: Stream response
            async def stream_generator():
                """Generate streaming response."""
                try:
                    async for chunk in document_chain.astream(
                        {"context": context, "input": query}
                    ):
                        if chunk:
                            yield chunk
                except Exception as e:
                    logger.error("Streaming failed: %s", str(e))
                    yield f"\n\nError: {str(e)}"

            logger.info("RAG streaming query setup complete")

            return stream_generator(), documents

        except Exception as e:
            logger.error("RAG streaming query failed: %s", str(e))
            raise

    def get_retriever(self) -> WeaviateLangChainRetriever:
        """Get the retriever instance.

        Returns:
            Weaviate retriever
        """
        return self.retriever

    def get_context_manager(self) -> ContextWindowManager:
        """Get the context manager instance.

        Returns:
            Context window manager
        """
        return self.context_manager

    def _create_langchain_wrapper(self):
        """Create a LangChain-compatible wrapper for our LLM provider."""
        from langchain_core.language_models.llms import BaseLLM
        from langchain_core.messages import BaseMessage
        from pydantic import Field
        from typing import Optional, Dict, Any, List

        class LangChainLLMWrapper(BaseLLM):
            """LangChain-compatible wrapper for our LLM provider."""

            llm_provider: LLMProvider = Field(...)

            def __init__(self, llm_provider: LLMProvider, **kwargs):
                super().__init__(llm_provider=llm_provider, **kwargs)

            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[Any] = None,
                **kwargs: Any,
            ) -> str:
                """Synchronous call - not supported for async providers."""
                raise NotImplementedError("Use async methods with this LLM provider")

            async def _acall(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[Any] = None,
                **kwargs: Any,
            ) -> str:
                """Async call to generate response."""
                # Convert prompt to message format
                from app.domain.llm.entities.message import Message
                from app.domain.entities.message_role import MessageRole

                message = Message(content=prompt, role=MessageRole.USER)
                return await self.llm_provider.generate([message], **kwargs)

            @property
            def _llm_type(self) -> str:
                """Return the LLM type."""
                return f"custom_{self.llm_provider.provider_name}"

        return LangChainLLMWrapper(llm_provider=self.llm_provider)
