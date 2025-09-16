import logging
from operator import itemgetter
from typing import AsyncGenerator, Dict

from langchain.globals import set_debug
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama

from app.core.settings import settings
from app.domain.entities.message_role import MessageRole
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
set_debug(True)

MAX_TOKENS_TABLE = {
    "llama3.2:3b": 128000,
}


class LlmService:
    """
    Service for handling LLM queries and responses

    Args:
        system_prompt (Optional[str]): Optional system prompt to initialize the chat model.
            If not provided, a default prompt will be used.
        turns_window (int): Number of turns to keep in memory for each session. Default is 10.
    """

    def __init__(self, turns_window: int = 10):
        # LLM model
        self.llm = ChatOllama(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

        # Chat prompt template
        self.prompt: ChatPromptTemplate = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful and versatile AI assistant. Answer user questions thoroughly and thoughtfully.",
                ),
                MessagesPlaceholder(variable_name="history", optional=True),
                ("human", "{input}"),
            ]
        )

        # Chat message history trimmesr
        self.trimmer = trim_messages(
            max_tokens=MAX_TOKENS_TABLE.get("llama3.2:3b", 128000),
            token_counter=self.llm,
            strategy="last",
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # start_on="human" makes sure we produce a valid chat history
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
        )

        # Session store
        self.session_store: Dict[str, ChatMessageHistory] = {}
        self.turns_window = turns_window

        # Create chain with memory
        # self.trimmer outputs a list (trimmed messages). ChatPromptTemplate expects a dict.
        # So we transform the list into a dict with "history" as the key.
        # The chain will use the trimmed history as input.
        # The prompt expects a "history" key, which will be filled with the trimmed messages.
        # The LLM will then generate a response based on the trimmed history.
        self.chain = RunnablePassthrough.assign(history=itemgetter("history") | self.trimmer) | self.prompt | self.llm

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve the chat message history for a given session.

        If the session does not exist in the session store, a new ChatMessageHistory
        instance is created and associated with the session ID.

        Args:
            session_id (str): The unique identifier for the chat session.

        Returns:
            BaseChatMessageHistory: The chat message history associated with the session.

        Raises:
            ValueError: If session_id is invalid or empty.
        """
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty or None")

        session_id = session_id.strip()

        if session_id not in self.session_store:
            try:
                self.session_store[session_id] = InMemoryChatMessageHistory()
                logger.info("New chat session: %s", session_id)
            except Exception as e:
                logger.error("Failed to initialize session %s: %s", session_id, str(e))
                raise ValueError(f"Failed to initialize chat session: {str(e)}") from e

        return self.session_store[session_id]

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a specific chat session from memory.

        Args:
            session_id (str): The session ID to clear

        Returns:
            bool: True if session was found and cleared, False if session didn't exist
        """
        if session_id in self.session_store:
            del self.session_store[session_id]
            logger.info("Cleared chat session: %s", session_id)
            return True
        return False

    def clear_all_sessions(self) -> int:
        """
        Clear all chat sessions from memory.

        Returns:
            int: Number of sessions that were cleared
        """
        session_count = len(self.session_store)
        self.session_store.clear()
        logger.info("Cleared %d chat sessions", session_count)
        return session_count

    def get_session_count(self) -> int:
        """
        Get the current number of active sessions.

        Returns:
            int: Number of active chat sessions
        """
        return len(self.session_store)

    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponse, None]:
        """
        Process a chat completion request and stream the LLM response.
        """
        # Validate request structure early
        if not request.message:
            raise ValueError("Message is required in the request")

        if not request.message.content or not request.message.content.strip():
            raise ValueError("Message content cannot be empty")

        if not request.session_id:
            raise ValueError("Session ID is required for chat history management")

        content = request.message.content.strip()

        logger.info("Chat: request - session_id: %s, content: %s", request.session_id, content)

        chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        try:
            # Stream the LLM response using the chain with trimmed history
            response = ""

            config = {"configurable": {"session_id": request.session_id}}
            async for chunk in chain_with_history.astream({"input": content}, config=config):
                # The astream method yields BaseMessage objects or subclasses
                if isinstance(chunk, BaseMessage):
                    response += chunk.content
                    yield self._parse_chat_stream_chunk(chunk)

            yield ChatResponse(
                model=self.llm.model,
                created_at="",
                done=True,
            )
        except Exception as e:
            logger.error(
                "Error during chat streaming for session %s: %s",
                request.session_id,
                str(e),
            )
            yield ChatResponse(
                model=self.llm.model,
                created_at="",
                done=True,
                success=False,
                error=f"Chat streaming failed: {str(e)}",
            )

    def _parse_chat_stream_chunk(self, chunk: BaseMessage) -> ChatResponse:
        """Parse a single chunk from the chat stream."""
        logger.debug("Chat: response - chunk: session_id: %s", chunk.content)
        return ChatResponse(
            model=self.llm.model,
            created_at="",  # This can be populated if needed
            message=ChatMessage(role=MessageRole.ASSISTANT.value, content=chunk.content),
            done=False,  # This needs to be determined based on the stream
        )

    async def health(self) -> bool:
        """Check if the LLM service is available"""
        try:
            await self.llm.ainvoke("Health check")
            return True
        except Exception as e:
            logger.warning("LLM health check failed: %s", str(e))
            return False

    async def close(self):
        """Close any open connections."""
        # No connections to close in current implementation


llm_service = LlmService()
