import json
import logging
from typing import AsyncGenerator, Dict

from app.core.config import settings
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from app.domain.entities.message_role import MessageRole
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama
from langchain.memory import ConversationBufferWindowMemory

logger = logging.getLogger(__name__)


class LlmService:
    """Service for handling LLM queries and responses"""

    def __init__(self):
        self.llm = ChatOllama(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a helpful assistant that provides concise answers regarding fitness and nutrition."
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="{input}"),
            ]
        )
        self.chain = self.prompt | self.llm
        self.session_store: Dict[str, ConversationBufferWindowMemory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve the chat message history for a given session.

        If the session does not exist in the session store, a new ChatMessageHistory
        instance is created and associated with the session ID.

        Args:
            session_id (str): The unique identifier for the chat session.

        Returns:
            ConversationBufferWindowMemory: The chat message history associated with the session.
        """
        if session_id not in self.session_store:
            # Initialize ConversationBufferWindowMemory for this session.
            # k=10 means it will keep the last 10 exchanges (user + AI messages).
            # return_messages=True ensures the history is returned as a list of BaseMessage objects,
            # which is compatible with ChatPromptTemplate and ChatOllama.
            self.session_store[session_id] = ConversationBufferWindowMemory(k=10, return_messages=True)

        return self.session_store[session_id].chat_memory

    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponse, None]:
        """
        Process a chat completion request and stream the LLM response.
        """
        chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": request.session_id}}
        
        # Validate that messages are provided
        if not request.message.content:
            raise ValueError("No message provided in the request")
        
        content = request.message.content;
        
        async for chunk in chain_with_history.astream({"input": content}, config=config):
            #yield self._parse_chat_stream_chunk(chunk)
            
            # The astream method of this chain yields BaseMessage objects or their subclasses (like AIMessageChunk) directly.
            if isinstance(chunk, BaseMessage):
                yield self._parse_chat_stream_chunk(chunk)
                
        yield ChatResponse(
            model=self.llm.model,
            created_at="",  # This can be populated if needed
            done=True,  # This needs to be determined based on the stream
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
            logger.warning(f"LLM health check failed: {str(e)}")
            return False

    async def close(self):
        """Close any open connections."""
        pass


llm_service = LlmService()
