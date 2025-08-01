import json
import logging
from typing import AsyncGenerator
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from backend.app.schemas.chat_payload_builder import ChatPayloadBuilder
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

class LlmService:
    """Service for handling LLM queries and responses"""
    
    def __init__(self):
        self.base_url = settings.llm_base_url.rstrip('/')
        self.model_name = settings.llm_model
        self.timeout = settings.llm_timeout
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens
        
        # Initialize HTTP client with proper timeout and error handling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # Initialize payload builder with default configuration
        self.chat_payload_builder = ChatPayloadBuilder(
            default_model=self.model_name,
            default_temperature=self.default_temperature,
            default_max_tokens=self.default_max_tokens
        )

    async def chat(self, request: ChatRequest) -> AsyncGenerator[ChatResponse, None]:
        """
        Process a chat completion request and stream the LLM response.
        
        This method sends a chat request to the Ollama API and yields
        a series of ChatResponse objects as they are received from the
        streaming API.
        
        Args:
            request: ChatRequest containing message history and optional parameters
            
        Yields:
            ChatResponse with generated message chunks and metadata
            
        Raises:
            ApiErrorResponse: If an API error or timeout occurs
        """
        try:
            payload = self.chat_payload_builder.build(request)
            logger.info(f"Sending payload to Ollama: {payload}")

            # The key change is here: Use a 'with' block for streaming
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            ) as response:
                logger.info(f"Ollama response status: {response.status_code}")
                response.raise_for_status()

                # Iterate over the response line by line as it arrives
                # This is the correct way to handle a streaming JSONL response
                async for line in response.aiter_lines():
                    if line.strip():
                        yield self._parse_chat_stream_chunk(line.encode())

        except httpx.TimeoutException:
            logger.error(f"LLM chat request timeout after {self.timeout}s")
            raise Exception("Request timeout - the AI service took too long to respond")
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM chat API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"AI service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error in LLM chat stream: {str(e)}")
            raise Exception("An unexpected error occurred while processing your chat request")

                
    def _parse_chat_stream_chunk(self, chunk: bytes) -> ChatResponse:
        """Parse a single chunk from the chat stream."""
        try:
            data = json.loads(chunk)
            
            # Extract message if present
            message_data = data.get("message")
            message = ChatMessage(**message_data) if message_data else None

            return ChatResponse(
                model=data.get("model", self.model_name),
                created_at=data.get("created_at", ""),
                message=message,
                done=data.get("done", False),
                done_reason=data.get("done_reason"),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                prompt_eval_duration=data.get("prompt_eval_duration"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration"),
                success=True
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse LLM chat stream chunk: {str(e)}")
            return ChatResponse(
                model=self.model_name,
                created_at="",
                done=True,
                success=False,
                error=f"Failed to parse AI service response chunk: {str(e)}"
            )
            
    
    async def health(self) -> bool:
        """Check if the LLM service is available"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"LLM health check failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the HTTP client connection"""
        await self.client.aclose()

# Global service instance
llm_service = LlmService()