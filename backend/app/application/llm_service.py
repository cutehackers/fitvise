import json
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
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
            payload = self._prepare_chat_request_payload(request)
            logger.info(f"Sending payload to Ollama: {payload}")
            
            # Use a regular POST request and manually parse the streaming response
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )
            logger.info(f"Ollama response status: {response.status_code}")
            response.raise_for_status()
            
            # Parse the response content line by line
            content = response.text
            for line in content.strip().split('\n'):
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

    def _prepare_chat_request_payload(self, request: ChatRequest) -> Dict[str, Any]:
        """
        Prepare the API payload for chat completion request.
        
        This method constructs the JSON payload for the Ollama chat API's
        /api/chat endpoint. It transforms the ChatRequest into the format
        expected by the chat completion API.
        
        Args:
            request (ChatRequest): Chat request with message history and parameters
        
        Returns:
            Dict[str, Any]: Formatted API payload for chat completion
        """
        payload = {
            "model": request.model or self.model_name,
            "messages": [self._format_ollama_message(msg) for msg in request.messages] if request.messages else [],
            "stream": request.stream
        }
        
        # Add optional fields if present
        if request.tools:
            payload["tools"] = request.tools
            
        if request.think is not None:
            payload["think"] = request.think
            
        if request.format:
            payload["format"] = request.format
            
        if request.keep_alive:
            payload["keep_alive"] = request.keep_alive
            
        # Handle options - merge request options with defaults
        options = request.options.copy() if request.options else {}
        
        # Set default options if not provided
        if "temperature" not in options:
            options["temperature"] = self.default_temperature
        if "top_p" not in options:
            options["top_p"] = 0.9
        if "num_predict" not in options:
            options["num_predict"] = self.default_max_tokens
            
        payload["options"] = options
        
        return payload

    def _format_ollama_message(self, msg: ChatMessage) -> Dict[str, Any]:
        """
        Format a ChatMessage for Ollama API, including only valid fields.
        
        Valid Ollama message fields:
        - role: required - "system", "user", "assistant", or "tool"  
        - content: required - message content
        - images: optional - list of base64 encoded images (for multimodal models)
        - tool_calls: optional - list of tool calls (for assistant messages)
        - tool_name: optional - name of tool that was executed (for tool messages)
        """
        message = {
            "role": msg.role,
            "content": msg.content
        }
        
        # Add optional fields only if they have values
        if (msg.thinking):
            message["thinking"] = msg.thinking
        
        if msg.images:
            message["images"] = msg.images
            
        if msg.tool_calls:
            message["tool_calls"] = msg.tool_calls
            
        if msg.tool_name:
            message["tool_name"] = msg.tool_name
            
        return message
                
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