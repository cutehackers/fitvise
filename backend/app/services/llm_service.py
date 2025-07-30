import json
import logging
from typing import Dict, Any, Optional
import httpx
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Request model for user queries"""
    query: str = Field(..., min_length=1, max_length=10000, description="User query text")
    context: Optional[str] = Field(None, description="Additional context for the query")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Response randomness")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum response tokens")

class QueryResponse(BaseModel):
    """Response model for LLM queries"""
    response: str = Field(..., description="Generated response text")
    model: str = Field(..., description="Model used for generation")
    tokens_used: Optional[int] = Field(None, description="Total number of tokens consumed")
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in prompt")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in completion")
    total_duration_ms: Optional[float] = Field(None, description="Total request duration in milliseconds")
    success: bool = Field(True, description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if any")
    done: Optional[bool] = Field(None, description="Whether the generation is complete")

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
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user query and return LLM response
        
        Args:
            request: QueryRequest containing user query and optional parameters
            
        Returns:
            QueryResponse with generated text and metadata
        """
        try:
            # Prepare the payload for the LLM API
            payload = self._prepare_request_payload(request)
            
            # Make the API request
            response = await self._make_request(payload)
            
            # Parse and return the response
            return self._parse_response(response)
            
        except httpx.TimeoutException:
            logger.error(f"LLM request timeout after {self.timeout}s")
            return QueryResponse(
                response="",
                model=self.model_name,
                success=False,
                error="Request timeout - the AI service took too long to respond"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
            return QueryResponse(
                response="",
                model=self.model_name,
                success=False,
                error=f"AI service error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in LLM query: {str(e)}")
            return QueryResponse(
                response="",
                model=self.model_name,
                success=False,
                error="An unexpected error occurred while processing your request"
            )
    
    def _prepare_request_payload(self, request: QueryRequest) -> Dict[str, Any]:
        """
        Prepare the API payload for LLM API generation request.
        
        This method constructs the JSON payload that will be sent to the LLM API's
        /api/generate endpoint. It transforms the user's QueryRequest into the specific
        format expected by the API, combining user preferences with service defaults.
        
        The payload follows the standard LLM API format with:
        - model: Specifies which LLM model to use for generation
        - prompt: The formatted user query with optional context
        - system: System prompt defining AI behavior and role
        - stream: Boolean flag for response streaming (disabled for simpler handling)
        - options: Generation parameters including temperature, top_p, and token limits
        
        User-provided parameters (temperature, max_tokens) override service defaults,
        while missing parameters fall back to configured default values.
        
        Args:
            request (QueryRequest): User's query request containing query text,
                optional context, temperature, and max_tokens preferences
        
        Returns:
            Dict[str, Any]: Formatted API payload ready for HTTP POST request
            
        Note:
            The system prompt is automatically included to ensure consistent
            AI behavior aligned with the Fitvise application context.
        """
        payload = {
            "model": self.model_name,
            "prompt": self._format_prompt(request.query, request.context),
            "system": self._get_system_prompt(),
            "stream": False,
            "options": {
                "temperature": request.temperature or self.default_temperature,
                "top_p": 0.9,
                "num_predict": request.max_tokens or self.default_max_tokens
            }
        }
        
        return payload
    
    def _format_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Format the user query with optional context"""
        if context:
            return f"Context: {context}\n\nQuery: {query}"
        return query
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for fitness-related queries"""
        return (
            "You are a helpful AI assistant for a fitness application called Fitvise. "
            "Provide accurate, helpful, and encouraging responses related to fitness, "
            "health, workouts, nutrition, and wellness. Keep responses concise but informative."
        )
    
    async def _make_request(self, payload: Dict[str, Any]) -> httpx.Response:
        """Make the HTTP request to the LLM API"""
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json=payload
        )
        response.raise_for_status()
        return response
    
    def _parse_response(self, response: httpx.Response) -> QueryResponse:
        """Parse the LLM API response"""
        try:
            data = response.json()
            
            # Extract the generated text from API response
            generation = data.get("response", "")
            
            # Extract token usage information
            prompt_tokens = data.get("prompt_eval_count")
            completion_tokens = data.get("eval_count")
            tokens_used = None
            if prompt_tokens is not None and completion_tokens is not None:
                tokens_used = prompt_tokens + completion_tokens
            
            # Convert duration from nanoseconds to milliseconds
            total_duration_ms = None
            if "total_duration" in data:
                total_duration_ms = data["total_duration"] / 1_000_000  # ns to ms
            
            return QueryResponse(
                response=generation,
                model=data.get("model", self.model_name),
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_duration_ms=total_duration_ms,
                done=data.get("done"),
                success=True
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return QueryResponse(
                response="",
                model=self.model_name,
                success=False,
                error="Failed to parse AI service response"
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