"""
Workout API schemas for request/response models.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    """Request model for fitness AI prompts"""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="User's fitness-related prompt or question",
        example="Create a 30-minute upper body workout for beginners"
    )
    context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional context about user's fitness level, preferences, or constraints",
        example="I'm a beginner with limited equipment, only have dumbbells"
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Response creativity level (0.0=deterministic, 2.0=very creative)"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=50,
        le=2000,
        description="Maximum length of the AI response"
    )


class PromptResponse(BaseModel):
    """Response model for fitness AI prompts"""
    response: str = Field(..., description="AI-generated fitness advice or plan")
    model: str = Field(..., description="AI model used for response generation")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    prompt_tokens: Optional[int] = Field(None, description="Tokens used for input")
    completion_tokens: Optional[int] = Field(None, description="Tokens used for output")
    duration_ms: Optional[float] = Field(None, description="Response generation time in milliseconds")
    success: bool = Field(True, description="Whether prompt processing was successful")
    error: Optional[str] = Field(None, description="Error message if prompt processing failed")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status", example="healthy")
    service: str = Field(..., description="Service name", example="workout-api")
    version: str = Field(..., description="API version", example="1.0.0")
    llm_service_available: bool = Field(..., description="LLM service availability")
    timestamp: str = Field(..., description="Health check timestamp")


class ApiErrorResponse(BaseModel):
    """Standard error response model"""
    code: Optional[str] = Field(None, description="A specific error code, if available")
    type: str = Field(..., description="The category or type of error (e.g., 'invalid_request_error', 'rate_limit_error')")
    param: Optional[str] = Field(None, description="The parameter that caused the error, if applicable")
    message: str = Field(..., description="A human-readable description of the error")