"""
Workout API schemas for request/response models.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ApiErrorResponse(BaseModel):
    """Standard error response model"""

    code: Optional[str] = Field(None, description="A specific error code, if available")
    type: str = Field(
        ...,
        description="The category or type of error (e.g., 'invalid_request_error', 'rate_limit_error')",
    )
    param: Optional[str] = Field(None, description="The parameter that caused the error, if applicable")
    message: str = Field(..., description="A human-readable description of the error")


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Service status", example="healthy")
    service: str = Field(..., description="Service name", example="workout-api")
    version: str = Field(..., description="API version", example="1.0.0")
    llm_service_available: bool = Field(..., description="LLM service availability")
    timestamp: str = Field(..., description="Health check timestamp")


class PromptRequest(BaseModel):
    """Request model for fitness AI prompts"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's fitness-related prompt or question",
        example="Create a 30-minute upper body workout for beginners",
    )
    context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional context about user's fitness level, preferences, or constraints",
        example="I'm a beginner with limited equipment, only have dumbbells",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Response creativity level (0.0=deterministic, 2.0=very creative)",
    )
    max_tokens: Optional[int] = Field(None, ge=50, le=2000, description="Maximum length of the AI response")


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


# Ollama chat request
# Ref: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
#
#  Parameters
#   model: (required) the model name
#   messages: the messages of the chat, this can be used to keep a chat memory
#   tools: list of tools in JSON for the model to use if supported
#   think: (for thinking models) should the model think before responding?
#
#  The message object has the following fields:
#   role: the role of the message, either system, user, assistant, or tool
#   content: the content of the message
#   thinking: (for thinking models) the model's thinking process
#   images (optional): a list of images to include in the message (for multimodal models such as llava)
#   tool_calls (optional): a list of tools in JSON that the model wants to use
#   tool_name (optional): add the name of the tool that was executed to inform the model of the result
#
#  Advanced parameters (optional):
#   format: the format to return a response in. Format can be json or a JSON schema.
#   options: additional model parameters listed in the documentation for the Modelfile such as temperature
#   stream: if false the response will be returned as a single response object, rather than a stream of objects
#   keep_alive: controls how long the model will stay loaded into memory following the request (default: 5m)
class ChatMessage(BaseModel):
    """Single message in a chat conversation"""

    role: str = Field(..., description="Message role: 'system', 'user', 'assistant', or 'tool'")
    content: str = Field(..., description="Message content")
    thinking: Optional[bool] = Field(False, description="(for thinking models) the model's thinking process")
    images: Optional[List[str]] = Field(
        None,
        description="A list of images to include in the message (for multimodal models)",
    )
    tool_calls: Optional[List[dict]] = Field(None, description="A list of tools in JSON that the model wants to use")
    tool_name: Optional[str] = Field(
        None,
        description="The name of the tool that was executed to inform the model of the result",
    )


class ChatRequest(BaseModel):
    """Request model for chat completion"""

    model: str = Field(None, description="The model name")
    message: ChatMessage = Field(..., description="A chat message")
    session_id: str = Field(..., description="Unique session identifier for chat history")
    tools: Optional[List[dict]] = Field(None, description="List of tools in JSON for the model to use if supported")
    think: Optional[bool] = Field(
        False,
        description="(for thinking models) should the model think before responding?",
    )
    format: Optional[str] = Field(
        None,
        description="The format to return a response in. Can be json or a JSON schema.",
    )
    options: Optional[dict] = Field(None, description="Additional model parameters")
    stream: bool = Field(True, description="Whether to stream the response")
    keep_alive: Optional[str] = Field(None, description="Controls how long the model will stay loaded into memory")


class ChatResponse(BaseModel):
    """
    Response model for chat completion, aligning with Ollama's API documentation.
    A stream of responses is received, with the final response containing usage statistics.
    """

    model: str = Field(..., description="The model name used for the response.")
    created_at: str = Field(..., description="Timestamp of the response creation.")
    message: Optional[ChatMessage] = Field(
        None,
        description="The message object from the assistant. Can be empty in the final chunk.",
    )
    done: bool = Field(..., description="Boolean indicating if this is the final response chunk.")
    done_reason: Optional[str] = Field(None, description="Reason why generation stopped (e.g., 'stop')")

    # --- Final response fields (when done=True) ---
    total_duration: Optional[int] = Field(None, description="Total time spent generating the response (ns).")
    load_duration: Optional[int] = Field(None, description="Time spent loading the model (ns).")
    prompt_eval_count: Optional[int] = Field(None, description="Number of tokens in the prompt.")
    prompt_eval_duration: Optional[int] = Field(None, description="Time spent evaluating the prompt (ns).")
    eval_count: Optional[int] = Field(None, description="Number of tokens in the response.")
    eval_duration: Optional[int] = Field(None, description="Time spent generating the response tokens (ns).")

    # --- Custom internal fields ---
    success: bool = Field(
        True,
        description="Internal flag indicating if the request was processed successfully.",
    )
    error: Optional[str] = Field(None, description="Internal error message if processing failed.")
