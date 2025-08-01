"""
Chat request payload builder for converting ChatRequest to API-specific formats.

This module provides a builder pattern implementation to separate payload construction
logic from the LLM service, improving testability and maintainability.
"""

from typing import Dict, Any, Optional
from app.schemas.chat import ChatRequest, ChatMessage


class ChatPayloadBuilder:
    """Builder for converting ChatRequest to Ollama API payload format."""
    
    def __init__(self, default_model: str, default_temperature: float, default_max_tokens: int):
        """
        Initialize the builder with default configuration values.
        
        Args:
            default_model: Default model name to use when not specified in request
            default_temperature: Default temperature for response generation
            default_max_tokens: Default maximum tokens for response
        """
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
    
    def build(self, request: ChatRequest) -> Dict[str, Any]:
        """
        Convert ChatRequest to Ollama API payload format.
        
        Args:
            request: ChatRequest object containing message history and parameters
            
        Returns:
            Dict containing the formatted API payload for Ollama /api/chat endpoint
        """
        builder = _PayloadBuilder(self)
        return (builder
                .core_fields(request)
                .optional_fields(request)
                .build_options(request.options)
                .build())


class _PayloadBuilder:
    """Internal fluent builder for step-by-step payload construction."""
    
    def __init__(self, config: ChatPayloadBuilder):
        """
        Initialize the payload builder with configuration.
        
        Args:
            config: ChatRequestPayloadBuilder instance with default values
        """
        self.config = config
        self.payload = {}
    
    def core_fields(self, request: ChatRequest) -> '_PayloadBuilder':
        """
        Set the core required fields for the Ollama API payload.
        
        Args:
            request: ChatRequest containing the core field values
            
        Returns:
            Self for method chaining
        """
        messages = request.messages or []
        self.payload.update({
            "model": request.model or self.config.default_model,
            "messages": [self._format_message(msg) for msg in messages],
            "stream": request.stream
        })
        return self
    
    def optional_fields(self, request: ChatRequest) -> '_PayloadBuilder':
        """
        Add optional fields to the payload if they have values.
        
        Args:
            request: ChatRequest containing optional field values
            
        Returns:
            Self for method chaining
        """
        optional_fields = {
            "tools": request.tools,
            "think": request.think,
            "format": request.format,
            "keep_alive": request.keep_alive
        }
        
        for field_name, field_value in optional_fields.items():
            if field_value is not None:
                self.payload[field_name] = field_value
        
        return self
    
    def build_options(self, request_options: Optional[Dict[str, Any]]) -> '_PayloadBuilder':
        """
        Build the options dictionary with defaults for missing values.
        
        Args:
            request_options: Optional dictionary of request-specific options
            
        Returns:
            Self for method chaining
        """
        options = request_options.copy() if request_options else {}
        
        # Apply defaults for common options
        default_options = {
            "temperature": self.config.default_temperature,
            "top_p": 0.9,
            "num_predict": self.config.default_max_tokens
        }
        
        for option_name, default_value in default_options.items():
            if option_name not in options:
                options[option_name] = default_value
        
        self.payload["options"] = options
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build and return the final payload dictionary.
        
        Returns:
            Complete payload dictionary ready for API submission
        """
        return self.payload
    
    def _format_message(self, msg: ChatMessage) -> Dict[str, Any]:
        """
        Format a ChatMessage for Ollama API, including only valid fields.
        
        Valid Ollama message fields:
        - role: required - "system", "user", "assistant", or "tool"
        - content: required - message content
        - images: optional - list of base64 encoded images (for multimodal models)
        - tool_calls: optional - list of tool calls (for assistant messages)
        - tool_name: optional - name of tool that was executed (for tool messages)
        
        Args:
            msg: ChatMessage object to format
            
        Returns:
            Dict containing the formatted message for Ollama API
        """
        message = {
            "role": msg.role,
            "content": msg.content
        }
        
        # Add optional fields only if they have values
        optional_fields = ["thinking", "images", "tool_calls", "tool_name"]
        for field in optional_fields:
            value = getattr(msg, field, None)
            if value:
                message[field] = value
        
        return message