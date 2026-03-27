"""Domain exceptions for LLM operations."""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(self, message: str, provider: str = None):
        super().__init__(message)
        self.message = message
        self.provider = provider


class LLMProviderError(LLMError):
    """Exception raised when LLM provider operations fail."""

    def __init__(self, message: str, provider: str = None, original_error: Exception = None):
        super().__init__(message, provider)
        self.original_error = original_error


class LLMServiceError(LLMError):
    """Exception raised when LLM service operations fail."""

    def __init__(self, message: str, provider: str = None, original_error: Exception = None):
        super().__init__(message, provider)
        self.original_error = original_error


class ChatOrchestratorError(LLMError):
    """Exception raised when chat orchestration operations fail."""

    def __init__(self, message: str, session_id: str = None, original_error: Exception = None):
        super().__init__(message)
        self.session_id = session_id
        self.original_error = original_error


class SessionNotFoundError(ChatOrchestratorError):
    """Exception raised when a chat session is not found."""

    def __init__(self, session_id: str):
        super().__init__(f"Session not found: {session_id}", session_id)


class MessageValidationError(LLMError):
    """Exception raised when message validation fails."""

    def __init__(self, message: str, field: str = None):
        super().__init__(message)
        self.field = field