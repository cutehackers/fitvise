"""
LangFuse Observability

LangFuse tracing integration for BotAdvisor workflows.
"""

import os
from typing import Any, Dict, Optional

# Dynamic LangFuse import with fallback
LANGFUSE_AVAILABLE = False
Langfuse = None
CallbackHandler = None

try:
    from langfuse import Langfuse
    try:
        from langfuse.callback import CallbackHandler
    except ImportError:
        # Fallback: LangFuse might have changed its API
        CallbackHandler = None
    LANGFUSE_AVAILABLE = True
except ImportError:
    # LangFuse not available
    pass

class LangFuseTracer:
    """
    LangFuse tracing integration for BotAdvisor.

    Provides centralized access to LangFuse client and callback handlers
    for tracing across scripts, retrieval, agent, and API flows.
    """

    def __init__(self):
        """
        Initialize LangFuse tracer.

        Configuration is loaded from environment variables:
        - LANGFUSE_SECRET_KEY
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_HOST
        - LANGFUSE_ENABLED
        """
        self._client = None
        self._enabled = self._is_enabled()

    def _is_enabled(self) -> bool:
        """Check if LangFuse tracing is enabled."""
        return (os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true" and
                LANGFUSE_AVAILABLE and
                Langfuse is not None)

    def get_client(self) -> Optional[Any]:
        """
        Get LangFuse client instance.

        Returns:
            LangFuse client if enabled and configured, None otherwise
        """
        if not self._enabled:
            return None

        if self._client is None:
            try:
                self._client = Langfuse(
                    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
                    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
                    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
            except Exception:
                # Fail silently if LangFuse client cannot be initialized
                self._enabled = False
                return None

        return self._client

    def get_callback_handler(self, **kwargs) -> Optional[Any]:
        """
        Get LangFuse callback handler for LangChain integration.

        Args:
            **kwargs: Additional arguments to pass to CallbackHandler

        Returns:
            CallbackHandler instance if enabled, None otherwise
        """
        if not self._enabled or CallbackHandler is None:
            return None

        client = self.get_client()
        if client is None:
            return None

        try:
            return CallbackHandler(
                langfuse_client=client,
                **kwargs
            )
        except Exception:
            return None

    def trace(
        self,
        name: str,
        trace_type: str = "task",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Create a LangFuse trace.

        Args:
            name: Trace name
            trace_type: Type of trace (task, script, api, etc.)
            metadata: Additional metadata
            **kwargs: Additional trace arguments

        Returns:
            Trace object if enabled, None otherwise
        """
        if not self._enabled:
            return None

        client = self.get_client()
        if client is None:
            return None

        try:
            return client.trace(
                name=name,
                type=trace_type,
                metadata=metadata or {},
                **kwargs
            )
        except Exception:
            return None

    def span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Create a LangFuse span.

        Args:
            name: Span name
            trace_id: Parent trace ID
            metadata: Additional metadata
            **kwargs: Additional span arguments

        Returns:
            Span object if enabled, None otherwise
        """
        if not self._enabled:
            return None

        client = self.get_client()
        if client is None:
            return None

        try:
            return client.span(
                name=name,
                trace_id=trace_id,
                metadata=metadata or {},
                **kwargs
            )
        except Exception:
            return None

    def generation(
        self,
        name: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Create a LangFuse generation.

        Args:
            name: Generation name
            trace_id: Parent trace ID
            metadata: Additional metadata
            **kwargs: Additional generation arguments

        Returns:
            Generation object if enabled, None otherwise
        """
        if not self._enabled:
            return None

        client = self.get_client()
        if client is None:
            return None

        try:
            return client.generation(
                name=name,
                trace_id=trace_id,
                metadata=metadata or {},
                **kwargs
            )
        except Exception:
            return None

    def is_enabled(self) -> bool:
        """
        Check if LangFuse tracing is enabled.

        Returns:
            True if LangFuse is enabled and configured, False otherwise
        """
        return self._enabled

# Global tracer instance for convenience
tracer = LangFuseTracer()

def get_tracer() -> LangFuseTracer:
    """
    Get the global LangFuse tracer instance.

    Returns:
        Global LangFuseTracer instance
    """
    return tracer
