"""
Structured Logging

Structured logging utilities with request-id binding for BotAdvisor.
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

# Context variable for request ID
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

class BotLogger:
    """
    BotAdvisor logger with request-id binding and JSON/text formatting.

    Provides consistent logging across API and script workflows with
    optional JSON formatting and request context propagation.
    """

    def __init__(self, name: str = "botadvisor"):
        """
        Initialize structured logger.

        Args:
            name: Logger name
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self._configure_logger()

    def _configure_logger(self):
        """Configure logger with appropriate handlers and formatters."""
        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            return

        # Set log level from environment
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))

        # Create formatter based on environment setting
        log_format = os.environ.get("LOG_FORMAT", "text").lower()

        if log_format == "json":
            formatter = self._create_json_formatter()
        else:
            formatter = self._create_text_formatter()

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Add file handler if log file is configured
        log_file = os.environ.get("LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data: Dict[str, Any] = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }

                # Add request ID if available
                request_id = _request_id_var.get()
                if request_id:
                    log_data["request_id"] = request_id

                # Add exception info if present
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)

                # Add extra fields
                if hasattr(record, "extra"):
                    log_data.update(record.extra)

                return json.dumps(log_data, ensure_ascii=False)

        return JSONFormatter()

    def _create_text_formatter(self) -> logging.Formatter:
        """Create text formatter for human-readable logging."""
        return logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: "
            "%(message)s" + (" [request_id=%(request_id)s]" if _request_id_var.get() else "")
        )

    def bind_request_id(self, request_id: str):
        """
        Bind request ID to current context for logging.

        Args:
            request_id: Request ID to bind
        """
        _request_id_var.set(request_id)

    def unbind_request_id(self):
        """Unbind request ID from current context."""
        _request_id_var.set(None)

    def get_request_id(self) -> Optional[str]:
        """
        Get current request ID from context.

        Returns:
            Current request ID if bound, None otherwise
        """
        return _request_id_var.get()

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log debug message.

        Args:
            message: Message to log
            extra: Additional fields for structured logging
        """
        self._log(logging.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log info message.

        Args:
            message: Message to log
            extra: Additional fields for structured logging
        """
        self._log(logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log warning message.

        Args:
            message: Message to log
            extra: Additional fields for structured logging
        """
        self._log(logging.WARNING, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log error message.

        Args:
            message: Message to log
            extra: Additional fields for structured logging
        """
        self._log(logging.ERROR, message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log critical message.

        Args:
            message: Message to log
            extra: Additional fields for structured logging
        """
        self._log(logging.CRITICAL, message, extra)

    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Internal logging method with request ID binding.

        Args:
            level: Logging level
            message: Message to log
            extra: Additional fields for structured logging
        """
        # Add request ID to extra fields if available
        request_id = self.get_request_id()
        log_extra = extra or {}

        if request_id:
            log_extra["request_id"] = request_id

        # Log with appropriate method
        if level == logging.DEBUG:
            self.logger.debug(message, extra={"extra": log_extra})
        elif level == logging.INFO:
            self.logger.info(message, extra={"extra": log_extra})
        elif level == logging.WARNING:
            self.logger.warning(message, extra={"extra": log_extra})
        elif level == logging.ERROR:
            self.logger.error(message, extra={"extra": log_extra})
        elif level == logging.CRITICAL:
            self.logger.critical(message, extra={"extra": log_extra})

    def with_request_id(self, request_id: str):
        """
        Context manager for binding request ID during logging operations.

        Args:
            request_id: Request ID to bind

        Returns:
            Context manager
        """
        from contextlib import contextmanager

        @contextmanager
        def _bind_request_id():
            original_id = self.get_request_id()
            try:
                self.bind_request_id(request_id)
                yield
            finally:
                if original_id:
                    self.bind_request_id(original_id)
                else:
                    self.unbind_request_id()

        return _bind_request_id()

# Global logger instance for convenience
logger = BotLogger()

def get_logger(name: Optional[str] = None) -> BotLogger:
    """
    Get structured logger instance.

    Args:
        name: Optional logger name

    Returns:
        StructuredLogger instance
    """
    return BotLogger(name or "botadvisor")

def configure_logger():
    """
    Configure global logging settings.

    This should be called early in application startup.
    """
    # Set up basic logging configuration
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout
    )

    # Suppress overly verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
