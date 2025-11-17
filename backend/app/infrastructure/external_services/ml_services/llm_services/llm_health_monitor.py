"""LLM health monitoring and performance metrics.

Tracks Ollama service health, response times, success rates,
and provides monitoring data for operational dashboards.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from app.infrastructure.external_services.ml_services.llm_services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """LLM health metrics.

    Attributes:
        response_times: Rolling window of response times (ms)
        error_count: Total error count
        success_count: Total success count
        last_check: Last health check timestamp
        last_error: Last error message
    """

    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_check: Optional[datetime] = None
    last_error: Optional[str] = None

    @property
    def avg_response_time(self) -> float:
        """Average response time in ms.

        Returns:
            Average response time, or 0.0 if no data
        """
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    @property
    def p95_response_time(self) -> float:
        """95th percentile response time in ms.

        Returns:
            95th percentile response time, or 0.0 if no data
        """
        if not self.response_times:
            return 0.0

        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[min(index, len(sorted_times) - 1)]

    @property
    def success_rate(self) -> float:
        """Success rate percentage.

        Returns:
            Success rate (0-100), or 0.0 if no checks
        """
        total = self.success_count + self.error_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100

    @property
    def total_checks(self) -> int:
        """Total health checks performed.

        Returns:
            Total check count
        """
        return self.success_count + self.error_count


class LlmHealthMonitor:
    """Monitor LLM service health and performance.

    Tracks health status, response times, and error rates for Ollama service.
    Provides metrics for monitoring dashboards and alerting.

    Args:
        ollama_service: Ollama service to monitor
        max_samples: Maximum response time samples to keep (default: 100)
    """

    def __init__(self, ollama_service: OllamaService, max_samples: int = 100):
        """Initialize health monitor.

        Args:
            ollama_service: Ollama service instance
            max_samples: Maximum response time samples to retain
        """
        self.ollama_service = ollama_service
        self.metrics = HealthMetrics()
        self.metrics.response_times = deque(maxlen=max_samples)

        logger.info(
            "LlmHealthMonitor initialized: model=%s, max_samples=%d",
            ollama_service._model_name,
            max_samples,
        )

    async def check_health(self) -> Dict[str, Any]:
        """Perform health check and update metrics.

        Returns:
            Health status dictionary with metrics

        Example:
            {
                "status": "healthy",
                "model": "llama3.2:3b",
                "response_time_ms": 123.45,
                "avg_response_time_ms": 150.20,
                "p95_response_time_ms": 250.00,
                "success_rate": 98.5,
                "error": None,
                "last_check": "2025-01-15T10:30:00"
            }
        """
        start_time = time.time()
        error_message = None

        try:
            # Use OllamaService's health_check method
            is_healthy = await self.ollama_service.health_check()
            response_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            is_healthy = False
            response_time_ms = (time.time() - start_time) * 1000
            error_message = str(e)
            logger.error("Health check exception: %s", error_message)

        # Update metrics
        self.metrics.response_times.append(response_time_ms)
        self.metrics.last_check = datetime.now()

        if is_healthy:
            self.metrics.success_count += 1
            self.metrics.last_error = None
            logger.debug("Health check passed: %.2fms", response_time_ms)
        else:
            self.metrics.error_count += 1
            self.metrics.last_error = error_message
            logger.warning("Health check failed: %s", error_message or "Unknown error")

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "model": self.ollama_service._model_name,
            "response_time_ms": response_time_ms,
            "avg_response_time_ms": self.metrics.avg_response_time,
            "p95_response_time_ms": self.metrics.p95_response_time,
            "success_rate": self.metrics.success_rate,
            "error": error_message,
            "last_check": self.metrics.last_check.isoformat()
            if self.metrics.last_check
            else None,
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current health metrics without performing check.

        Returns:
            Current metrics dictionary

        Example:
            {
                "avg_response_time_ms": 150.20,
                "p95_response_time_ms": 250.00,
                "success_rate": 98.5,
                "total_checks": 1000,
                "error_count": 15,
                "success_count": 985,
                "last_check": "2025-01-15T10:30:00",
                "last_error": None
            }
        """
        return {
            "avg_response_time_ms": self.metrics.avg_response_time,
            "p95_response_time_ms": self.metrics.p95_response_time,
            "success_rate": self.metrics.success_rate,
            "total_checks": self.metrics.total_checks,
            "error_count": self.metrics.error_count,
            "success_count": self.metrics.success_count,
            "last_check": self.metrics.last_check.isoformat()
            if self.metrics.last_check
            else None,
            "last_error": self.metrics.last_error,
        }

    def is_healthy(self, min_success_rate: float = 95.0, max_response_time_ms: float = 5000.0) -> bool:
        """Check if service meets health thresholds.

        Args:
            min_success_rate: Minimum success rate percentage (default: 95%)
            max_response_time_ms: Maximum acceptable response time (default: 5000ms)

        Returns:
            True if service is healthy, False otherwise
        """
        # Need at least some checks to determine health
        if self.metrics.total_checks == 0:
            return False

        # Check success rate threshold
        if self.metrics.success_rate < min_success_rate:
            logger.warning(
                "Success rate below threshold: %.2f%% < %.2f%%",
                self.metrics.success_rate,
                min_success_rate,
            )
            return False

        # Check response time threshold (use p95)
        if self.metrics.p95_response_time > max_response_time_ms:
            logger.warning(
                "P95 response time above threshold: %.2fms > %.2fms",
                self.metrics.p95_response_time,
                max_response_time_ms,
            )
            return False

        return True

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state.

        Useful for testing or after resolving persistent issues.
        """
        self.metrics = HealthMetrics()
        self.metrics.response_times = deque(maxlen=100)
        logger.info("Health metrics reset")

    def get_model_name(self) -> str:
        """Get the model name being monitored.

        Returns:
            Model name
        """
        return self.ollama_service._model_name
