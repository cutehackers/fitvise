"""Unit tests for LlmHealthMonitor."""

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

import pytest

from app.infrastructure.external_services.ml_services.llm_services.llm_health_monitor import (
    LlmHealthMonitor,
    HealthMetrics,
)
from app.infrastructure.external_services.ml_services.llm_services.base_llm_service import (
    LlmHealthStatus,
)


class TestHealthMetrics:
    """Test HealthMetrics dataclass calculations."""

    def test_avg_response_time_empty(self):
        """Test average response time with no data."""
        metrics = HealthMetrics()
        assert metrics.avg_response_time == 0.0

    def test_avg_response_time_single_value(self):
        """Test average response time with single value."""
        metrics = HealthMetrics()
        metrics.response_times.append(100.0)
        assert metrics.avg_response_time == 100.0

    def test_avg_response_time_multiple_values(self):
        """Test average response time with multiple values."""
        metrics = HealthMetrics()
        metrics.response_times.extend([100.0, 200.0, 300.0])
        assert metrics.avg_response_time == 200.0

    def test_p95_response_time_empty(self):
        """Test P95 response time with no data."""
        metrics = HealthMetrics()
        assert metrics.p95_response_time == 0.0

    def test_p95_response_time_single_value(self):
        """Test P95 response time with single value."""
        metrics = HealthMetrics()
        metrics.response_times.append(100.0)
        assert metrics.p95_response_time == 100.0

    def test_p95_response_time_calculation(self):
        """Test P95 response time calculation."""
        metrics = HealthMetrics()
        # Add 20 values: 10, 20, 30, ..., 200
        for i in range(1, 21):
            metrics.response_times.append(i * 10.0)

        # P95 of 20 values: int(20 * 0.95) = 19, accessing sorted_times[19] gives 200.0
        p95 = metrics.p95_response_time
        assert p95 == 200.0

    def test_success_rate_no_checks(self):
        """Test success rate with no checks performed."""
        metrics = HealthMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Test success rate with all successful checks."""
        metrics = HealthMetrics()
        metrics.success_count = 10
        metrics.error_count = 0
        assert metrics.success_rate == 100.0

    def test_success_rate_all_failures(self):
        """Test success rate with all failed checks."""
        metrics = HealthMetrics()
        metrics.success_count = 0
        metrics.error_count = 10
        assert metrics.success_rate == 0.0

    def test_success_rate_mixed(self):
        """Test success rate with mixed results."""
        metrics = HealthMetrics()
        metrics.success_count = 8
        metrics.error_count = 2
        assert metrics.success_rate == 80.0

    def test_total_checks(self):
        """Test total checks calculation."""
        metrics = HealthMetrics()
        metrics.success_count = 7
        metrics.error_count = 3
        assert metrics.total_checks == 10

    def test_rolling_window_maxlen(self):
        """Test that response_times has max length of 100."""
        metrics = HealthMetrics()

        # Add 150 values
        for i in range(150):
            metrics.response_times.append(float(i))

        # Should only keep last 100
        assert len(metrics.response_times) == 100
        assert metrics.response_times[0] == 50.0  # First of last 100
        assert metrics.response_times[-1] == 149.0


class TestLlmHealthMonitor:
    """Test LlmHealthMonitor functionality."""

    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock OllamaService."""
        mock_service = MagicMock()
        mock_service._model = "llama3.2:3b"
        return mock_service

    @pytest.fixture
    def monitor(self, mock_ollama_service):
        """Create LlmHealthMonitor with mock service."""
        return LlmHealthMonitor(mock_ollama_service, max_samples=100)

    @pytest.mark.asyncio
    async def test_check_health_success(self, monitor, mock_ollama_service):
        """Test successful health check."""
        # Mock healthy status
        mock_status = LlmHealthStatus(
            is_healthy=True,
            model="llama3.2:3b",
            response_time_ms=123.45,
            error=None,
        )
        mock_ollama_service.health_check = AsyncMock(return_value=mock_status)

        # Execute
        result = await monitor.check_health()

        # Verify
        assert result["status"] == "healthy"
        assert result["model"] == "llama3.2:3b"
        assert result["response_time_ms"] == 123.45
        assert result["error"] is None

        # Verify metrics updated
        assert monitor.metrics.success_count == 1
        assert monitor.metrics.error_count == 0
        assert len(monitor.metrics.response_times) == 1
        assert monitor.metrics.last_error is None

    @pytest.mark.asyncio
    async def test_check_health_failure(self, monitor, mock_ollama_service):
        """Test failed health check."""
        # Mock unhealthy status
        mock_status = LlmHealthStatus(
            is_healthy=False,
            model="llama3.2:3b",
            response_time_ms=234.56,
            error="Connection timeout",
        )
        mock_ollama_service.health_check = AsyncMock(return_value=mock_status)

        # Execute
        result = await monitor.check_health()

        # Verify
        assert result["status"] == "unhealthy"
        assert result["model"] == "llama3.2:3b"
        assert result["response_time_ms"] == 234.56
        assert result["error"] == "Connection timeout"

        # Verify metrics updated
        assert monitor.metrics.success_count == 0
        assert monitor.metrics.error_count == 1
        assert monitor.metrics.last_error == "Connection timeout"

    @pytest.mark.asyncio
    async def test_check_health_multiple_calls(self, monitor, mock_ollama_service):
        """Test multiple health checks accumulate metrics."""
        # Mock sequence of checks
        statuses = [
            LlmHealthStatus(True, "llama3.2:3b", 100.0, None),
            LlmHealthStatus(True, "llama3.2:3b", 150.0, None),
            LlmHealthStatus(False, "llama3.2:3b", 200.0, "Error"),
            LlmHealthStatus(True, "llama3.2:3b", 120.0, None),
        ]

        for status in statuses:
            mock_ollama_service.health_check = AsyncMock(return_value=status)
            await monitor.check_health()

        # Verify accumulated metrics
        assert monitor.metrics.success_count == 3
        assert monitor.metrics.error_count == 1
        assert monitor.metrics.total_checks == 4
        assert len(monitor.metrics.response_times) == 4
        assert monitor.metrics.success_rate == 75.0

    @pytest.mark.asyncio
    async def test_get_metrics(self, monitor, mock_ollama_service):
        """Test get_metrics without performing health check."""
        # Perform some checks first
        mock_status = LlmHealthStatus(True, "llama3.2:3b", 100.0, None)
        mock_ollama_service.health_check = AsyncMock(return_value=mock_status)
        await monitor.check_health()
        await monitor.check_health()

        # Get metrics
        metrics = await monitor.get_metrics()

        # Verify
        assert metrics["total_checks"] == 2
        assert metrics["success_count"] == 2
        assert metrics["error_count"] == 0
        assert metrics["success_rate"] == 100.0
        assert metrics["avg_response_time_ms"] == 100.0
        assert "last_check" in metrics

    @pytest.mark.asyncio
    async def test_get_metrics_no_checks(self, monitor):
        """Test get_metrics with no prior health checks."""
        metrics = await monitor.get_metrics()

        assert metrics["total_checks"] == 0
        assert metrics["success_rate"] == 0.0
        assert metrics["avg_response_time_ms"] == 0.0
        assert metrics["last_check"] is None

    def test_is_healthy_no_checks(self, monitor):
        """Test is_healthy returns False with no checks."""
        assert monitor.is_healthy() is False

    @pytest.mark.asyncio
    async def test_is_healthy_above_thresholds(self, monitor, mock_ollama_service):
        """Test is_healthy returns True when above thresholds."""
        # Perform successful checks
        mock_status = LlmHealthStatus(True, "llama3.2:3b", 1000.0, None)
        mock_ollama_service.health_check = AsyncMock(return_value=mock_status)

        for _ in range(10):
            await monitor.check_health()

        # Check with default thresholds (95% success, 5000ms)
        assert monitor.is_healthy(min_success_rate=95.0, max_response_time_ms=5000.0) is True

    @pytest.mark.asyncio
    async def test_is_healthy_below_success_rate(self, monitor, mock_ollama_service):
        """Test is_healthy returns False when success rate below threshold."""
        # 7 success, 3 failures = 70% success rate
        for i in range(10):
            is_healthy = i < 7
            mock_status = LlmHealthStatus(
                is_healthy, "llama3.2:3b", 1000.0, "Error" if not is_healthy else None
            )
            mock_ollama_service.health_check = AsyncMock(return_value=mock_status)
            await monitor.check_health()

        # Check with 95% threshold
        assert monitor.is_healthy(min_success_rate=95.0, max_response_time_ms=5000.0) is False

    @pytest.mark.asyncio
    async def test_is_healthy_above_response_time(self, monitor, mock_ollama_service):
        """Test is_healthy returns False when P95 response time exceeds threshold."""
        # All successful but slow responses
        for i in range(20):
            # Most are 1000ms, but a few are 6000ms
            response_time = 6000.0 if i >= 18 else 1000.0
            mock_status = LlmHealthStatus(True, "llama3.2:3b", response_time, None)
            mock_ollama_service.health_check = AsyncMock(return_value=mock_status)
            await monitor.check_health()

        # P95 will be high due to slow responses
        # Check with 5000ms threshold
        result = monitor.is_healthy(min_success_rate=95.0, max_response_time_ms=5000.0)
        # P95 should be 6000ms, exceeding threshold
        assert result is False

    def test_reset_metrics(self, monitor):
        """Test reset_metrics clears all accumulated data."""
        # Add some data
        monitor.metrics.success_count = 10
        monitor.metrics.error_count = 5
        monitor.metrics.response_times.extend([100.0, 200.0, 300.0])
        monitor.metrics.last_error = "Some error"

        # Reset
        monitor.reset_metrics()

        # Verify cleared
        assert monitor.metrics.success_count == 0
        assert monitor.metrics.error_count == 0
        assert len(monitor.metrics.response_times) == 0
        assert monitor.metrics.last_error is None

    def test_get_model_name(self, monitor):
        """Test get_model_name returns correct model."""
        assert monitor.get_model_name() == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_check_health_updates_last_check(self, monitor, mock_ollama_service):
        """Test that check_health updates last_check timestamp."""
        mock_status = LlmHealthStatus(True, "llama3.2:3b", 100.0, None)
        mock_ollama_service.health_check = AsyncMock(return_value=mock_status)

        # Initial state
        assert monitor.metrics.last_check is None

        # Perform check
        await monitor.check_health()

        # Verify timestamp updated
        assert monitor.metrics.last_check is not None
        assert isinstance(monitor.metrics.last_check, datetime)

    @pytest.mark.asyncio
    async def test_custom_max_samples(self, mock_ollama_service):
        """Test monitor with custom max_samples."""
        # Create monitor with small window
        monitor = LlmHealthMonitor(mock_ollama_service, max_samples=5)

        # Add more samples than window size
        mock_status = LlmHealthStatus(True, "llama3.2:3b", 100.0, None)
        mock_ollama_service.health_check = AsyncMock(return_value=mock_status)

        for _ in range(10):
            await monitor.check_health()

        # Should only keep last 5 samples
        assert len(monitor.metrics.response_times) == 5

    @pytest.mark.asyncio
    async def test_check_health_returns_computed_metrics(self, monitor, mock_ollama_service):
        """Test that check_health includes avg and P95 metrics."""
        # Perform multiple checks
        for i in range(5):
            mock_status = LlmHealthStatus(True, "llama3.2:3b", float(i * 100), None)
            mock_ollama_service.health_check = AsyncMock(return_value=mock_status)
            await monitor.check_health()

        # Get latest check
        result = await monitor.check_health()

        # Verify computed metrics included
        assert "avg_response_time_ms" in result
        assert "p95_response_time_ms" in result
        assert "success_rate" in result
        assert result["avg_response_time_ms"] > 0
        assert result["p95_response_time_ms"] > 0
