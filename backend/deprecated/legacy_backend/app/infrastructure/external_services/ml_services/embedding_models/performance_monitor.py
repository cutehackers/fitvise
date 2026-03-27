"""Performance monitoring and benchmarking for embedding operations.

Provides comprehensive metrics collection, real-time monitoring, and
automated alerting for embedding service performance optimization.
"""

from __future__ import annotations

import asyncio
import psutil
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Single performance measurement."""

    operation_type: str  # 'single', 'batch', 'query', etc.
    batch_size: int
    processing_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics over a time window."""

    window_start: float
    window_end: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_processing_time_ms: float
    max_processing_time_ms: float
    min_processing_time_ms: float
    avg_memory_usage_mb: float
    max_memory_usage_mb: float
    avg_cpu_percent: float
    throughput_ops_per_second: float
    embeddings_per_second: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_operations / self.total_operations if self.total_operations > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return 1.0 - self.success_rate


class RollingAverage:
    """Rolling average calculator for smooth metrics."""

    def __init__(self, window_size: int = 100) -> None:
        """Initialize rolling average.

        Args:
            window_size: Size of the rolling window
        """
        self.window_size = window_size
        self.values: Deque[float] = deque(maxlen=window_size)
        self.sum = 0.0

    def add(self, value: float) -> None:
        """Add value to rolling average.

        Args:
            value: Value to add
        """
        if len(self.values) == self.window_size:
            # Remove oldest value from sum
            self.sum -= self.values[0]

        self.values.append(value)
        self.sum += value

    def average(self) -> float:
        """Get current average.

        Returns:
            Rolling average value
        """
        return self.sum / len(self.values) if self.values else 0.0

    def reset(self) -> None:
        """Reset rolling average."""
        self.values.clear()
        self.sum = 0.0


class PerformanceAlert:
    """Performance alert configuration."""

    def __init__(
        self,
        name: str,
        threshold: float,
        comparison: str = "greater_than",
        window_minutes: int = 5,
        cooldown_minutes: int = 15
    ) -> None:
        """Initialize performance alert.

        Args:
            name: Alert name
            threshold: Alert threshold value
            comparison: Comparison type ('greater_than', 'less_than')
            window_minutes: Time window for evaluation
            cooldown_minutes: Cooldown period between alerts
        """
        self.name = name
        self.threshold = threshold
        self.comparison = comparison
        self.window_seconds = window_minutes * 60
        self.cooldown_seconds = cooldown_minutes * 60
        self.last_triggered = 0.0
        self.trigger_count = 0

    def should_trigger(self, current_value: float, current_time: float) -> bool:
        """Check if alert should be triggered.

        Args:
            current_value: Current metric value
            current_time: Current timestamp

        Returns:
            True if alert should trigger
        """
        # Check cooldown
        if current_time - self.last_triggered < self.cooldown_seconds:
            return False

        # Check threshold
        if self.comparison == "greater_than":
            should_trigger = current_value > self.threshold
        else:  # less_than
            should_trigger = current_value < self.threshold

        if should_trigger:
            self.last_triggered = current_time
            self.trigger_count += 1

        return should_trigger


class EmbeddingPerformanceMonitor:
    """Comprehensive performance monitoring for embedding operations.

    Features:
    - Real-time metrics collection and aggregation
    - Rolling averages for smooth performance tracking
    - Automated alerting for performance degradation
    - System resource monitoring
    - Historical performance analysis
    """

    def __init__(
        self,
        metrics_window_size: int = 1000,
        aggregation_window_seconds: int = 60,
        alert_cooldown_minutes: int = 5
    ) -> None:
        """Initialize performance monitor.

        Args:
            metrics_window_size: Size of metrics history window
            aggregation_window_seconds: Window for metric aggregation
            alert_cooldown_minutes: Default cooldown for alerts
        """
        self.metrics_window_size = metrics_window_size
        self.aggregation_window_seconds = aggregation_window_seconds

        # Metrics storage
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=metrics_window_size)

        # Rolling averages for key metrics
        self.rolling_metrics = {
            'processing_time_ms': RollingAverage(100),
            'memory_usage_mb': RollingAverage(50),
            'cpu_percent': RollingAverage(50),
            'batch_efficiency': RollingAverage(30),
            'cache_hit_rate': RollingAverage(100),
        }

        # Performance alerts
        self.alerts = {
            'high_processing_time': PerformanceAlert(
                'High Processing Time',
                threshold=1000.0,  # 1 second
                comparison='greater_than',
                window_minutes=2,
                cooldown_minutes=5
            ),
            'high_memory_usage': PerformanceAlert(
                'High Memory Usage',
                threshold=80.0,  # 80% of available memory
                comparison='greater_than',
                window_minutes=1,
                cooldown_minutes=3
            ),
            'low_success_rate': PerformanceAlert(
                'Low Success Rate',
                threshold=0.95,  # 95% success rate
                comparison='less_than',
                window_minutes=5,
                cooldown_minutes=10
            ),
            'low_throughput': PerformanceAlert(
                'Low Throughput',
                threshold=10.0,  # 10 ops per second
                comparison='less_than',
                window_minutes=3,
                cooldown_minutes=5
            ),
        }

        # System monitoring
        self.process = psutil.Process()
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Statistics
        self.total_operations = 0
        self.total_embeddings = 0
        self.start_time = time.time()

    @asynccontextmanager
    async def measure_operation(
        self,
        operation_type: str,
        batch_size: int = 1
    ):
        """Context manager for measuring operation performance.

        Args:
            operation_type: Type of operation being measured
            batch_size: Size of the batch being processed

        Yields:
            Performance measurement context
        """
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()

        success = True
        error_message = None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()

            processing_time_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            cpu_avg = (start_cpu + end_cpu) / 2

            # Record metrics
            metrics = PerformanceMetrics(
                operation_type=operation_type,
                batch_size=batch_size,
                processing_time_ms=processing_time_ms,
                memory_usage_mb=memory_delta,
                cpu_percent=cpu_avg,
                success=success,
                error_message=error_message
            )

            self.record_metrics(metrics)

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics.

        Args:
            metrics: Performance metrics to record
        """
        self.metrics_history.append(metrics)
        self.total_operations += 1
        self.total_embeddings += metrics.batch_size

        # Update rolling averages
        self.rolling_metrics['processing_time_ms'].add(metrics.processing_time_ms)
        self.rolling_metrics['memory_usage_mb'].add(metrics.memory_usage_mb)
        self.rolling_metrics['cpu_percent'].add(metrics.cpu_percent)

        # Calculate and update batch efficiency
        if metrics.batch_size > 1:
            efficiency = metrics.batch_size / max(1, metrics.processing_time_ms / 1000)
            self.rolling_metrics['batch_efficiency'].add(efficiency)

        # Check for alerts
        self._check_alerts(metrics)

    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check if any performance alerts should be triggered.

        Args:
            metrics: Current metrics to evaluate
        """
        current_time = time.time()

        # Check processing time alert
        if self.alerts['high_processing_time'].should_trigger(metrics.processing_time_ms, current_time):
            logger.warning(
                f"Performance Alert: High processing time detected "
                f"- {metrics.processing_time_ms:.2f}ms for {metrics.operation_type}"
            )

        # Check memory usage alert
        current_memory_mb = self._get_memory_usage()
        memory_percent = (current_memory_mb / (psutil.virtual_memory().total / (1024 * 1024))) * 100
        if self.alerts['high_memory_usage'].should_trigger(memory_percent, current_time):
            logger.warning(
                f"Performance Alert: High memory usage detected "
                f"- {memory_percent:.1f}% ({current_memory_mb:.1f}MB)"
            )

        # Check success rate (need window of recent metrics)
        recent_metrics = [
            m for m in self.metrics_history
            if current_time - m.timestamp <= self.alerts['low_success_rate'].window_seconds
        ]

        if recent_metrics:
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            if self.alerts['low_success_rate'].should_trigger(success_rate, current_time):
                logger.warning(
                    f"Performance Alert: Low success rate detected "
                    f"- {success_rate:.1%} over {len(recent_metrics)} operations"
                )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage.

        Returns:
            CPU usage percentage
        """
        try:
            return self.process.cpu_percent()
        except Exception:
            return 0.0

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.

        Returns:
            Current metrics dictionary
        """
        current_time = time.time()

        return {
            'timestamp': current_time,
            'uptime_seconds': current_time - self.start_time,
            'total_operations': self.total_operations,
            'total_embeddings': self.total_embeddings,
            'current_memory_mb': self._get_memory_usage(),
            'current_cpu_percent': self._get_cpu_usage(),
            'rolling_averages': {
                name: avg.average()
                for name, avg in self.rolling_metrics.items()
            },
            'alerts': {
                name: {
                    'threshold': alert.threshold,
                    'trigger_count': alert.trigger_count,
                    'last_triggered': alert.last_triggered,
                    'cooldown_remaining': max(0, alert.cooldown_seconds - (current_time - alert.last_triggered))
                }
                for name, alert in self.alerts.items()
            }
        }

    def get_aggregated_metrics(self, window_seconds: Optional[int] = None) -> Optional[AggregatedMetrics]:
        """Get aggregated metrics over a time window.

        Args:
            window_seconds: Time window in seconds (uses default if None)

        Returns:
            Aggregated metrics, or None if no data in window
        """
        if window_seconds is None:
            window_seconds = self.aggregation_window_seconds

        current_time = time.time()
        window_start = current_time - window_seconds

        # Filter metrics within window
        window_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= window_start
        ]

        if not window_metrics:
            return None

        successful_metrics = [m for m in window_metrics if m.success]

        return AggregatedMetrics(
            window_start=window_start,
            window_end=current_time,
            total_operations=len(window_metrics),
            successful_operations=len(successful_metrics),
            failed_operations=len(window_metrics) - len(successful_metrics),
            avg_processing_time_ms=np.mean([m.processing_time_ms for m in successful_metrics]) if successful_metrics else 0,
            max_processing_time_ms=max([m.processing_time_ms for m in window_metrics]),
            min_processing_time_ms=min([m.processing_time_ms for m in window_metrics]),
            avg_memory_usage_mb=np.mean([m.memory_usage_mb for m in window_metrics]),
            max_memory_usage_mb=max([m.memory_usage_mb for m in window_metrics]),
            avg_cpu_percent=np.mean([m.cpu_percent for m in window_metrics]),
            throughput_ops_per_second=len(window_metrics) / window_seconds,
            embeddings_per_second=sum(m.batch_size for m in window_metrics) / window_seconds,
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Performance summary dictionary
        """
        current_metrics = self.get_current_metrics()
        recent_metrics = self.get_aggregated_metrics(300)  # Last 5 minutes

        summary = {
            'current': current_metrics,
            'recent_5min': recent_metrics.__dict__ if recent_metrics else None,
            'alerts_triggered': sum(alert.trigger_count for alert in self.alerts.values()),
            'uptime_hours': current_metrics['uptime_seconds'] / 3600,
            'avg_embeddings_per_operation': (
                self.total_embeddings / max(1, self.total_operations)
            ),
        }

        # Performance trends
        if len(self.metrics_history) > 10:
            recent_times = [m.processing_time_ms for m in list(self.metrics_history)[-10:]]
            older_times = [m.processing_time_ms for m in list(self.metrics_history)[-50:-40]] if len(self.metrics_history) > 50 else recent_times

            if older_times:
                trend = np.mean(recent_times) - np.mean(older_times)
                summary['performance_trend_ms'] = trend
                summary['performance_status'] = 'improving' if trend < 0 else 'degrading' if trend > 0 else 'stable'

        return summary

    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start background monitoring.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info("Started performance monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped performance monitoring")

    async def _monitoring_loop(self, interval_seconds: int) -> None:
        """Background monitoring loop.

        Args:
            interval_seconds: Monitoring interval
        """
        while self.monitoring_active:
            try:
                await asyncio.sleep(interval_seconds)

                # Log current status
                metrics = self.get_current_metrics()
                logger.info(
                    f"Performance Status: {metrics['total_operations']} ops, "
                    f"{metrics['rolling_averages']['processing_time_ms']:.1f}ms avg, "
                    f"{metrics['current_memory_mb']:.1f}MB memory"
                )

                # Check for performance degradation
                recent = self.get_aggregated_metrics(300)  # 5 minutes
                if recent and recent.success_rate < 0.9:
                    logger.warning(f"Performance degradation detected: {recent.success_rate:.1%} success rate")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

    def reset_metrics(self) -> None:
        """Reset all metrics and statistics."""
        self.metrics_history.clear()
        for avg in self.rolling_metrics.values():
            avg.reset()
        self.total_operations = 0
        self.total_embeddings = 0
        self.start_time = time.time()
        for alert in self.alerts.values():
            alert.trigger_count = 0
            alert.last_triggered = 0
        logger.info("Performance metrics reset")


# Global monitor instance
_global_monitor: Optional[EmbeddingPerformanceMonitor] = None


def get_global_monitor() -> EmbeddingPerformanceMonitor:
    """Get global performance monitor instance.

    Returns:
        Global monitor singleton
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EmbeddingPerformanceMonitor()
    return _global_monitor


async def start_global_monitoring(interval_seconds: int = 30) -> None:
    """Start global performance monitoring.

    Args:
        interval_seconds: Monitoring interval
    """
    monitor = get_global_monitor()
    await monitor.start_monitoring(interval_seconds)


async def stop_global_monitoring() -> None:
    """Stop global performance monitoring."""
    monitor = get_global_monitor()
    await monitor.stop_monitoring()