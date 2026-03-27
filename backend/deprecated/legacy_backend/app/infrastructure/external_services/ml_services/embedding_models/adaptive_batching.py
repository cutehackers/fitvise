"""Adaptive batching engine for embedding operations.

Provides intelligent batch sizing based on memory constraints, text characteristics,
and historical performance data to optimize throughput.
"""

from __future__ import annotations

import os
import psutil
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchPerformanceMetrics:
    """Performance metrics for batch operations."""

    batch_size: int
    processing_time_ms: float
    memory_usage_mb: float
    throughput_items_per_second: float
    success_rate: float
    avg_text_length: float
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class BatchRecommendation:
    """Recommendation for optimal batch configuration."""

    optimal_batch_size: int
    estimated_memory_mb: float
    estimated_throughput: float
    confidence_score: float
    reasoning: List[str] = field(default_factory=list)


class AdaptiveBatchingEngine:
    """Engine for intelligent batch sizing optimization.

    Analyzes system resources, text characteristics, and historical performance
    to determine optimal batch sizes for embedding operations.
    """

    def __init__(
        self,
        memory_limit_percent: float = 70.0,
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        history_size: int = 100
    ) -> None:
        """Initialize adaptive batching engine.

        Args:
            memory_limit_percent: Maximum memory usage percentage
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum allowed batch size
            history_size: Number of historical metrics to retain
        """
        self.memory_limit_percent = memory_limit_percent
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

        # Performance history
        self.performance_history: Deque[BatchPerformanceMetrics] = deque(maxlen=history_size)

        # System constraints
        self.total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.available_memory_mb = self.total_memory_mb * (memory_limit_percent / 100.0)

        # Model-specific parameters (to be updated per model)
        self.base_memory_per_item_mb = 0.5  # Will be updated based on model
        self.model_dimension = 768  # Default, will be updated

        self._stats = {
            "total_recommendations": 0,
            "performance_updates": 0,
            "memory_adjustments": 0,
        }

    def update_model_parameters(self, dimension: int, base_memory_mb: float) -> None:
        """Update model-specific parameters.

        Args:
            dimension: Embedding vector dimension
            base_memory_mb: Base memory usage per item in MB
        """
        self.model_dimension = dimension
        self.base_memory_per_item_mb = base_memory_mb
        logger.info(f"Updated model parameters: dimension={dimension}, base_memory={base_memory_mb:.2f}MB")

    def calculate_text_characteristics(self, texts: List[str]) -> Tuple[float, float, float]:
        """Analyze text characteristics for batch optimization.

        Args:
            texts: List of texts to analyze

        Returns:
            Tuple of (avg_length, max_length, complexity_score)
        """
        if not texts:
            return 0.0, 0.0, 0.0

        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]

        avg_length = np.mean(lengths)
        max_length = np.max(lengths)

        # Complexity based on length variance and special characters
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        special_char_ratio = np.mean([
            sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
            for text in texts
        ])

        complexity_score = (length_variance / 1000) + (special_char_ratio * 2)

        return avg_length, max_length, complexity_score

    def estimate_memory_usage(
        self,
        batch_size: int,
        avg_text_length: float,
        complexity_score: float
    ) -> float:
        """Estimate memory usage for a given batch configuration.

        Args:
            batch_size: Number of texts in batch
            avg_text_length: Average text length in tokens
            complexity_score: Text complexity factor

        Returns:
            Estimated memory usage in MB
        """
        # Base memory for model processing
        base_memory = batch_size * self.base_memory_per_item_mb

        # Memory for input processing
        input_memory = batch_size * avg_text_length * 0.001  # ~1 byte per character

        # Memory for output vectors
        output_memory = batch_size * self.model_dimension * 4 / (1024 * 1024)  # float32 bytes

        # Complexity overhead
        complexity_overhead = complexity_score * batch_size * 0.1

        total_memory = base_memory + input_memory + output_memory + complexity_overhead
        return total_memory

    def analyze_historical_performance(self) -> Optional[BatchRecommendation]:
        """Analyze historical performance to find optimal batch size.

        Returns:
            Recommendation based on historical data, or None if insufficient data
        """
        if len(self.performance_history) < 5:
            return None

        # Group by batch size and calculate average performance
        batch_performance = {}
        for metrics in self.performance_history:
            if metrics.batch_size not in batch_performance:
                batch_performance[metrics.batch_size] = []
            batch_performance[metrics.batch_size].append(metrics)

        # Find best performing batch size
        best_batch_size = None
        best_throughput = 0.0
        best_success_rate = 0.0

        for batch_size, metrics_list in batch_performance.items():
            avg_throughput = np.mean([m.throughput_items_per_second for m in metrics_list])
            avg_success_rate = np.mean([m.success_rate for m in metrics_list])

            # Prioritize high success rate, then throughput
            if avg_success_rate >= 0.95 and avg_throughput > best_throughput:
                best_batch_size = batch_size
                best_throughput = avg_throughput
                best_success_rate = avg_success_rate

        if best_batch_size:
            confidence = min(len(batch_performance[best_batch_size]) / 10.0, 1.0)
            reasoning = [
                f"Historical best performer with {best_throughput:.1f} items/sec",
                f"Success rate: {best_success_rate:.1%}",
                f"Based on {len(batch_performance[best_batch_size])} data points"
            ]

            return BatchRecommendation(
                optimal_batch_size=best_batch_size,
                estimated_memory_mb=self.estimate_memory_usage(best_batch_size, 50, 0.5),
                estimated_throughput=best_throughput,
                confidence_score=confidence,
                reasoning=reasoning
            )

        return None

    def calculate_optimal_batch_size(self, texts: List[str]) -> BatchRecommendation:
        """Calculate optimal batch size for given texts.

        Args:
            texts: List of texts to embed

        Returns:
            Batch recommendation with optimal size and reasoning
        """
        self._stats["total_recommendations"] += 1

        # Analyze text characteristics
        avg_length, max_length, complexity = self.calculate_text_characteristics(texts)

        # Check historical performance first
        historical_rec = self.analyze_historical_performance()
        if historical_rec and historical_rec.confidence_score > 0.7:
            return historical_rec

        # Calculate memory constraints
        max_memory_constrained_size = int(
            self.available_memory_mb / self.base_memory_per_item_mb
        )

        # Calculate performance-optimized size based on text characteristics
        if avg_length < 50:  # Short texts
            performance_size = min(32, self.max_batch_size)
        elif avg_length < 200:  # Medium texts
            performance_size = min(16, self.max_batch_size)
        else:  # Long texts
            performance_size = min(8, self.max_batch_size)

        # Apply complexity adjustments
        if complexity > 1.0:
            performance_size = max(1, performance_size // 2)

        # Consider CPU cores
        cpu_optimized_size = os.cpu_count() * 2

        # Final optimization
        candidates = [
            max_memory_constrained_size,
            performance_size,
            cpu_optimized_size,
            self.min_batch_size
        ]

        optimal_size = min(
            max(candidates),
            self.max_batch_size,
            len(texts)  # Don't exceed input size
        )
        optimal_size = max(optimal_size, self.min_batch_size)

        # Calculate estimates
        estimated_memory = self.estimate_memory_usage(optimal_size, avg_length, complexity)

        # Generate reasoning
        reasoning = [
            f"Memory constraint allows up to {max_memory_constrained_size} items",
            f"Text analysis suggests {performance_size} items (avg length: {avg_length:.0f})",
            f"CPU optimization suggests {cpu_optimized_size} items",
            f"Complexity score: {complexity:.2f}"
        ]

        if historical_rec:
            reasoning.insert(0, f"Historical data suggests {historical_rec.optimal_batch_size} items")
            # Blend historical suggestion with current analysis
            optimal_size = int((optimal_size + historical_rec.optimal_batch_size) / 2)

        return BatchRecommendation(
            optimal_batch_size=optimal_size,
            estimated_memory_mb=estimated_memory,
            estimated_throughput=self._estimate_throughput(optimal_size, avg_length),
            confidence_score=0.8,
            reasoning=reasoning
        )

    def _estimate_throughput(self, batch_size: int, avg_text_length: float) -> float:
        """Estimate processing throughput.

        Args:
            batch_size: Batch size
            avg_text_length: Average text length

        Returns:
            Estimated throughput in items per second
        """
        # Base throughput degrades with text length
        base_throughput = 50.0  # items per second for short texts
        length_factor = max(0.2, 1.0 - (avg_text_length / 1000))

        # Batch efficiency (larger batches are more efficient up to a point)
        batch_factor = min(1.5, 1.0 + (batch_size / 64))

        estimated_throughput = base_throughput * length_factor * batch_factor
        return estimated_throughput

    def record_batch_performance(
        self,
        batch_size: int,
        processing_time_ms: float,
        memory_usage_mb: float,
        success_count: int,
        total_count: int,
        texts: List[str]
    ) -> None:
        """Record performance metrics for a batch operation.

        Args:
            batch_size: Size of the batch
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory used in MB
            success_count: Number of successful embeddings
            total_count: Total number of embeddings
            texts: Texts that were processed
        """
        avg_length, _, _ = self.calculate_text_characteristics(texts)

        throughput = (success_count / (processing_time_ms / 1000.0)) if processing_time_ms > 0 else 0
        success_rate = success_count / total_count if total_count > 0 else 0

        metrics = BatchPerformanceMetrics(
            batch_size=batch_size,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            throughput_items_per_second=throughput,
            success_rate=success_rate,
            avg_text_length=avg_length
        )

        self.performance_history.append(metrics)
        self._stats["performance_updates"] += 1

        # Log significant performance events
        if success_rate < 0.9:
            logger.warning(f"Low success rate: {success_rate:.1%} for batch size {batch_size}")

        if memory_usage_mb > self.available_memory_mb * 0.8:
            logger.warning(f"High memory usage: {memory_usage_mb:.1f}MB for batch size {batch_size}")
            self._stats["memory_adjustments"] += 1

    def get_stats(self) -> dict:
        """Get batching engine statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self._stats,
            "performance_history_size": len(self.performance_history),
            "total_memory_mb": self.total_memory_mb,
            "available_memory_mb": self.available_memory_mb,
            "model_dimension": self.model_dimension,
            "base_memory_per_item_mb": self.base_memory_per_item_mb,
        }

    def get_performance_summary(self) -> dict:
        """Get summary of recent performance.

        Returns:
            Performance summary dictionary
        """
        if not self.performance_history:
            return {"message": "No performance data available"}

        recent_metrics = list(self.performance_history)[-20:]  # Last 20 entries

        return {
            "recent_batch_sizes": [m.batch_size for m in recent_metrics],
            "avg_throughput": np.mean([m.throughput_items_per_second for m in recent_metrics]),
            "avg_success_rate": np.mean([m.success_rate for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage_mb for m in recent_metrics]),
            "best_batch_size": max(
                set(m.batch_size for m in recent_metrics),
                key=lambda bs: np.mean([m.throughput_items_per_second for m in recent_metrics if m.batch_size == bs])
            )
        }