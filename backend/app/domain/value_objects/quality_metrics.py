"""Quality metrics value object for RAG system."""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class QualityLevel(str, Enum):
    """Quality tiers - excellent (90%+), good (70-89%), fair (50-69%), poor (30-49%), critical (<30%)."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 70-89%
    FAIR = "fair"           # 50-69%
    POOR = "poor"           # 30-49%
    CRITICAL = "critical"   # 0-29%


class ValidationRule(str, Enum):
    """Types of validation rules."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    INTEGRITY = "integrity"


@dataclass(frozen=True)
class ValidationResult:
    """Result of a single validation rule."""
    rule: ValidationRule
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Optional[Dict] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.details is None:
            object.__setattr__(self, 'details', {})


@dataclass(frozen=True)
class ContentQualityMetrics:
    """Metrics for content quality assessment."""
    
    # Basic content metrics
    total_characters: int = 0
    total_words: int = 0
    total_sentences: int = 0
    total_paragraphs: int = 0
    
    # Language and readability
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    readability_score: Optional[float] = None  # Flesch-Kincaid or similar
    
    # Structure analysis
    has_title: bool = False
    has_headings: bool = False
    has_tables: bool = False
    has_images: bool = False
    has_links: bool = False
    
    # Content completeness
    empty_sections: int = 0
    broken_links: int = 0
    missing_metadata_fields: int = 0
    
    # Encoding and format
    encoding_issues: int = 0
    format_errors: int = 0
    
    @property
    def overall_content_score(self) -> float:
        """Calculate overall content quality score (0.0 to 1.0)."""
        if self.total_words == 0:
            return 0.0
        
        scores = []
        
        # Word count score (normalized)
        word_score = min(1.0, self.total_words / 1000)  # 1000 words = perfect
        scores.append(word_score)
        
        # Structure score
        structure_elements = [
            self.has_title, self.has_headings, 
            self.has_tables or self.has_images or self.has_links
        ]
        structure_score = sum(structure_elements) / len(structure_elements)
        scores.append(structure_score)
        
        # Error penalty
        error_penalty = max(0.0, 1.0 - (self.encoding_issues + self.format_errors) * 0.1)
        scores.append(error_penalty)
        
        return sum(scores) / len(scores)


@dataclass(frozen=True)
class DataQualityMetrics:
    """Quality assessment - validation results, content metrics, errors & ML confidence.
    
    Examples:
        >>> content_metrics = ContentQualityMetrics(total_words=150, has_title=True)
        >>> validation_results = [ValidationResult(ValidationRule.COMPLETENESS, True, 0.9, "Complete")]
        >>> quality = DataQualityMetrics(datetime.utcnow(), validation_results, content_metrics)
        >>> quality.overall_quality_score
        0.85
        >>> quality.quality_level
        QualityLevel.GOOD
    """
    
    # Timestamp
    measured_at: datetime
    
    # Validation results
    validation_results: List[ValidationResult]
    
    # Content quality
    content_metrics: ContentQualityMetrics
    
    # Processing metrics
    processing_time_seconds: float = 0.0
    memory_usage_mb: Optional[float] = None
    
    # Error tracking
    errors_encountered: List[str] = None
    warnings_encountered: List[str] = None
    
    # Source-specific metrics
    source_availability: bool = True
    source_response_time_ms: Optional[float] = None
    
    # Categorization confidence (from ML)
    category_confidence: Optional[float] = None
    predicted_categories: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.errors_encountered is None:
            object.__setattr__(self, 'errors_encountered', [])
        if self.warnings_encountered is None:
            object.__setattr__(self, 'warnings_encountered', [])
        if self.predicted_categories is None:
            object.__setattr__(self, 'predicted_categories', [])
    
    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        if not self.validation_results:
            return self.content_metrics.overall_content_score
        
        # Calculate validation score
        validation_scores = [result.score for result in self.validation_results]
        validation_score = sum(validation_scores) / len(validation_scores)
        
        # Combine with content score
        content_score = self.content_metrics.overall_content_score
        
        # Weight: 60% validation, 40% content
        return validation_score * 0.6 + content_score * 0.4
    
    @property
    def quality_level(self) -> QualityLevel:
        """Get quality level based on overall score."""
        score = self.overall_quality_score
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.FAIR
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors were encountered."""
        return len(self.errors_encountered) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were encountered."""
        return len(self.warnings_encountered) > 0
    
    @property
    def passed_all_validations(self) -> bool:
        """Check if all validation rules passed."""
        return all(result.passed for result in self.validation_results)
    
    @property
    def failed_validation_rules(self) -> List[ValidationRule]:
        """Get list of failed validation rules."""
        return [result.rule for result in self.validation_results if not result.passed]
    
    def get_validation_score(self, rule: ValidationRule) -> Optional[float]:
        """Get score for a specific validation rule."""
        for result in self.validation_results:
            if result.rule == rule:
                return result.score
        return None
    
    def add_error(self, error: str) -> 'DataQualityMetrics':
        """Create new metrics with additional error."""
        new_errors = self.errors_encountered.copy()
        new_errors.append(error)
        
        return DataQualityMetrics(
            measured_at=self.measured_at,
            validation_results=self.validation_results,
            content_metrics=self.content_metrics,
            processing_time_seconds=self.processing_time_seconds,
            memory_usage_mb=self.memory_usage_mb,
            errors_encountered=new_errors,
            warnings_encountered=self.warnings_encountered.copy(),
            source_availability=self.source_availability,
            source_response_time_ms=self.source_response_time_ms,
            category_confidence=self.category_confidence,
            predicted_categories=self.predicted_categories.copy()
        )


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable quality thresholds for validation."""
    
    # Minimum scores (0.0 to 1.0)
    min_overall_score: float = 0.5
    min_completeness_score: float = 0.7
    min_accuracy_score: float = 0.8
    min_consistency_score: float = 0.6
    
    # Content thresholds
    min_word_count: int = 10
    min_character_count: int = 100
    max_error_count: int = 5
    
    # Language detection
    min_language_confidence: float = 0.8
    
    # Processing limits
    max_processing_time_seconds: float = 300.0  # 5 minutes
    max_memory_usage_mb: float = 1024.0  # 1GB
    
    # ML categorization
    min_category_confidence: float = 0.6
    
    def is_acceptable_quality(self, metrics: DataQualityMetrics) -> bool:
        """Check if metrics meet minimum quality thresholds."""
        checks = [
            metrics.overall_quality_score >= self.min_overall_score,
            metrics.content_metrics.total_words >= self.min_word_count,
            metrics.content_metrics.total_characters >= self.min_character_count,
            len(metrics.errors_encountered) <= self.max_error_count,
            metrics.processing_time_seconds <= self.max_processing_time_seconds,
        ]
        
        # Optional checks
        if metrics.content_metrics.language_confidence:
            checks.append(metrics.content_metrics.language_confidence >= self.min_language_confidence)
        
        if metrics.memory_usage_mb:
            checks.append(metrics.memory_usage_mb <= self.max_memory_usage_mb)
        
        if metrics.category_confidence:
            checks.append(metrics.category_confidence >= self.min_category_confidence)
        
        return all(checks)