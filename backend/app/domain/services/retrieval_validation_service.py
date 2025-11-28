"""Retrieval validation domain service.

This module contains the RetrievalValidationService domain service that provides
comprehensive validation for retrieval operations with business rules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.domain.entities.query_context import QueryContext
from app.domain.entities.vector_retrieval_config import VectorRetrievalConfig
from app.domain.exceptions.retrieval_exceptions import QueryValidationError


class RetrievalValidationService:
    """Domain service for validating retrieval requests and responses.

    This service provides comprehensive validation logic for retrieval operations
    including query validation, configuration validation, and business rule enforcement.
    """

    def __init__(self) -> None:
        """Initialize retrieval validation service."""
        self._profanity_filter = self._initialize_profanity_filter()
        self._blocked_terms = self._initialize_blocked_terms()
        self._query_patterns = self._initialize_query_patterns()

    def validate_query(self, query: str, config: VectorRetrievalConfig) -> None:
        """Validate a search query against business rules.

        Args:
            query: The search query to validate
            config: Retrieval configuration

        Raises:
            QueryValidationError: If query violates any validation rules
        """
        if not query:
            raise QueryValidationError("Query cannot be empty")

        # Basic validation
        query = query.strip()
        if not query:
            raise QueryValidationError("Query cannot be empty or whitespace only")

        if len(query) > 10000:
            raise QueryValidationError("Query too long: maximum 10000 characters")

        # Content validation
        self._validate_query_content(query)

        # Pattern validation
        self._validate_query_patterns(query)

        # Configuration-specific validation
        self._validate_query_against_config(query, config)

    def validate_context(self, context: QueryContext) -> None:
        """Validate query context.

        Args:
            context: Query context to validate

        Raises:
            QueryValidationError: If context is invalid
        """
        if context is None:
            return  # Context is optional

        # Validate query text if present
        if context.query_text:
            self.validate_query(context.query_text, VectorRetrievalConfig())

        # Validate request metadata
        self._validate_request_metadata(context.request_metadata)

        # Validate user preferences
        self._validate_user_preferences(context.user_preferences)

        # Validate execution preferences
        self._validate_execution_preferences(context.execution_preferences)

    def validate_config(self, config: VectorRetrievalConfig) -> None:
        """Validate retrieval configuration.

        Args:
            config: Configuration to validate

        Raises:
            QueryValidationError: If configuration is invalid
        """
        # Basic validation is already done in VectorRetrievalConfig.__post_init__
        # Additional business rules validation here

        # Validate search mode compatibility
        self._validate_search_mode_compatibility(config)

        # Validate performance parameters
        self._validate_performance_parameters(config)

        # Validate business rule consistency
        self._validate_business_rule_consistency(config)

    def validate_results(self, results: Any, config: VectorRetrievalConfig) -> None:
        """Validate retrieval results.

        Args:
            results: Retrieval results to validate
            config: Configuration used for retrieval

        Raises:
            QueryValidationError: If results are invalid
        """
        if results is None:
            raise QueryValidationError("Results cannot be None")

        # Validate result count
        if hasattr(results, 'result_count'):
            if results.result_count > config.max_results * 2:  # Allow some flexibility
                raise QueryValidationError(
                    f"Result count ({results.result_count}) exceeds expected maximum ({config.max_results * 2})"
                )

        # Validate similarity scores if available
        if hasattr(results, 'similarity_scores'):
            self._validate_similarity_scores(results.similarity_scores, config)

    def _validate_query_content(self, query: str) -> None:
        """Validate query content against business rules."""
        # Check for profanity
        if self._contains_profanity(query):
            raise QueryValidationError("Query contains inappropriate content")

        # Check for blocked terms
        if self._contains_blocked_terms(query):
            raise QueryValidationError("Query contains blocked terms")

        # Check for potentially harmful content
        if self._contains_harmful_patterns(query):
            raise QueryValidationError("Query contains potentially harmful content")

        # Check for SQL injection attempts
        if self._contains_sql_injection(query):
            raise QueryValidationError("Query contains potentially malicious SQL patterns")

        # Check for XSS attempts
        if self._contains_xss_patterns(query):
            raise QueryValidationError("Query contains potentially malicious script patterns")

    def _validate_query_patterns(self, query: str) -> None:
        """Validate query patterns."""
        # Check for repeated characters (potential spam)
        if self._has_excessive_repetition(query):
            raise QueryValidationError("Query contains excessive repetition")

        # Check for gibberish
        if self._is_gibberish(query):
            raise QueryValidationError("Query appears to be gibberish or invalid input")

        # Check for encoding issues
        if self._has_encoding_issues(query):
            raise QueryValidationError("Query contains invalid character encoding")

    def _validate_query_against_config(self, query: str, config: VectorRetrievalConfig) -> None:
        """Validate query against specific configuration."""
        # Minimum query length for strict search
        if config.is_strict_search() and len(query.split()) < 2:
            raise QueryValidationError("Strict search requires at least 2 words")

        # Check query complexity for advanced features
        if config.should_apply_reranking() and len(query) < 5:
            # Reranking doesn't make sense for very short queries
            pass  # Could warn here, but not block

        # Validate language if specified
        language = config.metadata.get("language")
        if language and not self._is_valid_language(query, language):
            raise QueryValidationError(f"Query language doesn't match expected language: {language}")

    def _validate_search_mode_compatibility(self, config: VectorRetrievalConfig) -> None:
        """Validate search mode compatibility."""
        # Check if filters are supported by search mode
        if not config.search_mode.supports_filters() and config.metadata_filters:
            raise QueryValidationError(
                f"Search mode {config.search_mode.value} does not support metadata filters"
            )

        # Check if reranking is compatible with search mode
        if config.should_apply_reranking() and not config.search_mode.supports_hybrid_search():
            # This might be a warning rather than an error
            pass

    def _validate_performance_parameters(self, config: VectorRetrievalConfig) -> None:
        """Validate performance-related parameters."""
        # Check for performance-impacting configurations
        if config.max_results > 1000:
            raise QueryValidationError("Maximum results too large for performance: > 1000")

        if config.similarity_threshold < 0.1 and config.max_results > 500:
            raise QueryValidationError(
                "Low similarity threshold with high max_results may cause performance issues"
            )

        if config.timeout_seconds > 300:  # 5 minutes
            raise QueryValidationError("Timeout too large: > 300 seconds")

    def _validate_business_rule_consistency(self, config: VectorRetrievalConfig) -> None:
        """Validate business rule consistency."""
        # Check reranking configuration consistency
        if config.reranking_enabled and config.rerank_top_k > config.max_results:
            raise QueryValidationError(
                f"Rerank top_k ({config.rerank_top_k}) cannot exceed max_results ({config.max_results})"
            )

        # Check boosting configuration
        if config.boost_recent and config.boost_factor <= 1.0:
            # Boost factor > 1.0 should be used with boost_recent
            raise QueryValidationError("Boost factor must be > 1.0 when boost_recent is enabled")

    def _validate_request_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate request metadata."""
        # Validate client IP format
        if "client_ip" in metadata:
            ip = metadata["client_ip"]
            if not isinstance(ip, str) or not ip or ip.strip() == "":
                raise QueryValidationError("Client IP must be a non-empty string")

            # Basic IP format validation
            if not self._is_valid_ip(ip):
                raise QueryValidationError(f"Invalid IP address format: {ip}")

        # Validate user agent if present
        if "user_agent" in metadata:
            user_agent = metadata["user_agent"]
            if not isinstance(user_agent, str) or len(user_agent) > 500:
                raise QueryValidationError("User agent must be a string under 500 characters")

    def _validate_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Validate user preferences."""
        # Validate language preference
        if "language" in preferences:
            language = preferences["language"]
            valid_languages = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru"]
            if language not in valid_languages:
                raise QueryValidationError(f"Invalid language preference: {language}")

        # Validate max tokens preference
        if "max_tokens" in preferences:
            max_tokens = preferences["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens > 32000:
                raise QueryValidationError("Max tokens must be an integer between 1 and 32000")

        # Validate similarity threshold preference
        if "similarity_threshold" in preferences:
            threshold = preferences["similarity_threshold"]
            if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                raise QueryValidationError("Similarity threshold must be between 0.0 and 1.0")

    def _validate_execution_preferences(self, preferences: Dict[str, Any]) -> None:
        """Validate execution preferences."""
        # Validate timeout override
        if "timeout_seconds" in preferences:
            timeout = preferences["timeout_seconds"]
            if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 600:
                raise QueryValidationError("Timeout must be between 0 and 600 seconds")

        # Validate priority
        if "priority" in preferences:
            priority = preferences["priority"]
            valid_priorities = ["low", "normal", "high", "critical"]
            if priority not in valid_priorities:
                raise QueryValidationError(f"Invalid priority: {priority}")

    def _validate_similarity_scores(self, scores: List[float], config: VectorRetrievalConfig) -> None:
        """Validate similarity scores."""
        if not scores:
            return  # Empty list is valid

        for i, score in enumerate(scores):
            if not isinstance(score, (int, float)):
                raise QueryValidationError(f"Similarity score at index {i} is not numeric: {score}")

            if not 0.0 <= score <= 1.0:
                raise QueryValidationError(f"Similarity score at index {i} out of range: {score}")

        # Check if scores meet threshold requirements
        if config.similarity_threshold > 0:
            below_threshold = sum(1 for score in scores if score < config.similarity_threshold)
            if below_threshold == len(scores):
                raise QueryValidationError(
                    f"All similarity scores are below threshold ({config.similarity_threshold})"
                )

    def _contains_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        text_lower = text.lower()
        return any(word in text_lower for word in self._profanity_filter)

    def _contains_blocked_terms(self, text: str) -> bool:
        """Check if text contains blocked terms."""
        text_lower = text.lower()
        return any(term in text_lower for term in self._blocked_terms)

    def _contains_harmful_patterns(self, text: str) -> bool:
        """Check for potentially harmful patterns."""
        harmful_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone numbers
            r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',  # Credit card numbers
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # Email addresses
        ]

        import re
        for pattern in harmful_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _contains_sql_injection(self, text: str) -> bool:
        """Check for SQL injection patterns."""
        sql_patterns = [
            "union select", "drop table", "insert into", "delete from",
            "update set", "exec sp_", "xp_cmdshell", "--", "/*", "*/"
        ]

        text_lower = text.lower()
        return any(pattern in text_lower for pattern in sql_patterns)

    def _contains_xss_patterns(self, text: str) -> bool:
        """Check for XSS patterns."""
        xss_patterns = [
            "<script", "</script>", "javascript:", "onload=", "onerror=",
            "onclick=", "onmouseover=", "eval(", "alert("
        ]

        text_lower = text.lower()
        return any(pattern in text_lower for pattern in xss_patterns)

    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition."""
        # Check for same character repeated > 10 times
        for i, char in enumerate(text):
            if i > 10 and text[i] == text[i-1] == text[i-2] == text[i-3] == text[i-4] == text[i-5]:
                consecutive = 1
                j = i + 1
                while j < len(text) and text[j] == char:
                    consecutive += 1
                    j += 1
                if consecutive > 10:
                    return True

        # Check for word repetition
        words = text.lower().split()
        if len(words) > 20:
            word_count = {}
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1
                if word_count[word] > len(words) * 0.3:  # Same word > 30% of query
                    return True

        return False

    def _is_gibberish(self, text: str) -> bool:
        """Check if text appears to be gibberish."""
        # Remove whitespace and convert to lowercase
        clean_text = ''.join(text.lower().split())

        if len(clean_text) < 3:
            return False

        # Count unique characters
        unique_chars = len(set(clean_text))
        total_chars = len(clean_text)

        # If very few unique characters, likely gibberish
        if unique_chars / total_chars < 0.1:
            return True

        # Check for alternating patterns (ababab)
        if len(clean_text) >= 6:
            pattern1 = clean_text[0] + clean_text[1]
            pattern_count = 0
            for i in range(0, len(clean_text) - 1, 2):
                if i + 1 < len(clean_text) and clean_text[i] == clean_text[0] and clean_text[i+1] == clean_text[1]:
                    pattern_count += 1

            if pattern_count > len(clean_text) / 4:
                return True

        return False

    def _has_encoding_issues(self, text: str) -> bool:
        """Check for character encoding issues."""
        try:
            # Try to encode and decode
            text.encode('utf-8').decode('utf-8')
            return False
        except UnicodeError:
            return True

    def _is_valid_language(self, text: str, expected_language: str) -> bool:
        """Simple language validation (placeholder for actual language detection)."""
        # This is a simplified implementation
        # In practice, you'd use a proper language detection library

        # Basic heuristics for common languages
        if expected_language == "en":
            # Check for English characters
            english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
            return english_chars / len(text) > 0.5 if text else True
        elif expected_language == "zh":
            # Check for Chinese characters
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            return chinese_chars / len(text) > 0.3 if text else True
        elif expected_language == "ja":
            # Check for Japanese characters
            japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')
            return japanese_chars / len(text) > 0.3 if text else True

        return True  # Default to allowing

    def _is_valid_ip(self, ip: str) -> bool:
        """Basic IP address validation."""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            for part in parts:
                num = int(part)
                if not 0 <= num <= 255:
                    return False
            return True
        except (ValueError, AttributeError):
            return False

    def _initialize_profanity_filter(self) -> List[str]:
        """Initialize profanity filter list."""
        # This would typically be loaded from a configuration or database
        # Simplified placeholder list
        return []  # Empty for now - would contain actual profanity words

    def _initialize_blocked_terms(self) -> List[str]:
        """Initialize blocked terms list."""
        # This would typically be loaded from a configuration or database
        # Simplified placeholder list
        return ["admin", "password", "secret"]  # Example blocked terms

    def _initialize_query_patterns(self) -> Dict[str, Any]:
        """Initialize query pattern configurations."""
        return {
            "min_query_length": 1,
            "max_query_length": 10000,
            "max_repetition_length": 10,
            "gibberish_threshold": 0.1,
        }