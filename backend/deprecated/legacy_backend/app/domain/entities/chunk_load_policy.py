"""Chunk load policy for RAG pipeline Task 3 (Embedding Generation).

This module defines the policy for how Task 3 should handle chunk loading
from Task 2, including fallback strategies when chunks are missing.
"""

from enum import Enum


class ChunkLoadPolicy(str, Enum):
    """Policy for loading chunks in Task 3 (Embedding Generation).

    This enum defines how Task 3 should behave when loading chunks created by Task 2:

    - EXISTING_ONLY: Strict mode - fail immediately if chunks are missing (DEFAULT)
    - SENTENCE_FALLBACK: Allow fallback to sentence splitter if chunks missing
    - SEMANTIC_FALLBACK: Allow fallback to semantic splitter if chunks missing

    The default policy (EXISTING_ONLY) enforces strict separation between Task 2
    (chunking) and Task 3 (embedding), catching configuration issues immediately.
    """

    EXISTING_ONLY = "existing_only"
    """Only use existing chunks from Task 2. Fail if chunks are missing.

    This is the DEFAULT and RECOMMENDED policy for production use.
    It enforces strict Task 2 → Task 3 separation and fails fast if chunks
    are missing, making configuration issues immediately visible.

    Use this when:
    - Running production pipelines
    - You want to ensure Task 2 ran successfully
    - You want to catch missing chunks immediately
    """

    SENTENCE_FALLBACK = "sentence_fallback"
    """Use existing chunks, fallback to sentence splitter if missing.

    This policy allows Task 3 to generate chunks using the sentence splitter
    if chunks from Task 2 are missing. This is faster than semantic chunking
    but may produce lower quality chunks.

    Use this when:
    - Rapid recovery from missing chunks is needed
    - Development/testing scenarios
    - Chunk quality is less critical than speed
    """

    SEMANTIC_FALLBACK = "semantic_fallback"
    """Use existing chunks, fallback to semantic splitter if missing.

    This policy allows Task 3 to generate chunks using the semantic splitter
    (same as Task 2) if chunks are missing. This maintains chunk quality but
    is slower than sentence splitting.

    Use this when:
    - Recovery from missing chunks is needed
    - Chunk quality must match Task 2 output
    - Development/testing scenarios where Task 2 wasn't run
    """

    def requires_fallback(self) -> bool:
        """Check if this policy allows fallback chunking.

        Returns:
            True if fallback is allowed, False if strict mode
        """
        return self in (
            ChunkLoadPolicy.SENTENCE_FALLBACK,
            ChunkLoadPolicy.SEMANTIC_FALLBACK,
        )

    def uses_semantic_chunking(self) -> bool:
        """Check if fallback uses semantic chunking.

        Returns:
            True if semantic fallback, False otherwise
        """
        return self == ChunkLoadPolicy.SEMANTIC_FALLBACK

    def __str__(self) -> str:
        """Return human-readable policy name."""
        return self.value.replace("_", " ").title()
