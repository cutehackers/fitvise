"""
DEPRECATED: This module is kept for backward compatibility only.

All database models have been moved to:
    app.infrastructure.persistence.models

Please update your imports to use the new location.
"""

# Import from new location for backward compatibility
from app.infrastructure.persistence.models import DocumentModel

__all__ = ["DocumentModel"]