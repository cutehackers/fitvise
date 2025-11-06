"""ML Services infrastructure module.

Provides centralized access to ML model instances (embeddings, etc.)
through dependency injection containers.
"""

from app.infrastructure.ml_services.ml_services_container import (
    MLServicesContainer,
    MLServicesError,
)

__all__ = [
    "MLServicesContainer",
    "MLServicesError",
]
