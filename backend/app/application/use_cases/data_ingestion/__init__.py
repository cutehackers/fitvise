"""Use cases covering core ingestion infrastructure (Task 1.2.*)."""
from .setup_airflow import (
    SetupAirflowEnvironmentRequest,
    SetupAirflowEnvironmentResponse,
    SetupAirflowEnvironmentUseCase,
)
from .integrate_tika import (
    IntegrateTikaRequest,
    IntegrateTikaResponse,
    IntegrateTikaUseCase,
)
from .connect_databases import (
    DatabaseConnectorSpec,
    ConnectDatabasesRequest,
    ConnectDatabasesResponse,
    ConnectDatabasesUseCase,
)
from .setup_web_scraping import (
    SetupWebScrapingRequest,
    SetupWebScrapingResponse,
    SetupWebScrapingUseCase,
)

__all__ = [
    "SetupAirflowEnvironmentRequest",
    "SetupAirflowEnvironmentResponse",
    "SetupAirflowEnvironmentUseCase",
    "IntegrateTikaRequest",
    "IntegrateTikaResponse",
    "IntegrateTikaUseCase",
    "DatabaseConnectorSpec",
    "ConnectDatabasesRequest",
    "ConnectDatabasesResponse",
    "ConnectDatabasesUseCase",
    "SetupWebScrapingRequest",
    "SetupWebScrapingResponse",
    "SetupWebScrapingUseCase",
]
