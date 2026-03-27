"""Use case implementing Task 1.2.3 - database connector orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.infrastructure.external_services.data_sources.database_connectors import (
    DatabaseConnectionConfig,
    DatabaseConnector,
    ConnectionTestResult,
    QueryResult,
    PostgresConnector,
    MySQLConnector,
    MongoConnector,
)


@dataclass
class DatabaseConnectorSpec:
    """Configuration for a single connector execution."""

    connector_type: str
    config: DatabaseConnectionConfig
    sample_collection: Optional[str] = None
    sample_limit: int = 5


class ConnectDatabasesRequest:
    """Request for testing & sampling data sources."""

    def __init__(
        self,
        connectors: Sequence[DatabaseConnectorSpec],
        fetch_samples: bool = True,
    ) -> None:
        self.connectors = list(connectors)
        self.fetch_samples = fetch_samples


@dataclass
class DatabaseConnectorResult:
    """Result bundle for a connector execution."""

    connector_type: str
    config_name: str
    connection_result: ConnectionTestResult
    collections: Optional[List[str]] = None
    sample: Optional[QueryResult] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "connector_type": self.connector_type,
            "config_name": self.config_name,
            "connection_result": self.connection_result.as_dict(),
        }
        if self.collections is not None:
            payload["collections"] = self.collections
        if self.sample is not None:
            payload["sample"] = self.sample.as_dict()
        if self.error:
            payload["error"] = self.error
        return payload


class ConnectDatabasesResponse:
    """Aggregated response for Task 1.2.3."""

    def __init__(self, results: List[DatabaseConnectorResult]) -> None:
        self.results = results

    @property
    def successful(self) -> List[DatabaseConnectorResult]:
        return [result for result in self.results if result.connection_result.success]

    @property
    def failed(self) -> List[DatabaseConnectorResult]:
        return [result for result in self.results if not result.connection_result.success]

    def as_dict(self) -> Dict[str, object]:
        return {
            "successful": [result.as_dict() for result in self.successful],
            "failed": [result.as_dict() for result in self.failed],
            "all": [result.as_dict() for result in self.results],
        }


class ConnectDatabasesUseCase:
    """Instantiate connectors, validate connectivity and pull sample data."""

    def __init__(self) -> None:
        self.connector_map = {
            "postgres": PostgresConnector,
            "postgresql": PostgresConnector,
            "mysql": MySQLConnector,
            "mariadb": MySQLConnector,
            "mongo": MongoConnector,
            "mongodb": MongoConnector,
        }

    async def execute(self, request: ConnectDatabasesRequest) -> ConnectDatabasesResponse:
        results: List[DatabaseConnectorResult] = []

        for spec in request.connectors:
            connector = self._create_connector(spec)
            connection_result = await connector.test_connection()
            collections: Optional[List[str]] = None
            sample: Optional[QueryResult] = None
            error: Optional[str] = None

            try:
                if connection_result.success:
                    collections = await connector.list_collections()
                    if request.fetch_samples:
                        target_collection = spec.sample_collection or (collections[0] if collections else None)
                        if target_collection:
                            sample = await connector.fetch_sample(target_collection, limit=spec.sample_limit)
                else:
                    error = connection_result.error
            except Exception as exc:  # pragma: no cover - depends on runtime DB
                error = str(exc)
            finally:
                await connector.close()

            result = DatabaseConnectorResult(
                connector_type=spec.connector_type,
                config_name=spec.config.name,
                connection_result=connection_result,
                collections=collections,
                sample=sample,
                error=error,
            )
            results.append(result)

        return ConnectDatabasesResponse(results=results)

    def _create_connector(self, spec: DatabaseConnectorSpec) -> DatabaseConnector:
        connector_type = spec.connector_type.lower()
        if connector_type not in self.connector_map:
            raise ValueError(f"Unsupported connector type: {connector_type}")
        connector_cls = self.connector_map[connector_type]
        return connector_cls(spec.config)
