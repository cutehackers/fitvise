"""Shared abstractions for database connectors."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatabaseConnectionConfig:
    """Connection settings for SQL/NoSQL sources."""

    name: str
    driver: str
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    use_ssl: bool = False

    def dsn(self) -> str:
        """Return a SQLAlchemy-style DSN when possible."""
        auth = ""
        if self.username:
            password = self.password or ""
            auth = f"{self.username}:{password}@"
        return f"{self.driver}://{auth}{self.host}:{self.port}/{self.database or ''}"


@dataclass
class ConnectionTestResult:
    """Outcome of a connectivity test."""

    success: bool
    latency_ms: Optional[float] = None
    server_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "latency_ms": self.latency_ms,
            "server_info": self.server_info or {},
            "error": self.error,
        }


@dataclass
class QueryResult:
    """Data returned from sample extraction."""

    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    query: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "row_count": self.row_count,
            "columns": self.columns,
            "query": self.query,
        }


class DatabaseConnector(ABC):
    """Common interface that all connectors must implement."""

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        self.config = config

    @abstractmethod
    async def test_connection(self) -> ConnectionTestResult:
        """Validate connectivity and return server metadata."""
        raise NotImplementedError

    @abstractmethod
    async def fetch_sample(self, collection: str, limit: int = 5) -> QueryResult:
        """Grab a sample of rows/documents from the data source."""
        raise NotImplementedError

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List tables/collections that could be ingested."""
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - optional override for connectors with pools
        """Override when the concrete connector needs explicit teardown."""
        return None

    # Helper to run blocking DB calls in thread executors
    async def _to_thread(self, func, *args, **kwargs):  # type: ignore[no-untyped-def]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
