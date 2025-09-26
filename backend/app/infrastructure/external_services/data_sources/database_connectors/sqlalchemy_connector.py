"""Base connector implementation using SQLAlchemy."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .base import DatabaseConnector, DatabaseConnectionConfig, ConnectionTestResult, QueryResult

try:  # Optional dependency
    from sqlalchemy import create_engine, text, inspect  # type: ignore
    from sqlalchemy.engine import Engine
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore
    text = None  # type: ignore
    inspect = None  # type: ignore
    Engine = Any  # type: ignore


class SQLAlchemyDatabaseConnector(DatabaseConnector):
    """Shared logic for connectors backed by SQLAlchemy."""

    default_sample_query: str = """SELECT * FROM {table} LIMIT {limit}"""

    def __init__(self, config: DatabaseConnectionConfig, driver_package: str) -> None:
        super().__init__(config)
        self.driver_package = driver_package
        self._engine: Optional[Engine] = None

    # ------------------------------------------------------------------
    def _ensure_dependencies(self) -> None:
        if create_engine is None or text is None or inspect is None:
            raise ModuleNotFoundError(
                "SQLAlchemy is required for this connector but is not installed."
            )
        try:
            __import__(self.driver_package)
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
            raise ModuleNotFoundError(
                f"The database driver '{self.driver_package}' is not installed"
            ) from exc

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._ensure_dependencies()
            connect_args: Dict[str, Any] = {}
            if self.config.use_ssl:
                connect_args["ssl"] = True
            self._engine = create_engine(self.config.dsn(), connect_args=connect_args, pool_pre_ping=True)
        return self._engine

    # ------------------------------------------------------------------
    async def test_connection(self) -> ConnectionTestResult:
        def _sync_test() -> ConnectionTestResult:
            start = time.perf_counter()
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                try:
                    server_version = conn.execute(text("SELECT version() as version")).scalar()
                except Exception:
                    server_version = None
            latency = (time.perf_counter() - start) * 1000
            return ConnectionTestResult(
                success=True,
                latency_ms=latency,
                server_info={"version": server_version},
            )

        try:
            return await self._to_thread(_sync_test)
        except Exception as exc:
            return ConnectionTestResult(success=False, error=str(exc))

    async def list_collections(self) -> List[str]:
        def _sync_list() -> List[str]:
            inspector = inspect(self.engine)
            schema = self.config.schema
            tables = inspector.get_table_names(schema=schema)
            views = inspector.get_view_names(schema=schema)
            return sorted(set(tables + views))

        return await self._to_thread(_sync_list)

    async def fetch_sample(self, collection: str, limit: int = 5) -> QueryResult:
        def _sync_query() -> QueryResult:
            query_str = self.default_sample_query.format(table=collection, limit=limit)
            with self.engine.connect() as conn:
                result = conn.execute(text(query_str))
                rows = [dict(row._mapping) for row in result]
                columns = result.keys()
            return QueryResult(rows=rows, row_count=len(rows), columns=list(columns), query=query_str)

        return await self._to_thread(_sync_query)

    async def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
