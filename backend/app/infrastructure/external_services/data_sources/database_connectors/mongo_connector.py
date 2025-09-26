"""MongoDB connector built on PyMongo."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import DatabaseConnector, DatabaseConnectionConfig, ConnectionTestResult, QueryResult

try:  # Optional dependency
    from pymongo import MongoClient  # type: ignore
    from pymongo.errors import PyMongoError  # type: ignore
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore
    PyMongoError = Exception  # type: ignore


class MongoConnector(DatabaseConnector):
    """Simple MongoDB connector supporting sampling and metadata listings."""

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        super().__init__(config)
        self._client: Optional[MongoClient] = None

    # ------------------------------------------------------------------
    def _ensure_dependencies(self) -> None:
        if MongoClient is None:
            raise ModuleNotFoundError("The 'pymongo' package is required for MongoDB connectors")

    def _build_uri(self) -> str:
        custom_uri = self.config.params.get("uri") if self.config.params else None
        if custom_uri:
            return custom_uri
        auth = ""
        if self.config.username:
            password = self.config.password or ""
            auth = f"{self.config.username}:{password}@"
        database_segment = f"/{self.config.database}" if self.config.database else ""
        options = []
        if self.config.use_ssl:
            options.append("tls=true")
        if self.config.params:
            for key, value in self.config.params.items():
                if key == "uri":
                    continue
                options.append(f"{key}={value}")
        options_segment = f"?{'&'.join(options)}" if options else ""
        return f"mongodb://{auth}{self.config.host}:{self.config.port}{database_segment}{options_segment}"

    @property
    def client(self) -> MongoClient:
        self._ensure_dependencies()
        if self._client is None:
            self._client = MongoClient(self._build_uri(), serverSelectionTimeoutMS=5000)
        return self._client

    # ------------------------------------------------------------------
    async def test_connection(self) -> ConnectionTestResult:
        def _sync_test() -> ConnectionTestResult:
            try:
                info = self.client.server_info()
                return ConnectionTestResult(success=True, server_info=info)
            except PyMongoError as exc:  # pragma: no cover - depends on environment
                return ConnectionTestResult(success=False, error=str(exc))

        try:
            return await self._to_thread(_sync_test)
        except Exception as exc:
            return ConnectionTestResult(success=False, error=str(exc))

    async def list_collections(self) -> List[str]:
        def _sync_list() -> List[str]:
            db = self.client[self.config.database] if self.config.database else self.client.get_default_database()
            return sorted(db.list_collection_names())

        return await self._to_thread(_sync_list)

    async def fetch_sample(self, collection: str, limit: int = 5) -> QueryResult:
        def _sync_fetch() -> QueryResult:
            db = self.client[self.config.database] if self.config.database else self.client.get_default_database()
            cursor = db[collection].find({}, limit=limit)
            rows = [self._serialise_document(doc) for doc in cursor]
            columns = sorted({key for row in rows for key in row.keys()})
            return QueryResult(rows=rows, row_count=len(rows), columns=columns, query="find({})")

        return await self._to_thread(_sync_fetch)

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    @staticmethod
    def _serialise_document(document: Dict[str, Any]) -> Dict[str, Any]:
        serialised: Dict[str, Any] = {}
        for key, value in document.items():
            if key == "_id":
                serialised[key] = str(value)
            else:
                serialised[key] = value
        return serialised
