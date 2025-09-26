"""PostgreSQL connector using SQLAlchemy."""
from __future__ import annotations

from dataclasses import replace

from .base import DatabaseConnectionConfig
from .sqlalchemy_connector import SQLAlchemyDatabaseConnector


class PostgresConnector(SQLAlchemyDatabaseConnector):
    """Connector that targets PostgreSQL sources."""

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        driver = config.driver or "postgresql+psycopg2"
        if not driver.startswith("postgresql"):
            driver = "postgresql+psycopg2"
        if config.driver != driver:
            config = replace(config, driver=driver)
        super().__init__(config, driver_package="psycopg2")

    default_sample_query: str = """SELECT * FROM {table} LIMIT {limit}"""
