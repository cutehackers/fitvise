"""MySQL/MariaDB connector using SQLAlchemy."""
from __future__ import annotations

from dataclasses import replace

from .base import DatabaseConnectionConfig
from .sqlalchemy_connector import SQLAlchemyDatabaseConnector


class MySQLConnector(SQLAlchemyDatabaseConnector):
    """Connector targeting MySQL-compatible databases."""

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        driver = config.driver or "mysql+pymysql"
        if not driver.startswith("mysql"):
            driver = "mysql+pymysql"
        if config.driver != driver:
            config = replace(config, driver=driver)
        super().__init__(config, driver_package="pymysql")

    default_sample_query: str = """SELECT * FROM `{table}` LIMIT {limit}"""
