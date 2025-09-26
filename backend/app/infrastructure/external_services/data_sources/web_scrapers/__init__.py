"""Web scraping framework abstractions (Task 1.2.4)."""
from .scrapy_framework import (
    AuthenticationConfig,
    CrawlRequest,
    CrawledPage,
    CrawlResult,
    WebScrapingFramework,
)

__all__ = [
    "AuthenticationConfig",
    "CrawlRequest",
    "CrawledPage",
    "CrawlResult",
    "WebScrapingFramework",
]
