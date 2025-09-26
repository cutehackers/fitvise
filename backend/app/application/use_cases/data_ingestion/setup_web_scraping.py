"""Use case for configuring the web scraping framework (Task 1.2.4)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from app.infrastructure.external_services.data_sources.web_scrapers import (
    WebScrapingFramework,
    CrawlRequest,
    AuthenticationConfig,
    CrawlResult,
)


class SetupWebScrapingRequest:
    """Parameters for executing a crawl job."""

    def __init__(
        self,
        start_urls: Sequence[str],
        allowed_domains: Optional[Sequence[str]] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        include_patterns: Optional[Sequence[str]] = None,
        exclude_patterns: Optional[Sequence[str]] = None,
        follow_css_selectors: Optional[Sequence[str]] = None,
        follow_xpath: Optional[Sequence[str]] = None,
        authentication: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        follow_external_links: bool = False,
    ) -> None:
        self.start_urls = list(start_urls)
        self.allowed_domains = list(allowed_domains) if allowed_domains else None
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.include_patterns = list(include_patterns) if include_patterns else None
        self.exclude_patterns = list(exclude_patterns) if exclude_patterns else None
        self.follow_css_selectors = list(follow_css_selectors) if follow_css_selectors else None
        self.follow_xpath = list(follow_xpath) if follow_xpath else None
        auth_config = authentication or {}
        self.authentication = AuthenticationConfig(
            auth_type=auth_config.get("auth_type", "none"),
            username=auth_config.get("username"),
            password=auth_config.get("password"),
            token=auth_config.get("token"),
            headers=auth_config.get("headers", {}),
            cookies=auth_config.get("cookies", {}),
        )
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.follow_external_links = follow_external_links


class SetupWebScrapingResponse:
    """Summaries returned to the caller after a crawl."""

    def __init__(self, result: CrawlResult) -> None:
        self.result = result

    @property
    def pages(self) -> List[str]:
        return [page.url for page in self.result.pages]

    @property
    def errors(self) -> List[str]:
        return self.result.errors

    def as_dict(self) -> Dict[str, object]:
        payload = self.result.as_dict()
        payload["pages"] = self.pages
        return payload


class SetupWebScrapingUseCase:
    """Runs crawls through the WebScrapingFramework."""

    def __init__(self, framework: Optional[WebScrapingFramework] = None) -> None:
        self.framework = framework or WebScrapingFramework()

    async def execute(self, request: SetupWebScrapingRequest) -> SetupWebScrapingResponse:
        crawl_request = CrawlRequest(
            start_urls=request.start_urls,
            allowed_domains=request.allowed_domains,
            max_depth=request.max_depth,
            max_pages=request.max_pages,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
            follow_css_selectors=request.follow_css_selectors,
            follow_xpath=request.follow_xpath,
            authentication=request.authentication,
            headers=request.headers,
            cookies=request.cookies,
            follow_external_links=request.follow_external_links,
        )
        result = await self.framework.crawl(crawl_request)
        return SetupWebScrapingResponse(result=result)
