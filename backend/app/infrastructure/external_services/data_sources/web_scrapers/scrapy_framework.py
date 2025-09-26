"""Scrapy-inspired web scraping framework with graceful fallbacks (Task 1.2.4)."""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

try:  # Optional heavy dependency
    from scrapy import Spider, Request  # type: ignore
    from scrapy.crawler import CrawlerProcess  # type: ignore

    _SCRAPY_AVAILABLE = True
except Exception:  # pragma: no cover
    Spider = object  # type: ignore
    Request = object  # type: ignore
    CrawlerProcess = None  # type: ignore
    _SCRAPY_AVAILABLE = False

import logging

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

_logger = logging.getLogger(__name__)


@dataclass
class AuthenticationConfig:
    """Authentication parameters for crawl jobs."""

    auth_type: str = "none"  # none|basic|bearer|header|cookie
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)


@dataclass
class CrawlRequest:
    """Definition of a scraping job."""

    start_urls: List[str]
    allowed_domains: Optional[List[str]] = None
    max_depth: int = 2
    max_pages: int = 50
    follow_css_selectors: Optional[List[str]] = None
    follow_xpath: Optional[List[str]] = None
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 20
    follow_external_links: bool = False
    respect_robots_txt: bool = False  # Placeholder for future expansion


@dataclass
class CrawledPage:
    """Represents a single crawled page."""

    url: str
    status_code: int
    content: str
    extracted_links: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class CrawlResult:
    """Outcome of a crawl."""

    pages: List[CrawledPage]
    elapsed_seconds: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def as_dict(self) -> Dict[str, any]:  # type: ignore[override]
        return {
            "page_count": self.page_count,
            "elapsed_seconds": self.elapsed_seconds,
            "errors": self.errors,
            "warnings": self.warnings,
            "pages": [
                {
                    "url": page.url,
                    "status_code": page.status_code,
                    "metadata": page.metadata,
                    "extracted_links": page.extracted_links,
                }
                for page in self.pages
            ],
        }


class WebScrapingFramework:
    """Scraping helper that uses Scrapy when present, otherwise falls back."""

    def __init__(self) -> None:
        self.scrapy_available = _SCRAPY_AVAILABLE and CrawlerProcess is not None

    async def crawl(self, request: CrawlRequest) -> CrawlResult:
        if self.scrapy_available:
            try:
                return await asyncio.to_thread(self._crawl_with_scrapy, request)
            except Exception as exc:  # pragma: no cover - fallback path
                _logger.warning("Scrapy crawl failed, using httpx fallback: %s", exc)

        return await asyncio.to_thread(self._crawl_with_httpx, request)

    # ------------------------------------------------------------------
    def _crawl_with_scrapy(self, request: CrawlRequest) -> CrawlResult:
        assert self.scrapy_available and CrawlerProcess is not None
        pages: List[CrawledPage] = []
        errors: List[str] = []
        start_time = time.perf_counter()

        authentication = request.authentication
        headers = self._build_headers(request)

        class FitviseSpider(Spider):  # type: ignore[misc]
            name = "fitvise_ingestion"
            custom_settings = {
                "LOG_ENABLED": False,
                "DEPTH_LIMIT": request.max_depth,
                "CONCURRENT_REQUESTS": 8,
            }
            start_urls = request.start_urls

            def start_requests(self):  # type: ignore[override]
                for url in request.start_urls:
                    yield self._make_request(url, 0)

            def _make_request(self, url: str, depth: int):
                meta = {"depth": depth}
                req = Request(url, callback=self.parse, headers=headers, cookies=request.cookies, meta=meta, dont_filter=True)
                if authentication.auth_type == "basic" and authentication.username:
                    req.headers["Authorization"] = self._basic_auth(authentication.username, authentication.password)
                elif authentication.auth_type == "bearer" and authentication.token:
                    req.headers["Authorization"] = f"Bearer {authentication.token}"
                for key, value in authentication.headers.items():
                    req.headers[key] = value
                return req

            def _basic_auth(self, username: str, password: Optional[str]) -> str:
                import base64

                raw = f"{username}:{password or ''}".encode("utf-8")
                return "Basic " + base64.b64encode(raw).decode("ascii")

            def parse(self, response):  # type: ignore[override]
                depth = response.meta.get("depth", 0)
                current_url = response.url
                links = self._extract_links(response)
                pages.append(
                    CrawledPage(
                        url=current_url,
                        status_code=response.status,
                        content=response.text,
                        extracted_links=links,
                        metadata={"depth": str(depth)},
                    )
                )
                if len(pages) >= request.max_pages:
                    return
                for link in links:
                    if not self._should_follow(link, request):
                        continue
                    next_depth = depth + 1
                    if next_depth > request.max_depth:
                        continue
                    yield response.follow(link, callback=self.parse, meta={"depth": next_depth})

            def _extract_links(self, response) -> List[str]:
                links: Set[str] = set()
                if request.follow_css_selectors:
                    for selector in request.follow_css_selectors:
                        links.update(response.css(f"{selector}::attr(href)").getall())
                if request.follow_xpath:
                    for selector in request.follow_xpath:
                        links.update(response.xpath(selector).getall())
                links.update(response.css("a::attr(href)").getall())
                absolute_links = [response.urljoin(url) for url in links]
                return [url for url in absolute_links if self._should_follow(url, request)]

            def _should_follow(self, url: str, req: CrawlRequest) -> bool:
                return WebScrapingFramework._should_follow_static(url, req)

        process = CrawlerProcess()
        try:
            process.crawl(FitviseSpider)
            process.start()
        except Exception as exc:
            errors.append(str(exc))
        elapsed = time.perf_counter() - start_time
        return CrawlResult(pages=pages, elapsed_seconds=elapsed, errors=errors)

    # ------------------------------------------------------------------
    def _crawl_with_httpx(self, request: CrawlRequest) -> CrawlResult:
        if httpx is None:
            raise ModuleNotFoundError("httpx is required for the fallback scraping implementation")
        start_time = time.perf_counter()
        pages: List[CrawledPage] = []
        errors: List[str] = []
        warnings: List[str] = []
        visited: Set[str] = set()
        queue = deque([(url, 0) for url in request.start_urls])
        headers = self._build_headers(request)
        auth = self._build_httpx_auth(request.authentication)
        client = httpx.Client(timeout=request.timeout_seconds)
        session_cookies = request.cookies.copy()
        session_cookies.update(request.authentication.cookies)

        while queue and len(pages) < request.max_pages:
            url, depth = queue.popleft()
            if url in visited or depth > request.max_depth:
                continue
            visited.add(url)
            try:
                response = client.get(url, headers=headers, auth=auth, cookies=session_cookies, follow_redirects=True)
                content = response.text
                links = self._extract_links_from_html(content, url)
                page = CrawledPage(
                    url=url,
                    status_code=response.status_code,
                    content=content,
                    extracted_links=links,
                    metadata={"depth": str(depth)},
                )
                pages.append(page)
                for link in links:
                    if len(pages) + len(queue) >= request.max_pages:
                        break
                    if self._should_follow_static(link, request) and link not in visited:
                        queue.append((link, depth + 1))
            except Exception as exc:  # pragma: no cover - network dependent
                errors.append(f"{url}: {exc}")

        client.close()
        elapsed = time.perf_counter() - start_time
        if not self.scrapy_available:
            warnings.append("Scrapy not available - httpx fallback used")
        return CrawlResult(pages=pages, elapsed_seconds=elapsed, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------
    def _build_headers(self, request: CrawlRequest) -> Dict[str, str]:
        headers = {
            "User-Agent": "FitVise-RAG-Crawler/1.0",
        }
        headers.update(request.headers)
        headers.update(request.authentication.headers)
        if request.authentication.auth_type == "bearer" and request.authentication.token:
            headers.setdefault("Authorization", f"Bearer {request.authentication.token}")
        return headers

    def _build_httpx_auth(self, auth_config: AuthenticationConfig):
        if auth_config.auth_type == "basic" and auth_config.username:
            return (auth_config.username, auth_config.password or "")
        return None

    def _extract_links_from_html(self, html_text: str, base_url: str) -> List[str]:
        if BeautifulSoup is None:
            return []
        soup = BeautifulSoup(html_text, "html.parser")
        links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        return [urljoin(base_url, link) for link in links]

    @staticmethod
    def _should_follow_static(url: str, request: CrawlRequest) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if request.allowed_domains and parsed.hostname:
            if not any(parsed.hostname.endswith(domain) for domain in request.allowed_domains):
                if not request.follow_external_links:
                    return False
        if request.exclude_patterns:
            for pattern in request.exclude_patterns:
                if re.search(pattern, url):
                    return False
        if request.include_patterns:
            return any(re.search(pattern, url) for pattern in request.include_patterns)
        return True
