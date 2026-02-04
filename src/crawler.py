"""
Async Web Crawler Module
High-performance asynchronous web crawler using aiohttp
Supports Googlebot user agent for bypassing certain blocks
"""

import asyncio
import aiohttp
from aiohttp import ClientTimeout, ClientSession, TCPConnector
from typing import List, Dict, Any, Optional, Callable, Set
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default user agents
USER_AGENTS = {
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'googlebot_mobile': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'firefox': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
}


@dataclass
class CrawlResult:
    """Result of crawling a single URL"""
    url: str
    final_url: str = ""
    status: str = "pending"  # pending, success, error
    http_status: int = 0
    html: str = ""
    content_type: str = ""
    error: str = ""
    crawled_at: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0


@dataclass
class CrawlerConfig:
    """Configuration for the crawler"""
    user_agent: str = "googlebot"
    max_concurrent: int = 10
    timeout: int = 30
    delay: float = 0.5
    max_retries: int = 3
    retry_delay: float = 1.0
    follow_redirects: bool = True
    max_redirects: int = 5
    verify_ssl: bool = True
    max_content_length: int = 10 * 1024 * 1024  # 10MB


class AsyncCrawler:
    """
    Async web crawler with rate limiting and concurrent request management.
    
    Usage:
        crawler = AsyncCrawler(user_agent='googlebot', max_concurrent=10)
        results = await crawler.crawl_urls(urls)
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None, **kwargs):
        """
        Initialize the crawler.
        
        Args:
            config: CrawlerConfig object
            **kwargs: Override config parameters
        """
        self.config = config or CrawlerConfig(**kwargs)
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.session: Optional[ClientSession] = None
        self._seen_urls: Set[str] = set()
        
        # Get user agent string
        self.user_agent = USER_AGENTS.get(
            self.config.user_agent, 
            self.config.user_agent  # Use as-is if custom
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def _fetch_url(self, url: str, retry_count: int = 0) -> CrawlResult:
        """
        Fetch a single URL with retry logic.
        
        Args:
            url: URL to fetch
            retry_count: Current retry attempt
            
        Returns:
            CrawlResult object
        """
        result = CrawlResult(url=url)
        start_time = datetime.now()
        
        try:
            # Rate limiting via semaphore
            async with self.semaphore:
                # Add delay between requests
                if self.config.delay > 0:
                    await asyncio.sleep(self.config.delay)
                
                timeout = ClientTimeout(total=self.config.timeout)
                
                async with self.session.get(
                    url,
                    timeout=timeout,
                    allow_redirects=self.config.follow_redirects,
                    max_redirects=self.config.max_redirects,
                    ssl=self.config.verify_ssl
                ) as response:
                    result.http_status = response.status
                    result.final_url = str(response.url)
                    result.content_type = response.headers.get('Content-Type', '')
                    
                    if response.status == 200:
                        # Check content type
                        if 'text/html' in result.content_type or 'application/xhtml' in result.content_type:
                            # Check content length
                            content_length = response.headers.get('Content-Length')
                            if content_length and int(content_length) > self.config.max_content_length:
                                result.status = "error"
                                result.error = "Content too large"
                            else:
                                result.html = await response.text()
                                result.status = "success"
                        else:
                            result.status = "error"
                            result.error = f"Non-HTML content: {result.content_type}"
                    else:
                        result.status = "error"
                        result.error = f"HTTP {response.status}"
                        
        except asyncio.TimeoutError:
            result.status = "error"
            result.error = "Timeout"
            
            # Retry on timeout
            if retry_count < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay * (retry_count + 1))
                return await self._fetch_url(url, retry_count + 1)
                
        except aiohttp.ClientError as e:
            result.status = "error"
            result.error = f"Client error: {str(e)}"
            
            # Retry on certain errors
            if retry_count < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay * (retry_count + 1))
                return await self._fetch_url(url, retry_count + 1)
                
        except Exception as e:
            result.status = "error"
            result.error = f"Unexpected error: {str(e)}"
        
        result.crawled_at = datetime.now()
        result.response_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    async def crawl_urls(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[CrawlResult]:
        """
        Crawl multiple URLs concurrently.
        
        Args:
            urls: List of URLs to crawl
            progress_callback: Optional callback function(current, total)
            
        Returns:
            List of CrawlResult objects
        """
        # Initialize semaphore
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create connector with connection limits
        connector = TCPConnector(
            limit=self.config.max_concurrent,
            limit_per_host=5,
            ttl_dns_cache=300
        )
        
        # Create session
        async with ClientSession(
            headers=self._get_headers(),
            connector=connector
        ) as self.session:
            
            # Create tasks
            tasks = [self._fetch_url(url) for url in urls]
            results = []
            
            # Process as completed
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(urls))
            
            return results
    
    async def crawl_url(self, url: str) -> CrawlResult:
        """
        Crawl a single URL.
        
        Args:
            url: URL to crawl
            
        Returns:
            CrawlResult object
        """
        results = await self.crawl_urls([url])
        return results[0] if results else CrawlResult(url=url, status="error", error="No result")
    
    def crawl_urls_sync(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[CrawlResult]:
        """
        Synchronous wrapper for crawl_urls.
        
        Args:
            urls: List of URLs to crawl
            progress_callback: Optional callback function
            
        Returns:
            List of CrawlResult objects
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self.crawl_urls(urls, progress_callback)
            )
        finally:
            loop.close()


class SitemapCrawler:
    """
    Crawler that discovers URLs from sitemaps.
    """
    
    def __init__(self, crawler: Optional[AsyncCrawler] = None):
        self.crawler = crawler or AsyncCrawler()
    
    async def get_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """
        Extract URLs from a sitemap.
        
        Args:
            sitemap_url: URL of the sitemap
            
        Returns:
            List of URLs found in sitemap
        """
        result = await self.crawler.crawl_url(sitemap_url)
        
        if result.status != "success":
            logger.error(f"Failed to fetch sitemap: {result.error}")
            return []
        
        urls = []
        
        # Parse sitemap XML
        # Handle both regular sitemaps and sitemap indexes
        loc_pattern = re.compile(r'<loc>\s*(.*?)\s*</loc>', re.IGNORECASE)
        
        for match in loc_pattern.finditer(result.html):
            url = match.group(1).strip()
            
            # Check if it's a nested sitemap
            if url.endswith('.xml') or 'sitemap' in url.lower():
                # Recursively fetch nested sitemap
                nested_urls = await self.get_sitemap_urls(url)
                urls.extend(nested_urls)
            else:
                urls.append(url)
        
        return urls
    
    async def discover_sitemaps(self, domain: str) -> List[str]:
        """
        Try to discover sitemaps for a domain.
        
        Args:
            domain: Domain to check (e.g., "example.com")
            
        Returns:
            List of sitemap URLs found
        """
        # Ensure domain has protocol
        if not domain.startswith('http'):
            domain = f'https://{domain}'
        
        # Common sitemap locations
        sitemap_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap/',
            '/sitemap/sitemap.xml',
            '/sitemaps/sitemap.xml',
        ]
        
        sitemaps = []
        
        for path in sitemap_paths:
            url = urljoin(domain, path)
            result = await self.crawler.crawl_url(url)
            
            if result.status == "success" and 'xml' in result.content_type.lower():
                sitemaps.append(url)
        
        # Try robots.txt
        robots_url = urljoin(domain, '/robots.txt')
        result = await self.crawler.crawl_url(robots_url)
        
        if result.status == "success":
            # Extract sitemap URLs from robots.txt
            sitemap_pattern = re.compile(r'Sitemap:\s*(\S+)', re.IGNORECASE)
            for match in sitemap_pattern.finditer(result.html):
                sitemap_url = match.group(1).strip()
                if sitemap_url not in sitemaps:
                    sitemaps.append(sitemap_url)
        
        return sitemaps


def run_crawler(
    urls: List[str],
    user_agent: str = "googlebot",
    max_concurrent: int = 10,
    timeout: int = 30,
    delay: float = 0.5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[CrawlResult]:
    """
    Convenience function to run the crawler synchronously.
    
    Args:
        urls: List of URLs to crawl
        user_agent: User agent name or string
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout in seconds
        delay: Delay between requests in seconds
        progress_callback: Optional progress callback
        
    Returns:
        List of CrawlResult objects
    """
    config = CrawlerConfig(
        user_agent=user_agent,
        max_concurrent=max_concurrent,
        timeout=timeout,
        delay=delay
    )
    
    crawler = AsyncCrawler(config)
    return crawler.crawl_urls_sync(urls, progress_callback)


# Example usage
if __name__ == "__main__":
    # Test URLs
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]
    
    def progress(current, total):
        print(f"Progress: {current}/{total}")
    
    results = run_crawler(
        test_urls,
        user_agent='googlebot',
        max_concurrent=5,
        progress_callback=progress
    )
    
    for result in results:
        print(f"{result.url}: {result.status} ({result.http_status})")
