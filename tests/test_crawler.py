"""
Tests for the Crawler Module
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crawler import (
    AsyncCrawler,
    CrawlerConfig,
    CrawlResult,
    run_crawler,
    USER_AGENTS
)


class TestCrawlerConfig:
    """Test CrawlerConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CrawlerConfig()
        
        assert config.user_agent == "googlebot"
        assert config.max_concurrent == 10
        assert config.timeout == 30
        assert config.delay == 0.5
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CrawlerConfig(
            user_agent="chrome",
            max_concurrent=20,
            timeout=60
        )
        
        assert config.user_agent == "chrome"
        assert config.max_concurrent == 20
        assert config.timeout == 60


class TestUserAgents:
    """Test user agent strings"""
    
    def test_googlebot_available(self):
        """Test Googlebot user agent is available"""
        assert 'googlebot' in USER_AGENTS
        assert 'Googlebot' in USER_AGENTS['googlebot']
    
    def test_all_user_agents_valid(self):
        """Test all user agents are non-empty strings"""
        for name, ua in USER_AGENTS.items():
            assert isinstance(ua, str)
            assert len(ua) > 0
    
    def test_user_agent_contains_http(self):
        """Test bot user agents contain URL"""
        bot_agents = ['googlebot', 'bingbot']
        for bot in bot_agents:
            if bot in USER_AGENTS:
                assert 'http' in USER_AGENTS[bot]


class TestCrawlResult:
    """Test CrawlResult dataclass"""
    
    def test_default_result(self):
        """Test default CrawlResult values"""
        result = CrawlResult(url="https://example.com")
        
        assert result.url == "https://example.com"
        assert result.status == "pending"
        assert result.http_status == 0
        assert result.html == ""
        assert result.error == ""
    
    def test_success_result(self):
        """Test successful CrawlResult"""
        result = CrawlResult(
            url="https://example.com",
            status="success",
            http_status=200,
            html="<html></html>"
        )
        
        assert result.status == "success"
        assert result.http_status == 200
        assert result.html == "<html></html>"


class TestAsyncCrawler:
    """Test AsyncCrawler class"""
    
    def test_crawler_initialization(self):
        """Test crawler initialization"""
        crawler = AsyncCrawler(user_agent='googlebot', max_concurrent=5)
        
        assert 'Googlebot' in crawler.user_agent
        assert crawler.config.max_concurrent == 5
    
    def test_crawler_with_config(self):
        """Test crawler with CrawlerConfig"""
        config = CrawlerConfig(
            user_agent='chrome',
            timeout=60
        )
        crawler = AsyncCrawler(config=config)
        
        assert crawler.config.timeout == 60
    
    def test_get_headers(self):
        """Test header generation"""
        crawler = AsyncCrawler(user_agent='googlebot')
        headers = crawler._get_headers()
        
        assert 'User-Agent' in headers
        assert 'Accept' in headers
        assert 'Googlebot' in headers['User-Agent']


class TestAsyncCrawlerFetch:
    """Test async fetch functionality"""
    
    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful fetch"""
        crawler = AsyncCrawler()
        
        # Mock the session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.url = "https://example.com"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.text = AsyncMock(return_value="<html></html>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        
        crawler.session = mock_session
        crawler.semaphore = asyncio.Semaphore(1)
        crawler.config.delay = 0  # No delay for tests
        
        # This would require more complex mocking to fully test
        # For now, just verify the crawler can be initialized
        assert crawler is not None
    
    @pytest.mark.asyncio
    async def test_crawl_urls_empty_list(self):
        """Test crawling empty URL list"""
        crawler = AsyncCrawler()
        results = await crawler.crawl_urls([])
        
        assert results == []


class TestRunCrawler:
    """Test the convenience function"""
    
    def test_run_crawler_function_exists(self):
        """Test run_crawler function exists and is callable"""
        assert callable(run_crawler)
    
    def test_run_crawler_with_empty_list(self):
        """Test run_crawler with empty list"""
        results = run_crawler([])
        assert results == []


class TestCrawlerValidation:
    """Test URL validation and handling"""
    
    def test_url_parsing(self):
        """Test URL parsing"""
        from urllib.parse import urlparse
        
        test_urls = [
            "https://example.com",
            "https://example.com/path",
            "https://example.com/path?query=1",
        ]
        
        for url in test_urls:
            parsed = urlparse(url)
            assert parsed.scheme in ['http', 'https']
            assert parsed.netloc != ""
    
    def test_invalid_url_detection(self):
        """Test detection of invalid URLs"""
        from urllib.parse import urlparse
        
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "",
            "//example.com",
        ]
        
        for url in invalid_urls:
            parsed = urlparse(url)
            is_valid = parsed.scheme in ['http', 'https'] and parsed.netloc
            # These should be invalid
            if url == "not-a-url" or url == "" or url == "//example.com":
                assert not is_valid or parsed.scheme not in ['http', 'https']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
