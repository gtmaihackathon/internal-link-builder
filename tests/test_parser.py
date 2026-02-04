"""
Tests for the HTML Parser Module
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import (
    HTMLParser,
    PageData,
    LinkData,
    parse_html,
    extract_text_for_embedding
)


class TestHTMLParser:
    """Test HTMLParser class"""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance"""
        return HTMLParser()
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML for testing"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Test Page Title - Site Name</title>
            <meta name="description" content="This is a test meta description">
            <meta name="keywords" content="test, keywords, seo">
            <link rel="canonical" href="https://example.com/test-page">
        </head>
        <body>
            <nav>
                <a href="/nav-link-1">Nav Link 1</a>
                <a href="/nav-link-2">Nav Link 2</a>
            </nav>
            <main>
                <article>
                    <h1>Main Heading H1</h1>
                    <h2>Subheading H2 One</h2>
                    <p>This is the first paragraph with some content.</p>
                    <p>Second paragraph with an <a href="/internal-link">internal link</a>.</p>
                    <h2>Subheading H2 Two</h2>
                    <p>More content with <a href="https://external.com" rel="nofollow">external link</a>.</p>
                    <img src="/image.jpg" alt="Test image" title="Image title">
                </article>
            </main>
            <footer>
                <a href="/footer-link">Footer Link</a>
            </footer>
        </body>
        </html>
        """
    
    def test_parse_title(self, parser, sample_html):
        """Test title extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        assert result.meta_title == "Test Page Title - Site Name"
    
    def test_parse_meta_description(self, parser, sample_html):
        """Test meta description extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        assert result.meta_description == "This is a test meta description"
    
    def test_parse_meta_keywords(self, parser, sample_html):
        """Test meta keywords extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        assert result.meta_keywords == "test, keywords, seo"
    
    def test_parse_canonical(self, parser, sample_html):
        """Test canonical URL extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        assert result.canonical_url == "https://example.com/test-page"
    
    def test_parse_h1(self, parser, sample_html):
        """Test H1 extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        assert result.h1 == "Main Heading H1"
    
    def test_parse_h2_tags(self, parser, sample_html):
        """Test H2 extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        assert len(result.h2_tags) == 2
        assert "Subheading H2 One" in result.h2_tags
        assert "Subheading H2 Two" in result.h2_tags
    
    def test_parse_internal_links(self, parser, sample_html):
        """Test internal link extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        
        # Should find internal links (excluding nav/footer based on position)
        internal_urls = [link.target_url for link in result.internal_links]
        
        # Check that internal links are properly resolved
        assert any("example.com" in url for url in internal_urls)
    
    def test_parse_external_links(self, parser, sample_html):
        """Test external link extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        
        external_urls = [link.target_url for link in result.external_links]
        assert any("external.com" in url for url in external_urls)
    
    def test_parse_nofollow_detection(self, parser, sample_html):
        """Test nofollow attribute detection"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        
        # Find the external link
        external_links = [link for link in result.external_links if "external.com" in link.target_url]
        
        if external_links:
            # The external link has rel="nofollow"
            assert external_links[0].is_dofollow == False
    
    def test_parse_images(self, parser, sample_html):
        """Test image extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        
        assert len(result.images) >= 1
        image = result.images[0]
        assert image['alt'] == "Test image"
        assert image['title'] == "Image title"
    
    def test_parse_body_content(self, parser, sample_html):
        """Test body content extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        
        assert len(result.body_content) > 0
        assert "first paragraph" in result.body_content
        assert result.word_count > 0
    
    def test_parse_language(self, parser, sample_html):
        """Test language extraction"""
        result = parser.parse(sample_html, "https://example.com/test-page")
        assert result.language == "en"


class TestPageData:
    """Test PageData dataclass"""
    
    def test_page_data_creation(self):
        """Test PageData creation"""
        page = PageData(url="https://example.com")
        
        assert page.url == "https://example.com"
        assert page.meta_title == ""
        assert page.word_count == 0
        assert page.internal_links == []
    
    def test_page_data_to_dict(self):
        """Test PageData to_dict method"""
        page = PageData(
            url="https://example.com",
            meta_title="Test Title",
            word_count=100
        )
        
        data = page.to_dict()
        
        assert data['url'] == "https://example.com"
        assert data['meta_title'] == "Test Title"
        assert data['word_count'] == 100


class TestLinkData:
    """Test LinkData dataclass"""
    
    def test_link_data_creation(self):
        """Test LinkData creation"""
        link = LinkData(target_url="https://example.com/page")
        
        assert link.target_url == "https://example.com/page"
        assert link.anchor_text == ""
        assert link.is_dofollow == True
        assert link.is_in_content == True
    
    def test_link_data_full(self):
        """Test LinkData with all fields"""
        link = LinkData(
            target_url="https://example.com/page",
            anchor_text="Click here",
            context="Some surrounding text",
            position=5,
            is_dofollow=False,
            is_in_nav=True,
            is_in_content=False
        )
        
        assert link.anchor_text == "Click here"
        assert link.is_dofollow == False
        assert link.is_in_nav == True


class TestParseHtmlFunction:
    """Test the convenience parse_html function"""
    
    def test_parse_html_simple(self):
        """Test parse_html with simple HTML"""
        html = "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        result = parse_html(html, "https://example.com")
        
        assert result.meta_title == "Test"
        assert result.h1 == "Hello"
    
    def test_parse_html_empty(self):
        """Test parse_html with empty HTML"""
        result = parse_html("", "https://example.com")
        
        assert result.url == "https://example.com"
        assert result.meta_title == ""


class TestExtractTextForEmbedding:
    """Test the extract_text_for_embedding function"""
    
    def test_extract_text_basic(self):
        """Test basic text extraction"""
        page = PageData(
            url="https://example.com",
            meta_title="Test Title",
            h1="Main Heading",
            body_content="This is the body content."
        )
        
        text = extract_text_for_embedding(page)
        
        assert "Test Title" in text
        assert "Main Heading" in text
        assert "body content" in text
    
    def test_extract_text_max_length(self):
        """Test text extraction respects max length"""
        long_content = "word " * 1000
        page = PageData(
            url="https://example.com",
            body_content=long_content
        )
        
        text = extract_text_for_embedding(page, max_length=100)
        
        assert len(text) <= 100
    
    def test_extract_text_empty_page(self):
        """Test text extraction with empty page"""
        page = PageData(url="https://example.com")
        text = extract_text_for_embedding(page)
        
        # Should not raise an error
        assert isinstance(text, str)


class TestEdgeCases:
    """Test edge cases and malformed HTML"""
    
    @pytest.fixture
    def parser(self):
        return HTMLParser()
    
    def test_missing_head(self, parser):
        """Test HTML without head tag"""
        html = "<html><body><h1>Test</h1></body></html>"
        result = parser.parse(html, "https://example.com")
        
        assert result.h1 == "Test"
        assert result.meta_title == ""
    
    def test_missing_body(self, parser):
        """Test HTML without body tag"""
        html = "<html><head><title>Test</title></head></html>"
        result = parser.parse(html, "https://example.com")
        
        assert result.meta_title == "Test"
    
    def test_malformed_html(self, parser):
        """Test malformed HTML"""
        html = "<html><head><title>Test<body><h1>Hello</h1>"
        result = parser.parse(html, "https://example.com")
        
        # Should not raise an error
        assert result.url == "https://example.com"
    
    def test_unicode_content(self, parser):
        """Test HTML with unicode content"""
        html = """
        <html>
        <head><title>Test 日本語</title></head>
        <body><h1>Héllo Wörld 中文</h1></body>
        </html>
        """
        result = parser.parse(html, "https://example.com")
        
        assert "日本語" in result.meta_title
        assert "中文" in result.h1
    
    def test_relative_url_resolution(self, parser):
        """Test relative URL resolution"""
        html = """
        <html><body>
            <a href="/page1">Link 1</a>
            <a href="page2">Link 2</a>
            <a href="../page3">Link 3</a>
        </body></html>
        """
        result = parser.parse(html, "https://example.com/dir/current")
        
        internal_urls = [link.target_url for link in result.internal_links]
        
        # All should be resolved to absolute URLs
        for url in internal_urls:
            assert url.startswith("https://")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
