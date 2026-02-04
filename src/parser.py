"""
HTML Parser Module
Extracts SEO-relevant data from HTML pages including:
- Meta tags (title, description)
- Headings (H1, H2, etc.)
- Body content
- Internal and external links
- Images
"""

from bs4 import BeautifulSoup, Tag
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field
import re


@dataclass
class LinkData:
    """Data structure for extracted links"""
    target_url: str
    anchor_text: str = ""
    context: str = ""
    position: int = 0
    is_dofollow: bool = True
    is_in_nav: bool = False
    is_in_footer: bool = False
    is_in_content: bool = True


@dataclass
class PageData:
    """Data structure for parsed page data"""
    url: str
    domain: str = ""
    meta_title: str = ""
    meta_description: str = ""
    meta_keywords: str = ""
    canonical_url: str = ""
    h1: str = ""
    h2_tags: List[str] = field(default_factory=list)
    h3_tags: List[str] = field(default_factory=list)
    body_content: str = ""
    word_count: int = 0
    internal_links: List[LinkData] = field(default_factory=list)
    external_links: List[LinkData] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    schema_data: List[Dict] = field(default_factory=list)
    language: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'url': self.url,
            'domain': self.domain,
            'meta_title': self.meta_title,
            'meta_description': self.meta_description,
            'meta_keywords': self.meta_keywords,
            'canonical_url': self.canonical_url,
            'h1': self.h1,
            'h2_tags': self.h2_tags,
            'h3_tags': self.h3_tags,
            'body_content': self.body_content,
            'word_count': self.word_count,
            'internal_links_count': len(self.internal_links),
            'external_links_count': len(self.external_links),
            'images_count': len(self.images),
            'language': self.language,
        }


class HTMLParser:
    """
    Parser for extracting SEO-relevant data from HTML.
    
    Usage:
        parser = HTMLParser()
        page_data = parser.parse(html, url)
    """
    
    # Tags to remove from content extraction
    REMOVE_TAGS = [
        'script', 'style', 'noscript', 'iframe', 'svg', 
        'canvas', 'video', 'audio', 'map', 'object', 'embed'
    ]
    
    # Tags that indicate navigation areas
    NAV_TAGS = ['nav', 'header']
    NAV_CLASSES = ['nav', 'menu', 'navigation', 'navbar', 'header', 'topbar', 'sidebar']
    
    # Tags that indicate footer areas
    FOOTER_TAGS = ['footer']
    FOOTER_CLASSES = ['footer', 'foot', 'bottom', 'copyright']
    
    # Tags that indicate main content
    CONTENT_TAGS = ['main', 'article']
    CONTENT_CLASSES = ['content', 'article', 'post', 'entry', 'main', 'body-content']
    
    def __init__(self, parser_type: str = 'html.parser'):
        """
        Initialize the parser.
        
        Args:
            parser_type: BeautifulSoup parser to use ('html.parser', 'lxml', 'html5lib')
        """
        self.parser_type = parser_type
    
    def parse(self, html: str, url: str) -> PageData:
        """
        Parse HTML and extract SEO-relevant data.
        
        Args:
            html: HTML content
            url: URL of the page (for resolving relative URLs)
            
        Returns:
            PageData object with extracted data
        """
        soup = BeautifulSoup(html, self.parser_type)
        domain = urlparse(url).netloc
        
        page_data = PageData(url=url, domain=domain)
        
        # Extract meta data
        page_data.meta_title = self._extract_title(soup)
        page_data.meta_description = self._extract_meta_description(soup)
        page_data.meta_keywords = self._extract_meta_keywords(soup)
        page_data.canonical_url = self._extract_canonical(soup, url)
        page_data.language = self._extract_language(soup)
        
        # Extract headings
        page_data.h1 = self._extract_h1(soup)
        page_data.h2_tags = self._extract_headings(soup, 'h2')
        page_data.h3_tags = self._extract_headings(soup, 'h3')
        
        # Extract body content (after removing unwanted elements)
        page_data.body_content = self._extract_body_content(soup)
        page_data.word_count = len(page_data.body_content.split())
        
        # Extract links
        internal_links, external_links = self._extract_links(soup, url, domain)
        page_data.internal_links = internal_links
        page_data.external_links = external_links
        
        # Extract images
        page_data.images = self._extract_images(soup, url)
        
        # Extract schema/structured data
        page_data.schema_data = self._extract_schema(soup)
        
        return page_data
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Try og:title as fallback
        og_title = soup.find('meta', property='og:title')
        if og_title:
            return og_title.get('content', '')
        
        return ""
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '')
        
        # Try og:description as fallback
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            return og_desc.get('content', '')
        
        return ""
    
    def _extract_meta_keywords(self, soup: BeautifulSoup) -> str:
        """Extract meta keywords"""
        meta_kw = soup.find('meta', attrs={'name': 'keywords'})
        return meta_kw.get('content', '') if meta_kw else ""
    
    def _extract_canonical(self, soup: BeautifulSoup, url: str) -> str:
        """Extract canonical URL"""
        canonical = soup.find('link', rel='canonical')
        if canonical:
            return canonical.get('href', url)
        return url
    
    def _extract_language(self, soup: BeautifulSoup) -> str:
        """Extract page language"""
        html_tag = soup.find('html')
        if html_tag:
            return html_tag.get('lang', '')
        return ""
    
    def _extract_h1(self, soup: BeautifulSoup) -> str:
        """Extract first H1 heading"""
        h1_tag = soup.find('h1')
        return h1_tag.get_text(strip=True) if h1_tag else ""
    
    def _extract_headings(self, soup: BeautifulSoup, tag: str, limit: int = 10) -> List[str]:
        """Extract headings of specific type"""
        headings = []
        for heading in soup.find_all(tag)[:limit]:
            text = heading.get_text(strip=True)
            if text:
                headings.append(text)
        return headings
    
    def _extract_body_content(self, soup: BeautifulSoup, max_length: int = 50000) -> str:
        """
        Extract main body content, excluding navigation and footer.
        """
        # Create a copy to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), self.parser_type)
        
        # Remove unwanted tags
        for tag in self.REMOVE_TAGS:
            for element in soup_copy.find_all(tag):
                element.decompose()
        
        # Remove navigation elements
        for nav in soup_copy.find_all(self.NAV_TAGS):
            nav.decompose()
        
        # Remove footer elements
        for footer in soup_copy.find_all(self.FOOTER_TAGS):
            footer.decompose()
        
        # Remove elements by class
        for class_name in self.NAV_CLASSES + self.FOOTER_CLASSES:
            for element in soup_copy.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()
        
        # Try to find main content area
        main_content = None
        
        for tag in self.CONTENT_TAGS:
            main_content = soup_copy.find(tag)
            if main_content:
                break
        
        if not main_content:
            for class_name in self.CONTENT_CLASSES:
                main_content = soup_copy.find(class_=re.compile(class_name, re.I))
                if main_content:
                    break
        
        # Fall back to body
        if not main_content:
            main_content = soup_copy.find('body')
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup_copy.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text[:max_length]
    
    def _extract_links(
        self, 
        soup: BeautifulSoup, 
        base_url: str, 
        domain: str
    ) -> Tuple[List[LinkData], List[LinkData]]:
        """
        Extract and categorize links.
        
        Returns:
            Tuple of (internal_links, external_links)
        """
        internal_links = []
        external_links = []
        position = 0
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip non-http links
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'data:')):
                continue
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Skip if invalid
            if not parsed.scheme or not parsed.netloc:
                continue
            
            # Get anchor text
            anchor_text = a_tag.get_text(strip=True)
            
            # Get surrounding context
            context = self._get_link_context(a_tag)
            
            # Check if dofollow
            rel = a_tag.get('rel', [])
            if isinstance(rel, str):
                rel = [rel]
            is_dofollow = 'nofollow' not in rel
            
            # Determine link location
            is_in_nav = self._is_in_nav(a_tag)
            is_in_footer = self._is_in_footer(a_tag)
            is_in_content = not is_in_nav and not is_in_footer
            
            link_data = LinkData(
                target_url=full_url,
                anchor_text=anchor_text,
                context=context,
                position=position,
                is_dofollow=is_dofollow,
                is_in_nav=is_in_nav,
                is_in_footer=is_in_footer,
                is_in_content=is_in_content
            )
            
            # Categorize as internal or external
            if parsed.netloc == domain or parsed.netloc.endswith('.' + domain):
                internal_links.append(link_data)
            else:
                external_links.append(link_data)
            
            position += 1
        
        return internal_links, external_links
    
    def _get_link_context(self, a_tag: Tag, max_length: int = 200) -> str:
        """Get surrounding text context for a link"""
        parent = a_tag.find_parent(['p', 'li', 'div', 'td', 'span', 'article'])
        if parent:
            context = parent.get_text(strip=True)
            return context[:max_length]
        return ""
    
    def _is_in_nav(self, element: Tag) -> bool:
        """Check if element is within navigation"""
        for parent in element.parents:
            if parent.name in self.NAV_TAGS:
                return True
            parent_class = parent.get('class', [])
            if isinstance(parent_class, list):
                for class_name in parent_class:
                    if any(nav in class_name.lower() for nav in self.NAV_CLASSES):
                        return True
        return False
    
    def _is_in_footer(self, element: Tag) -> bool:
        """Check if element is within footer"""
        for parent in element.parents:
            if parent.name in self.FOOTER_TAGS:
                return True
            parent_class = parent.get('class', [])
            if isinstance(parent_class, list):
                for class_name in parent_class:
                    if any(foot in class_name.lower() for foot in self.FOOTER_CLASSES):
                        return True
        return False
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract image data"""
        images = []
        
        for img in soup.find_all('img', src=True)[:100]:  # Limit to 100 images
            src = img['src']
            full_url = urljoin(base_url, src)
            
            images.append({
                'src': full_url,
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
            })
        
        return images
    
    def _extract_schema(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract JSON-LD schema data"""
        schemas = []
        
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                import json
                data = json.loads(script.string)
                schemas.append(data)
            except:
                pass
        
        return schemas


def parse_html(html: str, url: str) -> PageData:
    """
    Convenience function to parse HTML.
    
    Args:
        html: HTML content
        url: URL of the page
        
    Returns:
        PageData object
    """
    parser = HTMLParser()
    return parser.parse(html, url)


def extract_text_for_embedding(page_data: PageData, max_length: int = 2000) -> str:
    """
    Extract text suitable for generating embeddings.
    
    Args:
        page_data: Parsed page data
        max_length: Maximum text length
        
    Returns:
        Combined text string
    """
    parts = []
    
    # Add title (weighted more by repetition)
    if page_data.meta_title:
        parts.append(page_data.meta_title)
    
    # Add H1
    if page_data.h1:
        parts.append(page_data.h1)
    
    # Add meta description
    if page_data.meta_description:
        parts.append(page_data.meta_description)
    
    # Add H2s
    for h2 in page_data.h2_tags[:5]:
        parts.append(h2)
    
    # Add body content
    if page_data.body_content:
        parts.append(page_data.body_content)
    
    text = ' '.join(parts)
    return text[:max_length]


# Example usage
if __name__ == "__main__":
    sample_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Sample Page Title</title>
        <meta name="description" content="This is a sample description">
    </head>
    <body>
        <nav>
            <a href="/nav-link">Nav Link</a>
        </nav>
        <main>
            <h1>Main Heading</h1>
            <p>This is the main content with an <a href="/internal">internal link</a>.</p>
            <p>And here is an <a href="https://external.com">external link</a>.</p>
        </main>
        <footer>
            <a href="/footer-link">Footer Link</a>
        </footer>
    </body>
    </html>
    """
    
    page_data = parse_html(sample_html, "https://example.com/page")
    print(f"Title: {page_data.meta_title}")
    print(f"H1: {page_data.h1}")
    print(f"Internal links: {len(page_data.internal_links)}")
    print(f"External links: {len(page_data.external_links)}")
    print(f"Word count: {page_data.word_count}")
