"""
Link Suggestion Engine Module
Generates intelligent internal link suggestions based on semantic similarity
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LinkSuggestion:
    """A single link suggestion"""
    source_url: str
    target_url: str
    suggested_anchor: str = ""
    suggested_context: str = ""
    insertion_point: str = ""
    relevance_score: float = 0.0
    action: str = "add"  # add, update, remove, keep
    priority: str = "medium"  # high, medium, low
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_url': self.source_url,
            'target_url': self.target_url,
            'suggested_anchor': self.suggested_anchor,
            'suggested_context': self.suggested_context,
            'insertion_point': self.insertion_point,
            'relevance_score': self.relevance_score,
            'action': self.action,
            'priority': self.priority,
            'reason': self.reason
        }


@dataclass
class SuggestionConfig:
    """Configuration for suggestion generation"""
    min_similarity_threshold: float = 0.3
    high_priority_threshold: float = 0.7
    medium_priority_threshold: float = 0.5
    top_k_per_page: int = 5
    max_total_suggestions: int = 1000
    exclude_self_links: bool = True
    exclude_existing_links: bool = True
    prefer_content_links: bool = True  # Prefer links in main content vs nav/footer


class SuggestionEngine:
    """
    Engine for generating internal link suggestions.
    
    Usage:
        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(pages, embeddings, existing_links)
    """
    
    def __init__(self, config: Optional[SuggestionConfig] = None):
        """
        Initialize the suggestion engine.
        
        Args:
            config: SuggestionConfig object
        """
        self.config = config or SuggestionConfig()
    
    def generate_suggestions(
        self,
        pages: List[Dict[str, Any]],
        embeddings: np.ndarray,
        existing_links: Optional[Set[Tuple[str, str]]] = None,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> List[LinkSuggestion]:
        """
        Generate link suggestions based on semantic similarity.
        
        Args:
            pages: List of page dictionaries with 'url', 'meta_title', 'h1', etc.
            embeddings: Numpy array of page embeddings
            existing_links: Set of existing (source_url, target_url) pairs
            similarity_matrix: Pre-computed similarity matrix (optional)
            
        Returns:
            List of LinkSuggestion objects
        """
        if existing_links is None:
            existing_links = set()
        
        # Calculate similarity matrix if not provided
        if similarity_matrix is None:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
        
        logger.info(f"Generating suggestions for {len(pages)} pages")
        
        suggestions = []
        
        for i, source_page in enumerate(pages):
            source_url = source_page.get('url', '')
            
            if not source_url:
                continue
            
            # Get top similar pages
            page_suggestions = self._get_suggestions_for_page(
                i,
                source_page,
                pages,
                similarity_matrix[i],
                existing_links
            )
            
            suggestions.extend(page_suggestions)
            
            # Limit total suggestions
            if len(suggestions) >= self.config.max_total_suggestions:
                break
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Generated {len(suggestions)} suggestions")
        
        return suggestions[:self.config.max_total_suggestions]
    
    def _get_suggestions_for_page(
        self,
        page_index: int,
        source_page: Dict[str, Any],
        all_pages: List[Dict[str, Any]],
        similarities: np.ndarray,
        existing_links: Set[Tuple[str, str]]
    ) -> List[LinkSuggestion]:
        """Get suggestions for a single page"""
        source_url = source_page.get('url', '')
        suggestions = []
        
        # Get indices of top similar pages
        top_indices = np.argsort(similarities)[::-1]
        
        count = 0
        for target_index in top_indices:
            if count >= self.config.top_k_per_page:
                break
            
            # Skip self
            if self.config.exclude_self_links and target_index == page_index:
                continue
            
            target_page = all_pages[target_index]
            target_url = target_page.get('url', '')
            
            if not target_url:
                continue
            
            # Skip existing links
            if self.config.exclude_existing_links and (source_url, target_url) in existing_links:
                continue
            
            similarity = similarities[target_index]
            
            # Check threshold
            if similarity < self.config.min_similarity_threshold:
                break  # Sorted, so all remaining will be below threshold
            
            # Generate suggestion
            suggestion = self._create_suggestion(
                source_page, 
                target_page, 
                similarity
            )
            
            suggestions.append(suggestion)
            count += 1
        
        return suggestions
    
    def _create_suggestion(
        self,
        source_page: Dict[str, Any],
        target_page: Dict[str, Any],
        similarity: float
    ) -> LinkSuggestion:
        """Create a link suggestion"""
        source_url = source_page.get('url', '')
        target_url = target_page.get('url', '')
        
        # Generate anchor text
        anchor_text = self._generate_anchor_text(target_page)
        
        # Find insertion point in source content
        insertion_point = self._find_insertion_point(
            source_page.get('body_content', ''),
            target_page
        )
        
        # Determine priority
        priority = self._determine_priority(similarity)
        
        # Generate reason
        reason = f"Semantic similarity: {similarity:.2f}"
        
        return LinkSuggestion(
            source_url=source_url,
            target_url=target_url,
            suggested_anchor=anchor_text,
            suggested_context=insertion_point,
            insertion_point=insertion_point[:200] if insertion_point else "",
            relevance_score=float(similarity),
            action="add",
            priority=priority,
            reason=reason
        )
    
    def _generate_anchor_text(self, target_page: Dict[str, Any]) -> str:
        """Generate suggested anchor text for target page"""
        # Priority: H1 > Title > URL slug
        h1 = target_page.get('h1', '')
        title = target_page.get('meta_title', '')
        url = target_page.get('url', '')
        
        if h1 and len(h1) <= 60:
            return h1
        
        if title:
            # Remove site name suffix if present
            title = re.sub(r'\s*[\|\-–—]\s*[^|\-–—]+$', '', title)
            if len(title) <= 60:
                return title
        
        # Extract from URL
        if url:
            slug = url.rstrip('/').split('/')[-1]
            # Convert slug to readable text
            slug = slug.replace('-', ' ').replace('_', ' ')
            slug = re.sub(r'\.[a-z]+$', '', slug)  # Remove extension
            return slug.title()
        
        return "Learn more"
    
    def _find_insertion_point(
        self,
        source_content: str,
        target_page: Dict[str, Any]
    ) -> str:
        """Find a good place in source content to insert the link"""
        if not source_content:
            return ""
        
        # Get keywords from target page
        keywords = self._extract_keywords(target_page)
        
        if not keywords:
            return ""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', source_content)
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            
            # Score based on keyword presence
            score = sum(1 for kw in keywords if kw.lower() in sentence.lower())
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence
    
    def _extract_keywords(self, page: Dict[str, Any], max_keywords: int = 5) -> List[str]:
        """Extract key terms from page"""
        keywords = []
        
        # From H1
        h1 = page.get('h1', '')
        if h1:
            words = re.findall(r'\b\w{4,}\b', h1.lower())
            keywords.extend(words[:2])
        
        # From title
        title = page.get('meta_title', '')
        if title:
            words = re.findall(r'\b\w{4,}\b', title.lower())
            keywords.extend(words[:2])
        
        # Remove duplicates and common words
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'where', 'which', 'their', 'about', 'would', 'could', 'should', 'there', 'these', 'those', 'your', 'more', 'some', 'into', 'them', 'other'}
        keywords = [kw for kw in keywords if kw not in stopwords]
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates, preserve order
        
        return keywords[:max_keywords]
    
    def _determine_priority(self, similarity: float) -> str:
        """Determine suggestion priority based on similarity"""
        if similarity >= self.config.high_priority_threshold:
            return "high"
        elif similarity >= self.config.medium_priority_threshold:
            return "medium"
        else:
            return "low"
    
    def analyze_existing_links(
        self,
        pages: List[Dict[str, Any]],
        embeddings: np.ndarray,
        existing_links: List[Dict[str, Any]],
        similarity_matrix: Optional[np.ndarray] = None
    ) -> List[LinkSuggestion]:
        """
        Analyze existing links and suggest updates/removals.
        
        Args:
            pages: List of page dictionaries
            embeddings: Page embeddings
            existing_links: List of existing link dictionaries
            similarity_matrix: Pre-computed similarity matrix
            
        Returns:
            List of suggestions for existing links
        """
        if similarity_matrix is None:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
        
        # Build URL to index mapping
        url_to_idx = {p.get('url'): i for i, p in enumerate(pages)}
        
        suggestions = []
        
        for link in existing_links:
            source_url = link.get('source_url', '')
            target_url = link.get('target_url', '')
            anchor_text = link.get('anchor_text', '')
            
            source_idx = url_to_idx.get(source_url)
            target_idx = url_to_idx.get(target_url)
            
            if source_idx is None or target_idx is None:
                continue
            
            similarity = similarity_matrix[source_idx][target_idx]
            
            # Analyze the link
            if similarity < 0.2:
                # Low relevance - suggest removal
                suggestions.append(LinkSuggestion(
                    source_url=source_url,
                    target_url=target_url,
                    suggested_anchor=anchor_text,
                    relevance_score=float(similarity),
                    action="remove",
                    priority="medium",
                    reason=f"Low semantic relevance ({similarity:.2f})"
                ))
            elif similarity >= 0.5:
                # Good link - keep it
                target_page = pages[target_idx]
                better_anchor = self._generate_anchor_text(target_page)
                
                if anchor_text and better_anchor.lower() != anchor_text.lower():
                    # Suggest anchor update
                    suggestions.append(LinkSuggestion(
                        source_url=source_url,
                        target_url=target_url,
                        suggested_anchor=better_anchor,
                        relevance_score=float(similarity),
                        action="update",
                        priority="low",
                        reason=f"Consider better anchor text"
                    ))
                else:
                    # Keep as-is
                    suggestions.append(LinkSuggestion(
                        source_url=source_url,
                        target_url=target_url,
                        suggested_anchor=anchor_text,
                        relevance_score=float(similarity),
                        action="keep",
                        priority="low",
                        reason=f"Good semantic relevance ({similarity:.2f})"
                    ))
        
        return suggestions
    
    def find_orphan_pages(
        self,
        pages: List[Dict[str, Any]],
        existing_links: Set[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Find pages with no incoming internal links.
        
        Args:
            pages: List of page dictionaries
            existing_links: Set of (source_url, target_url) pairs
            
        Returns:
            List of orphan page dictionaries
        """
        # Get all URLs that receive links
        linked_urls = {target for _, target in existing_links}
        
        # Find pages not in linked_urls
        orphans = []
        for page in pages:
            url = page.get('url', '')
            if url and url not in linked_urls:
                orphans.append(page)
        
        logger.info(f"Found {len(orphans)} orphan pages")
        return orphans
    
    def find_cannibalization(
        self,
        pages: List[Dict[str, Any]],
        similarity_matrix: np.ndarray,
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Find potentially cannibalizing pages (too similar).
        
        Args:
            pages: List of page dictionaries
            similarity_matrix: Cosine similarity matrix
            threshold: Similarity threshold for cannibalization
            
        Returns:
            List of cannibalization pairs
        """
        cannibalization = []
        n = len(pages)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= threshold:
                    cannibalization.append({
                        'url1': pages[i].get('url', ''),
                        'url2': pages[j].get('url', ''),
                        'title1': pages[i].get('meta_title', ''),
                        'title2': pages[j].get('meta_title', ''),
                        'similarity': float(similarity_matrix[i][j])
                    })
        
        # Sort by similarity descending
        cannibalization.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(cannibalization)} potential cannibalization pairs")
        return cannibalization


def generate_suggestions(
    pages: List[Dict[str, Any]],
    embeddings: np.ndarray,
    existing_links: Optional[Set[Tuple[str, str]]] = None,
    top_k: int = 5,
    min_threshold: float = 0.3
) -> List[LinkSuggestion]:
    """
    Convenience function to generate suggestions.
    
    Args:
        pages: List of page dictionaries
        embeddings: Page embeddings
        existing_links: Existing link pairs
        top_k: Suggestions per page
        min_threshold: Minimum similarity threshold
        
    Returns:
        List of suggestions
    """
    config = SuggestionConfig(
        top_k_per_page=top_k,
        min_similarity_threshold=min_threshold
    )
    engine = SuggestionEngine(config)
    return engine.generate_suggestions(pages, embeddings, existing_links)


def suggestions_to_dataframe(suggestions: List[LinkSuggestion]):
    """Convert suggestions to pandas DataFrame"""
    import pandas as pd
    return pd.DataFrame([s.to_dict() for s in suggestions])


# Example usage
if __name__ == "__main__":
    # Sample data
    pages = [
        {'url': 'https://example.com/seo-guide', 'meta_title': 'SEO Guide', 'h1': 'Complete SEO Guide'},
        {'url': 'https://example.com/internal-links', 'meta_title': 'Internal Links', 'h1': 'Internal Linking Best Practices'},
        {'url': 'https://example.com/cooking', 'meta_title': 'Cooking Tips', 'h1': 'Cooking for Beginners'},
    ]
    
    # Dummy embeddings (in reality these would be generated)
    np.random.seed(42)
    embeddings = np.random.randn(3, 384)
    
    # Make first two more similar
    embeddings[1] = embeddings[0] + np.random.randn(384) * 0.1
    
    # Generate suggestions
    engine = SuggestionEngine()
    suggestions = engine.generate_suggestions(pages, embeddings)
    
    print(f"Generated {len(suggestions)} suggestions:")
    for s in suggestions:
        print(f"  {s.source_url} -> {s.target_url}")
        print(f"    Anchor: {s.suggested_anchor}")
        print(f"    Score: {s.relevance_score:.3f}")
        print(f"    Priority: {s.priority}")
