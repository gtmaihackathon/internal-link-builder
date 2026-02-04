"""
Internal Link Builder - AI-powered internal linking tool for SEO at scale

Modules:
- app: Streamlit web application
- api: FastAPI REST API  
- cli: Command-line interface
- crawler: Async web crawler
- parser: HTML parser
- embeddings: Semantic embedding generation
- clustering: Page clustering algorithms
- suggestions: Link suggestion engine
- database: Database operations
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for convenience
from .database import (
    init_database,
    get_connection,
    insert_page,
    get_page,
    get_all_pages,
    insert_internal_link,
    get_internal_links,
    insert_suggestion,
    get_suggestions,
    get_statistics,
)

from .crawler import (
    AsyncCrawler,
    CrawlerConfig,
    CrawlResult,
    run_crawler,
)

from .parser import (
    HTMLParser,
    PageData,
    LinkData,
    parse_html,
    extract_text_for_embedding,
)

from .embeddings import (
    EmbeddingGenerator,
    EmbeddingResult,
    generate_embeddings,
    calculate_similarity_matrix,
)

from .clustering import (
    ClusteringEngine,
    ClusterResult,
    DimensionReducer,
    cluster_embeddings,
    reduce_dimensions,
)

from .suggestions import (
    SuggestionEngine,
    SuggestionConfig,
    LinkSuggestion,
    generate_suggestions,
)

__all__ = [
    # Version
    '__version__',
    
    # Database
    'init_database',
    'get_connection', 
    'insert_page',
    'get_page',
    'get_all_pages',
    'insert_internal_link',
    'get_internal_links',
    'insert_suggestion',
    'get_suggestions',
    'get_statistics',
    
    # Crawler
    'AsyncCrawler',
    'CrawlerConfig',
    'CrawlResult',
    'run_crawler',
    
    # Parser
    'HTMLParser',
    'PageData',
    'LinkData',
    'parse_html',
    'extract_text_for_embedding',
    
    # Embeddings
    'EmbeddingGenerator',
    'EmbeddingResult',
    'generate_embeddings',
    'calculate_similarity_matrix',
    
    # Clustering
    'ClusteringEngine',
    'ClusterResult',
    'DimensionReducer',
    'cluster_embeddings',
    'reduce_dimensions',
    
    # Suggestions
    'SuggestionEngine',
    'SuggestionConfig',
    'LinkSuggestion',
    'generate_suggestions',
]
