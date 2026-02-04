"""
Application Settings for Internal Link Builder
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "internal_links.db"))

# Crawler settings
DEFAULT_USER_AGENT = "googlebot"
DEFAULT_CONCURRENT_REQUESTS = 10
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_REQUEST_DELAY = 0.5

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 2000  # Max characters for embedding

# Clustering settings
DEFAULT_CLUSTERING_METHOD = "hdbscan"
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_N_CLUSTERS = 10

# Suggestion settings
DEFAULT_TOP_K_SUGGESTIONS = 5
MIN_SIMILARITY_THRESHOLD = 0.3
HIGH_PRIORITY_THRESHOLD = 0.7
MEDIUM_PRIORITY_THRESHOLD = 0.5

# Visualization settings
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 30

# Rate limiting
MAX_REQUESTS_PER_SECOND = 2
BACKOFF_FACTOR = 1.5
MAX_RETRIES = 3
