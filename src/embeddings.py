"""
Embedding Generation Module
Generates semantic embeddings for pages using sentence-transformers
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging
import hashlib
import os
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cache directory
CACHE_DIR = os.getenv("EMBEDDINGS_CACHE_DIR", ".embeddings_cache")


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: np.ndarray
    model: str
    dimension: int
    
    def to_bytes(self) -> bytes:
        """Convert embedding to bytes for database storage"""
        return self.embedding.tobytes()
    
    @classmethod
    def from_bytes(cls, embedding_bytes: bytes, model: str = DEFAULT_MODEL, text: str = "") -> 'EmbeddingResult':
        """Create from bytes"""
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        return cls(
            text=text,
            embedding=embedding,
            model=model,
            dimension=len(embedding)
        )


class EmbeddingGenerator:
    """
    Generate semantic embeddings using sentence-transformers.
    
    Usage:
        generator = EmbeddingGenerator()
        embeddings = generator.generate(texts)
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_enabled: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
            cache_enabled: Whether to cache embeddings
            device: Device to use ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.device = device
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}
        
        # Create cache directory
        if cache_enabled:
            os.makedirs(CACHE_DIR, exist_ok=True)
    
    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Model loaded successfully. Dimension: {self._model.get_sentence_embedding_dimension()}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        hash_input = f"{self.model_name}:{text}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if not self.cache_enabled:
            return None
        
        key = self._get_cache_key(text)
        
        # Check memory cache
        if key in self._cache:
            return self._cache[key]
        
        # Check disk cache
        cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                    self._cache[key] = embedding
                    return embedding
            except:
                pass
        
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        if not self.cache_enabled:
            return
        
        key = self._get_cache_key(text)
        
        # Save to memory cache
        self._cache[key] = embedding
        
        # Save to disk cache (limit disk writes)
        if len(self._cache) % 100 == 0:
            cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            except:
                pass
    
    def generate(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings (n_texts x dimension)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Check cache for all texts
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            logger.info(f"Encoding {len(texts_to_encode)} texts (batch_size={batch_size})")
            
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize
            )
            
            # Save to cache and collect results
            for idx, (text, emb) in enumerate(zip(texts_to_encode, new_embeddings)):
                self._save_to_cache(text, emb)
                embeddings.append((text_indices[idx], emb))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        
        if single_input:
            return result[0]
        return result
    
    def generate_for_pages(
        self,
        pages: List[Dict[str, Any]],
        text_fields: List[str] = ['meta_title', 'h1', 'body_content'],
        max_text_length: int = 2000,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for page data.
        
        Args:
            pages: List of page dictionaries
            text_fields: Fields to concatenate for embedding
            max_text_length: Maximum text length
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        texts = []
        
        for page in pages:
            parts = []
            for field in text_fields:
                value = page.get(field, '')
                if value:
                    parts.append(str(value))
            
            text = ' '.join(parts)[:max_text_length]
            texts.append(text)
        
        return self.generate(texts, batch_size=batch_size)
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize if not already
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix.
        
        Args:
            embeddings: Array of embeddings (n x dimension)
            
        Returns:
            Similarity matrix (n x n)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings)
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[tuple]:
        """
        Find most similar embeddings in corpus.
        
        Args:
            query_embedding: Query embedding
            corpus_embeddings: Corpus of embeddings
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity) tuples
        """
        similarities = []
        
        for i, emb in enumerate(corpus_embeddings):
            sim = self.similarity(query_embedding, emb)
            if sim >= threshold:
                similarities.append((i, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
        
        if os.path.exists(CACHE_DIR):
            import shutil
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)


def generate_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32
) -> np.ndarray:
    """
    Convenience function to generate embeddings.
    
    Args:
        texts: List of texts
        model_name: Model to use
        batch_size: Batch size
        
    Returns:
        Numpy array of embeddings
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.generate(texts, batch_size=batch_size)


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Convenience function to calculate similarity matrix.
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        Similarity matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)


def embeddings_to_bytes(embeddings: np.ndarray) -> List[bytes]:
    """Convert embeddings array to list of bytes for database storage"""
    return [emb.tobytes() for emb in embeddings]


def bytes_to_embeddings(embedding_bytes_list: List[bytes]) -> np.ndarray:
    """Convert list of bytes back to embeddings array"""
    return np.array([
        np.frombuffer(b, dtype=np.float32) 
        for b in embedding_bytes_list
    ])


# Example usage
if __name__ == "__main__":
    # Test embedding generation
    texts = [
        "How to optimize internal links for SEO",
        "Internal linking best practices for websites",
        "Cooking recipes for beginners",
    ]
    
    generator = EmbeddingGenerator()
    embeddings = generator.generate(texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimension: {embeddings.shape[1]}")
    
    # Calculate similarities
    sim_matrix = generator.similarity_matrix(embeddings)
    
    print("\nSimilarity matrix:")
    for i, text in enumerate(texts):
        print(f"\n{text[:50]}...")
        for j, other_text in enumerate(texts):
            print(f"  vs '{other_text[:30]}...': {sim_matrix[i][j]:.3f}")
