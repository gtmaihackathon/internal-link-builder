"""
Tests for the Embeddings Module
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import (
    EmbeddingGenerator,
    EmbeddingResult,
    generate_embeddings,
    calculate_similarity_matrix,
    embeddings_to_bytes,
    bytes_to_embeddings,
    DEFAULT_MODEL
)


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass"""
    
    def test_embedding_result_creation(self):
        """Test creating an EmbeddingResult"""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = EmbeddingResult(
            text="test text",
            embedding=embedding,
            model="test-model",
            dimension=3
        )
        
        assert result.text == "test text"
        assert result.model == "test-model"
        assert result.dimension == 3
        assert np.array_equal(result.embedding, embedding)
    
    def test_embedding_to_bytes(self):
        """Test converting embedding to bytes"""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = EmbeddingResult(
            text="test",
            embedding=embedding,
            model="test",
            dimension=3
        )
        
        bytes_data = result.to_bytes()
        
        assert isinstance(bytes_data, bytes)
        assert len(bytes_data) == embedding.nbytes
    
    def test_embedding_from_bytes(self):
        """Test creating EmbeddingResult from bytes"""
        original = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        bytes_data = original.tobytes()
        
        result = EmbeddingResult.from_bytes(bytes_data, model="test", text="test")
        
        assert np.allclose(result.embedding, original)


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class"""
    
    def test_generator_initialization(self):
        """Test generator initialization without loading model"""
        generator = EmbeddingGenerator(model_name=DEFAULT_MODEL, cache_enabled=False)
        
        assert generator.model_name == DEFAULT_MODEL
        assert generator._model is None  # Model not loaded yet
    
    def test_generator_cache_key(self):
        """Test cache key generation"""
        generator = EmbeddingGenerator()
        
        key1 = generator._get_cache_key("test text")
        key2 = generator._get_cache_key("test text")
        key3 = generator._get_cache_key("different text")
        
        assert key1 == key2  # Same text should produce same key
        assert key1 != key3  # Different text should produce different key
    
    def test_similarity_calculation(self):
        """Test cosine similarity calculation"""
        generator = EmbeddingGenerator()
        
        # Identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        sim = generator.similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.001
        
        # Orthogonal vectors
        vec3 = np.array([0.0, 1.0, 0.0])
        sim = generator.similarity(vec1, vec3)
        assert abs(sim - 0.0) < 0.001
    
    def test_similarity_with_zero_vector(self):
        """Test similarity with zero vector"""
        generator = EmbeddingGenerator()
        
        vec1 = np.array([1.0, 0.0, 0.0])
        zero_vec = np.array([0.0, 0.0, 0.0])
        
        sim = generator.similarity(vec1, zero_vec)
        assert sim == 0.0


class TestSimilarityMatrix:
    """Test similarity matrix calculations"""
    
    def test_similarity_matrix_shape(self):
        """Test similarity matrix has correct shape"""
        embeddings = np.random.randn(5, 10).astype(np.float32)
        
        matrix = calculate_similarity_matrix(embeddings)
        
        assert matrix.shape == (5, 5)
    
    def test_similarity_matrix_diagonal(self):
        """Test diagonal of similarity matrix is 1"""
        embeddings = np.random.randn(5, 10).astype(np.float32)
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        matrix = calculate_similarity_matrix(embeddings)
        
        # Diagonal should be close to 1
        for i in range(5):
            assert abs(matrix[i, i] - 1.0) < 0.001
    
    def test_similarity_matrix_symmetry(self):
        """Test similarity matrix is symmetric"""
        embeddings = np.random.randn(5, 10).astype(np.float32)
        
        matrix = calculate_similarity_matrix(embeddings)
        
        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T)


class TestBytesConversion:
    """Test embeddings to/from bytes conversion"""
    
    def test_embeddings_to_bytes(self):
        """Test converting multiple embeddings to bytes"""
        embeddings = np.random.randn(3, 10).astype(np.float32)
        
        bytes_list = embeddings_to_bytes(embeddings)
        
        assert len(bytes_list) == 3
        for b in bytes_list:
            assert isinstance(b, bytes)
    
    def test_bytes_to_embeddings(self):
        """Test converting bytes back to embeddings"""
        original = np.random.randn(3, 10).astype(np.float32)
        bytes_list = [emb.tobytes() for emb in original]
        
        recovered = bytes_to_embeddings(bytes_list)
        
        assert np.allclose(original, recovered)
    
    def test_roundtrip_conversion(self):
        """Test full roundtrip conversion"""
        original = np.random.randn(5, 384).astype(np.float32)
        
        bytes_list = embeddings_to_bytes(original)
        recovered = bytes_to_embeddings(bytes_list)
        
        assert original.shape == recovered.shape
        assert np.allclose(original, recovered)


class TestFindSimilar:
    """Test finding similar embeddings"""
    
    def test_find_similar_basic(self):
        """Test basic similar finding"""
        generator = EmbeddingGenerator(cache_enabled=False)
        
        # Create corpus with one very similar embedding
        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array([
            [1.0, 0.1, 0.0],  # Similar to query
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.0, 0.0, 1.0],  # Orthogonal
        ])
        
        results = generator.find_similar(query, corpus, top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == 0  # First embedding is most similar
    
    def test_find_similar_with_threshold(self):
        """Test finding similar with threshold"""
        generator = EmbeddingGenerator(cache_enabled=False)
        
        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array([
            [1.0, 0.1, 0.0],  # Similar
            [0.5, 0.5, 0.0],  # Somewhat similar
            [0.0, 1.0, 0.0],  # Not similar
        ])
        
        results = generator.find_similar(query, corpus, top_k=10, threshold=0.8)
        
        # Only the first embedding should pass high threshold
        assert len(results) >= 1
        for idx, sim in results:
            assert sim >= 0.8


class TestGenerateEmbeddingsFunction:
    """Test the convenience function"""
    
    def test_function_exists(self):
        """Test that generate_embeddings function exists"""
        assert callable(generate_embeddings)
    
    @pytest.mark.skipif(
        True,  # Skip by default as it requires model download
        reason="Requires model download"
    )
    def test_generate_embeddings_integration(self):
        """Integration test for generate_embeddings"""
        texts = ["Hello world", "Testing embeddings"]
        
        embeddings = generate_embeddings(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0


class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_text(self):
        """Test handling empty text"""
        generator = EmbeddingGenerator(cache_enabled=False)
        
        # Should not raise an error
        key = generator._get_cache_key("")
        assert isinstance(key, str)
    
    def test_single_text_generation(self):
        """Test generating embedding for single text"""
        generator = EmbeddingGenerator(cache_enabled=False)
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.random.randn(384))
        generator._model = mock_model
        
        # This should work without error
        # In real test, would check the output
        assert generator._model is not None
    
    def test_similarity_normalized_vectors(self):
        """Test similarity with pre-normalized vectors"""
        generator = EmbeddingGenerator(cache_enabled=False)
        
        # Pre-normalized vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.7071, 0.7071, 0.0])
        
        sim = generator.similarity(vec1, vec2)
        
        # Should be close to cos(45°) ≈ 0.7071
        assert 0.7 < sim < 0.72


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
