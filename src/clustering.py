"""
Clustering Module
Implements various clustering algorithms for grouping semantically similar pages
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering operation"""
    labels: np.ndarray
    n_clusters: int
    noise_count: int = 0
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    cluster_centers: Optional[np.ndarray] = None
    silhouette_score: Optional[float] = None
    
    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """Get indices of members in a cluster"""
        return list(np.where(self.labels == cluster_id)[0])


@dataclass 
class DimensionReductionResult:
    """Result of dimension reduction"""
    embeddings_2d: np.ndarray
    method: str
    original_dimension: int


class ClusteringEngine:
    """
    Engine for clustering embeddings using various algorithms.
    
    Usage:
        engine = ClusteringEngine()
        result = engine.cluster(embeddings, method='hdbscan')
    """
    
    AVAILABLE_METHODS = ['hdbscan', 'kmeans', 'agglomerative', 'dbscan']
    
    def __init__(self):
        self._hdbscan = None
        self._kmeans = None
    
    def cluster(
        self,
        embeddings: np.ndarray,
        method: str = 'hdbscan',
        **kwargs
    ) -> ClusterResult:
        """
        Cluster embeddings using specified method.
        
        Args:
            embeddings: Array of embeddings (n_samples x dimension)
            method: Clustering method ('hdbscan', 'kmeans', 'agglomerative', 'dbscan')
            **kwargs: Method-specific parameters
            
        Returns:
            ClusterResult object
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {self.AVAILABLE_METHODS}")
        
        if method == 'hdbscan':
            return self._cluster_hdbscan(embeddings, **kwargs)
        elif method == 'kmeans':
            return self._cluster_kmeans(embeddings, **kwargs)
        elif method == 'agglomerative':
            return self._cluster_agglomerative(embeddings, **kwargs)
        elif method == 'dbscan':
            return self._cluster_dbscan(embeddings, **kwargs)
    
    def _cluster_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int = None,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom'
    ) -> ClusterResult:
        """
        Cluster using HDBSCAN.
        
        HDBSCAN is good for:
        - Finding clusters of varying densities
        - Automatically determining number of clusters
        - Identifying noise/outliers
        
        Args:
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in neighborhood
            metric: Distance metric
            cluster_selection_method: 'eom' (excess of mass) or 'leaf'
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan is required. Install with: pip install hdbscan")
        
        logger.info(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Calculate statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = list(labels).count(-1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[label] = list(labels).count(label)
        
        logger.info(f"Found {n_clusters} clusters, {noise_count} noise points")
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            noise_count=noise_count,
            cluster_sizes=cluster_sizes
        )
    
    def _cluster_kmeans(
        self,
        embeddings: np.ndarray,
        n_clusters: int = None,
        max_clusters: int = 20,
        random_state: int = 42
    ) -> ClusterResult:
        """
        Cluster using K-Means.
        
        K-Means is good for:
        - When you know/want specific number of clusters
        - Fast clustering of large datasets
        - Spherical clusters
        
        Args:
            n_clusters: Number of clusters (if None, will be estimated)
            max_clusters: Maximum clusters to consider for estimation
            random_state: Random seed
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Estimate n_clusters if not provided
        if n_clusters is None:
            n_clusters = self._estimate_n_clusters(
                embeddings, 
                max_clusters=max_clusters
            )
        
        logger.info(f"Clustering with K-Means (n_clusters={n_clusters})")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        sil_score = silhouette_score(embeddings, labels) if n_clusters > 1 else None
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for label in range(n_clusters):
            cluster_sizes[label] = list(labels).count(label)
        
        logger.info(f"Created {n_clusters} clusters (silhouette: {sil_score:.3f})")
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            cluster_centers=kmeans.cluster_centers_,
            silhouette_score=sil_score
        )
    
    def _cluster_agglomerative(
        self,
        embeddings: np.ndarray,
        n_clusters: int = None,
        distance_threshold: float = None,
        linkage: str = 'ward'
    ) -> ClusterResult:
        """
        Cluster using Agglomerative Clustering.
        
        Good for:
        - Hierarchical relationships
        - When you want to visualize dendrograms
        
        Args:
            n_clusters: Number of clusters
            distance_threshold: Distance threshold for cluster merging
            linkage: Linkage method ('ward', 'complete', 'average', 'single')
        """
        from sklearn.cluster import AgglomerativeClustering
        
        if n_clusters is None and distance_threshold is None:
            n_clusters = min(20, len(embeddings) // 10)
        
        logger.info(f"Clustering with Agglomerative (n_clusters={n_clusters})")
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage=linkage
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        n_found = len(set(labels))
        cluster_sizes = {}
        for label in set(labels):
            cluster_sizes[label] = list(labels).count(label)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_found,
            cluster_sizes=cluster_sizes
        )
    
    def _cluster_dbscan(
        self,
        embeddings: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> ClusterResult:
        """
        Cluster using DBSCAN.
        
        Good for:
        - Finding clusters of arbitrary shape
        - Identifying outliers
        
        Args:
            eps: Maximum distance between points in same cluster
            min_samples: Minimum points to form a cluster
        """
        from sklearn.cluster import DBSCAN
        
        logger.info(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})")
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(embeddings)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = list(labels).count(-1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[label] = list(labels).count(label)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            noise_count=noise_count,
            cluster_sizes=cluster_sizes
        )
    
    def _estimate_n_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 20
    ) -> int:
        """
        Estimate optimal number of clusters using elbow method.
        """
        from sklearn.cluster import KMeans
        
        max_clusters = min(max_clusters, len(embeddings) // 2)
        
        if max_clusters < min_clusters:
            return min_clusters
        
        inertias = []
        k_range = range(min_clusters, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Find elbow using second derivative
        if len(inertias) < 3:
            return min_clusters
        
        # Calculate differences
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        
        # Find elbow (maximum second derivative)
        elbow_idx = np.argmax(diffs2) + min_clusters
        
        return elbow_idx


class DimensionReducer:
    """
    Reduce embedding dimensions for visualization.
    """
    
    AVAILABLE_METHODS = ['umap', 'tsne', 'pca']
    
    def reduce(
        self,
        embeddings: np.ndarray,
        method: str = 'umap',
        n_components: int = 2,
        **kwargs
    ) -> DimensionReductionResult:
        """
        Reduce embedding dimensions.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method ('umap', 'tsne', 'pca')
            n_components: Target dimensions
            **kwargs: Method-specific parameters
            
        Returns:
            DimensionReductionResult
        """
        original_dim = embeddings.shape[1]
        
        if method == 'umap':
            reduced = self._reduce_umap(embeddings, n_components, **kwargs)
        elif method == 'tsne':
            reduced = self._reduce_tsne(embeddings, n_components, **kwargs)
        elif method == 'pca':
            reduced = self._reduce_pca(embeddings, n_components)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return DimensionReductionResult(
            embeddings_2d=reduced,
            method=method,
            original_dimension=original_dim
        )
    
    def _reduce_umap(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ) -> np.ndarray:
        """Reduce using UMAP"""
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")
        
        logger.info(f"Reducing dimensions with UMAP (n_neighbors={n_neighbors})")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
        return reducer.fit_transform(embeddings)
    
    def _reduce_tsne(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42
    ) -> np.ndarray:
        """Reduce using t-SNE"""
        from sklearn.manifold import TSNE
        
        # Adjust perplexity for small datasets
        perplexity = min(perplexity, len(embeddings) - 1)
        
        logger.info(f"Reducing dimensions with t-SNE (perplexity={perplexity})")
        
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state
        )
        
        return reducer.fit_transform(embeddings)
    
    def _reduce_pca(
        self,
        embeddings: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce using PCA"""
        from sklearn.decomposition import PCA
        
        logger.info(f"Reducing dimensions with PCA")
        
        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = 'hdbscan',
    **kwargs
) -> ClusterResult:
    """
    Convenience function to cluster embeddings.
    
    Args:
        embeddings: Array of embeddings
        method: Clustering method
        **kwargs: Method parameters
        
    Returns:
        ClusterResult
    """
    engine = ClusteringEngine()
    return engine.cluster(embeddings, method=method, **kwargs)


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = 'umap',
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Convenience function to reduce dimensions.
    
    Args:
        embeddings: High-dimensional embeddings
        method: Reduction method
        n_components: Target dimensions
        
    Returns:
        Reduced embeddings
    """
    reducer = DimensionReducer()
    result = reducer.reduce(embeddings, method=method, n_components=n_components, **kwargs)
    return result.embeddings_2d


def get_cluster_statistics(
    labels: np.ndarray,
    urls: List[str],
    titles: List[str]
) -> List[Dict[str, Any]]:
    """
    Get statistics for each cluster.
    
    Args:
        labels: Cluster labels
        urls: List of page URLs
        titles: List of page titles
        
    Returns:
        List of cluster statistics
    """
    clusters = []
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        if label == -1:
            name = "Unclustered/Noise"
        else:
            name = f"Cluster {label}"
        
        indices = np.where(labels == label)[0]
        cluster_urls = [urls[i] for i in indices]
        cluster_titles = [titles[i] for i in indices]
        
        clusters.append({
            'id': int(label),
            'name': name,
            'size': len(indices),
            'urls': cluster_urls[:10],  # Limit to 10 examples
            'titles': cluster_titles[:10],
            'sample_title': cluster_titles[0] if cluster_titles else ''
        })
    
    return clusters


# Example usage
if __name__ == "__main__":
    # Generate sample embeddings
    np.random.seed(42)
    
    # Create 3 clusters of points
    cluster1 = np.random.randn(20, 384) + np.array([0] * 384)
    cluster2 = np.random.randn(20, 384) + np.array([5] * 384)
    cluster3 = np.random.randn(20, 384) + np.array([10] * 384)
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    
    # Test HDBSCAN
    engine = ClusteringEngine()
    result = engine.cluster(embeddings, method='hdbscan', min_cluster_size=5)
    
    print(f"HDBSCAN: {result.n_clusters} clusters, {result.noise_count} noise")
    print(f"Cluster sizes: {result.cluster_sizes}")
    
    # Test K-Means
    result_km = engine.cluster(embeddings, method='kmeans', n_clusters=3)
    print(f"K-Means: {result_km.n_clusters} clusters")
    print(f"Silhouette score: {result_km.silhouette_score:.3f}")
    
    # Test dimension reduction
    reducer = DimensionReducer()
    reduced = reducer.reduce(embeddings, method='pca')
    print(f"Reduced from {reduced.original_dimension}D to {reduced.embeddings_2d.shape[1]}D")
