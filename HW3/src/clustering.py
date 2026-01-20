"""
Clustering algorithm wrappers for unified interface.
"""

import numpy as np
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    algorithm_name: str
    labels: np.ndarray
    n_clusters_found: int
    params: Dict[str, Any]
    model: Any  # The fitted model object


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for display."""
        pass
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        """Fit the model and return clustering results."""
        pass


class KMeansClustering(ClusteringAlgorithm):
    """K-Means / K-Means++ clustering."""
    
    def __init__(
        self,
        n_clusters: int,
        init: str = "k-means++",
        n_init: int = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
    
    @property
    def name(self) -> str:
        return f"K-Means (k={self.n_clusters})"
    
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels = model.fit_predict(X)
        return ClusteringResult(
            algorithm_name=self.name,
            labels=labels,
            n_clusters_found=self.n_clusters,
            params={
                "n_clusters": self.n_clusters,
                "init": self.init,
                "n_init": self.n_init,
            },
            model=model,
        )


class HierarchicalClustering(ClusteringAlgorithm):
    """Agglomerative Hierarchical Clustering."""
    
    def __init__(
        self,
        n_clusters: int,
        linkage: str = "ward",
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    @property
    def name(self) -> str:
        return f"Hierarchical ({self.linkage})"
    
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
        )
        labels = model.fit_predict(X)
        return ClusteringResult(
            algorithm_name=self.name,
            labels=labels,
            n_clusters_found=self.n_clusters,
            params={
                "n_clusters": self.n_clusters,
                "linkage": self.linkage,
            },
            model=model,
        )


class DBSCANClustering(ClusteringAlgorithm):
    """DBSCAN density-based clustering."""
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
    ):
        self.eps = eps
        self.min_samples = min_samples
    
    @property
    def name(self) -> str:
        return f"DBSCAN (eps={self.eps})"
    
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
        )
        labels = model.fit_predict(X)
        # Count clusters (excluding noise label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return ClusteringResult(
            algorithm_name=self.name,
            labels=labels,
            n_clusters_found=n_clusters,
            params={
                "eps": self.eps,
                "min_samples": self.min_samples,
            },
            model=model,
        )


class GMMClustering(ClusteringAlgorithm):
    """Gaussian Mixture Model clustering (EM algorithm)."""
    
    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
    
    @property
    def name(self) -> str:
        return f"GMM (k={self.n_components})"
    
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        labels = model.fit_predict(X)
        return ClusteringResult(
            algorithm_name=self.name,
            labels=labels,
            n_clusters_found=self.n_components,
            params={
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
            },
            model=model,
        )


class SpectralClusteringWrapper(ClusteringAlgorithm):
    """Spectral Clustering with RBF affinity."""
    
    def __init__(
        self,
        n_clusters: int,
        affinity: str = "rbf",
        gamma: float = 1.0,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
    
    @property
    def name(self) -> str:
        return f"Spectral (k={self.n_clusters})"
    
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            random_state=self.random_state,
            assign_labels="kmeans",
        )
        labels = model.fit_predict(X)
        return ClusteringResult(
            algorithm_name=self.name,
            labels=labels,
            n_clusters_found=self.n_clusters,
            params={
                "n_clusters": self.n_clusters,
                "affinity": self.affinity,
                "gamma": self.gamma,
            },
            model=model,
        )


class BayesianGMMClustering(ClusteringAlgorithm):
    """Bayesian Gaussian Mixture Model (Variational Inference)."""
    
    def __init__(
        self,
        n_components: int = 10,
        weight_concentration_prior_type: str = "dirichlet_process",
        weight_concentration_prior: float = 0.01,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.covariance_type = covariance_type
        self.random_state = random_state
    
    @property
    def name(self) -> str:
        return f"Bayesian GMM (K_max={self.n_components})"
    
    def fit_predict(self, X: np.ndarray) -> ClusteringResult:
        model = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type=self.weight_concentration_prior_type,
            weight_concentration_prior=self.weight_concentration_prior,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        labels = model.fit_predict(X)
        # Count effective clusters (components with significant weight)
        effective_clusters = np.sum(model.weights_ > 0.01)
        return ClusteringResult(
            algorithm_name=self.name,
            labels=labels,
            n_clusters_found=effective_clusters,
            params={
                "n_components": self.n_components,
                "weight_concentration_prior_type": self.weight_concentration_prior_type,
                "weight_concentration_prior": self.weight_concentration_prior,
                "weights": model.weights_,
            },
            model=model,
        )


def get_main_algorithms(n_clusters: int, random_state: int = 42) -> list[ClusteringAlgorithm]:
    """
    Get the five main clustering algorithms for Experiment 1.
    
    Args:
        n_clusters: Target number of clusters (from ground truth)
        random_state: Random seed for reproducibility
    
    Returns:
        List of configured clustering algorithms
    """
    return [
        KMeansClustering(n_clusters=n_clusters, random_state=random_state),
        HierarchicalClustering(n_clusters=n_clusters, linkage="ward"),
        DBSCANClustering(eps=0.5, min_samples=5),
        GMMClustering(n_components=n_clusters, random_state=random_state),
        SpectralClusteringWrapper(n_clusters=n_clusters, random_state=random_state),
    ]


def get_hierarchical_variants(n_clusters: int) -> list[ClusteringAlgorithm]:
    """Get all hierarchical clustering linkage variants."""
    return [
        HierarchicalClustering(n_clusters=n_clusters, linkage="single"),
        HierarchicalClustering(n_clusters=n_clusters, linkage="complete"),
        HierarchicalClustering(n_clusters=n_clusters, linkage="ward"),
    ]


def tune_dbscan_eps(X: np.ndarray, target_n_clusters: int) -> float:
    """
    Heuristic to tune DBSCAN eps parameter based on data.
    
    Uses k-distance graph approach: compute distance to k-th nearest neighbor
    and use a percentile of those distances.
    """
    from sklearn.neighbors import NearestNeighbors
    
    k = min(5, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = distances[:, -1]
    
    # Use the knee point (elbow) in the k-distance graph
    # Approximate by taking a percentile
    eps = np.percentile(k_distances, 90)
    return eps
