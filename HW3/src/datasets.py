"""
Synthetic and real-world dataset generation for clustering experiments.
"""

import numpy as np
from sklearn.datasets import (
    make_blobs,
    make_moons,
    make_circles,
    load_iris,
    load_wine,
    load_digits,
)
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ClusteringDataset:
    """Container for clustering dataset with metadata."""
    name: str
    X: np.ndarray
    y: np.ndarray
    n_clusters: int
    description: str


def generate_blobs(
    n_samples: int = 500,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int = 42,
) -> ClusteringDataset:
    """
    Generate isotropic Gaussian blobs (spherical clusters).
    
    This is a baseline sanity check dataset where K-Means should perform well.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return ClusteringDataset(
        name="Blobs",
        X=X,
        y=y,
        n_clusters=centers,
        description="Spherical Gaussian clusters - baseline for centroid-based methods",
    )


def generate_moons(
    n_samples: int = 500,
    noise: float = 0.1,
    random_state: int = 42,
) -> ClusteringDataset:
    """
    Generate two interleaving half circles (moons).
    
    This dataset has non-convex structure that challenges K-Means.
    """
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )
    return ClusteringDataset(
        name="Two Moons",
        X=X,
        y=y,
        n_clusters=2,
        description="Non-convex interleaving crescents - challenges centroid methods",
    )


def generate_circles(
    n_samples: int = 500,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: int = 42,
) -> ClusteringDataset:
    """
    Generate concentric circles (nested cluster structure).
    
    This dataset has nested structure that requires density or graph-based methods.
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )
    return ClusteringDataset(
        name="Concentric Circles",
        X=X,
        y=y,
        n_clusters=2,
        description="Nested circular clusters - requires non-linear methods",
    )


def generate_anisotropic(
    n_samples: int = 500,
    centers: int = 3,
    random_state: int = 42,
) -> ClusteringDataset:
    """
    Generate anisotropic (elliptical) Gaussian blobs.
    
    Created by applying a linear transformation to standard blobs.
    This challenges K-Means which assumes spherical clusters.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        cluster_std=1.0,
        random_state=random_state,
    )
    # Apply linear transformation to create elongated clusters
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X @ transformation
    
    return ClusteringDataset(
        name="Anisotropic Blobs",
        X=X,
        y=y,
        n_clusters=centers,
        description="Elliptical clusters via linear transformation - GMM advantage",
    )


def load_iris_dataset() -> ClusteringDataset:
    """Load Iris dataset for clustering evaluation."""
    data = load_iris()
    return ClusteringDataset(
        name="Iris",
        X=data.data,
        y=data.target,
        n_clusters=3,
        description="Classic 4-feature flower dataset (150 samples, 3 classes)",
    )


def load_wine_dataset() -> ClusteringDataset:
    """Load Wine dataset for clustering evaluation."""
    data = load_wine()
    return ClusteringDataset(
        name="Wine",
        X=data.data,
        y=data.target,
        n_clusters=3,
        description="Wine recognition dataset (178 samples, 13 features, 3 classes)",
    )


def load_digits_dataset() -> ClusteringDataset:
    """Load Digits dataset for high-dimensional clustering evaluation."""
    data = load_digits()
    return ClusteringDataset(
        name="Digits",
        X=data.data,
        y=data.target,
        n_clusters=10,
        description="8x8 handwritten digits (1797 samples, 64 features, 10 classes)",
    )


def get_synthetic_datasets(random_state: int = 42) -> list[ClusteringDataset]:
    """Generate all synthetic datasets for Experiment 1."""
    return [
        generate_blobs(random_state=random_state),
        generate_moons(random_state=random_state),
        generate_circles(random_state=random_state),
        generate_anisotropic(random_state=random_state),
    ]


def get_real_datasets() -> list[ClusteringDataset]:
    """Load all real-world datasets."""
    return [
        load_iris_dataset(),
        load_wine_dataset(),
        load_digits_dataset(),
    ]


def standardize_data(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Apply z-score normalization to features.
    
    Returns:
        Tuple of (standardized data, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
