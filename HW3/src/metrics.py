"""
Evaluation metrics for clustering quality assessment.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ClusteringMetrics:
    """Container for all clustering evaluation metrics."""
    # Internal metrics (no ground truth required)
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None
    
    # External metrics (requires ground truth)
    ari: Optional[float] = None  # Adjusted Rand Index
    nmi: Optional[float] = None  # Normalized Mutual Information
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert metrics to dictionary."""
        return {
            "Silhouette": self.silhouette,
            "Calinski-Harabasz": self.calinski_harabasz,
            "Davies-Bouldin": self.davies_bouldin,
            "ARI": self.ari,
            "NMI": self.nmi,
        }
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = []
        if self.silhouette is not None:
            lines.append(f"  Silhouette Score: {self.silhouette:.4f}")
        if self.calinski_harabasz is not None:
            lines.append(f"  Calinski-Harabasz Index: {self.calinski_harabasz:.4f}")
        if self.davies_bouldin is not None:
            lines.append(f"  Davies-Bouldin Index: {self.davies_bouldin:.4f}")
        if self.ari is not None:
            lines.append(f"  Adjusted Rand Index: {self.ari:.4f}")
        if self.nmi is not None:
            lines.append(f"  Normalized Mutual Information: {self.nmi:.4f}")
        return "\n".join(lines)


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> ClusteringMetrics:
    """
    Compute internal clustering metrics (no ground truth required).
    
    Args:
        X: Feature matrix
        labels: Cluster assignments
    
    Returns:
        ClusteringMetrics with internal metrics populated
    """
    metrics = ClusteringMetrics()
    
    # Check if we have valid clustering (at least 2 clusters, not all noise)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters < 2:
        # Cannot compute metrics with less than 2 clusters
        return metrics
    
    # Filter out noise points for metric computation
    mask = labels != -1
    if mask.sum() < 2:
        return metrics
    
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    # Silhouette Score: [-1, 1], higher is better
    try:
        metrics.silhouette = silhouette_score(X_valid, labels_valid)
    except Exception:
        pass
    
    # Calinski-Harabasz Index: higher is better (no upper bound)
    try:
        metrics.calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
    except Exception:
        pass
    
    # Davies-Bouldin Index: lower is better (minimum 0)
    try:
        metrics.davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
    except Exception:
        pass
    
    return metrics


def compute_external_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> ClusteringMetrics:
    """
    Compute external clustering metrics (requires ground truth).
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster assignments
    
    Returns:
        ClusteringMetrics with external metrics populated
    """
    metrics = ClusteringMetrics()
    
    # Filter out noise points (-1 labels) for fair comparison
    mask = labels_pred != -1
    if mask.sum() < 2:
        return metrics
    
    labels_true_valid = labels_true[mask]
    labels_pred_valid = labels_pred[mask]
    
    # Adjusted Rand Index: [-1, 1], 1 is perfect, 0 is random
    try:
        metrics.ari = adjusted_rand_score(labels_true_valid, labels_pred_valid)
    except Exception:
        pass
    
    # Normalized Mutual Information: [0, 1], 1 is perfect
    try:
        metrics.nmi = normalized_mutual_info_score(
            labels_true_valid, labels_pred_valid, average_method="arithmetic"
        )
    except Exception:
        pass
    
    return metrics


def compute_all_metrics(
    X: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
) -> ClusteringMetrics:
    """
    Compute all applicable metrics.
    
    Args:
        X: Feature matrix
        labels_pred: Predicted cluster assignments
        labels_true: Ground truth labels (optional)
    
    Returns:
        ClusteringMetrics with all applicable metrics populated
    """
    # Compute internal metrics
    internal = compute_internal_metrics(X, labels_pred)
    
    # Compute external metrics if ground truth is available
    if labels_true is not None:
        external = compute_external_metrics(labels_true, labels_pred)
        internal.ari = external.ari
        internal.nmi = external.nmi
    
    return internal


def summarize_metrics(
    results: list[tuple[str, ClusteringMetrics]],
) -> str:
    """
    Create a summary table of metrics for multiple algorithms.
    
    Args:
        results: List of (algorithm_name, metrics) tuples
    
    Returns:
        Formatted table string
    """
    import pandas as pd
    
    data = []
    for name, metrics in results:
        row = {"Algorithm": name}
        row.update(metrics.to_dict())
        data.append(row)
    
    df = pd.DataFrame(data)
    return df.to_string(index=False)


def rank_algorithms(
    results: list[tuple[str, ClusteringMetrics]],
    metric: str = "ari",
) -> list[tuple[str, float]]:
    """
    Rank algorithms by a specific metric.
    
    Args:
        results: List of (algorithm_name, metrics) tuples
        metric: Metric to rank by ('silhouette', 'ari', 'nmi', etc.)
    
    Returns:
        Sorted list of (algorithm_name, metric_value) tuples
    """
    rankings = []
    for name, metrics in results:
        value = getattr(metrics, metric, None)
        if value is not None:
            rankings.append((name, value))
    
    # Higher is better for most metrics, except Davies-Bouldin
    reverse = metric != "davies_bouldin"
    rankings.sort(key=lambda x: x[1], reverse=reverse)
    
    return rankings
