"""
Visualization utilities for clustering experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_clustering_comparison(
    X: np.ndarray,
    labels_true: np.ndarray,
    results: List[Tuple[str, np.ndarray]],
    dataset_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Create a comparison plot of clustering results across algorithms.
    
    Args:
        X: 2D feature matrix (n_samples, 2)
        labels_true: Ground truth labels
        results: List of (algorithm_name, predicted_labels) tuples
        dataset_name: Name of the dataset for title
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    n_algorithms = len(results)
    n_cols = min(3, n_algorithms + 1)  # +1 for ground truth
    n_rows = (n_algorithms + 1 + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    axes = axes.flatten()
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_scatter(ax, X, labels, title):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if label == -1:
                # Noise points in DBSCAN
                ax.scatter(X[mask, 0], X[mask, 1], c='gray', marker='x', 
                          s=20, alpha=0.5, label='Noise')
            else:
                color = colors[label % len(colors)]
                ax.scatter(X[mask, 0], X[mask, 1], c=[color], 
                          s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    # Plot ground truth
    plot_scatter(axes[0], X, labels_true, f"{dataset_name}\n(Ground Truth)")
    
    # Plot each algorithm's results
    for idx, (name, labels) in enumerate(results, start=1):
        plot_scatter(axes[idx], X, labels, name)
    
    # Hide unused subplots
    for idx in range(len(results) + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_metrics_heatmap(
    metrics_data: Dict[str, Dict[str, float]],
    dataset_names: List[str],
    algorithm_names: List[str],
    metric_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create a heatmap of metric values across datasets and algorithms.
    
    Args:
        metrics_data: Nested dict {dataset: {algorithm: value}}
        dataset_names: List of dataset names
        algorithm_names: List of algorithm names
        metric_name: Name of the metric for title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    # Create matrix
    matrix = np.zeros((len(dataset_names), len(algorithm_names)))
    for i, dataset in enumerate(dataset_names):
        for j, algo in enumerate(algorithm_names):
            value = metrics_data.get(dataset, {}).get(algo, np.nan)
            matrix[i, j] = value if value is not None else np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color map based on metric (DB is lower-is-better)
    cmap = 'RdYlGn' if metric_name != 'Davies-Bouldin' else 'RdYlGn_r'
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_name, rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(len(algorithm_names)))
    ax.set_yticks(np.arange(len(dataset_names)))
    ax.set_xticklabels(algorithm_names, rotation=45, ha='right')
    ax.set_yticklabels(dataset_names)
    
    # Add text annotations
    for i in range(len(dataset_names)):
        for j in range(len(algorithm_names)):
            value = matrix[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}',
                              ha='center', va='center', 
                              color='black' if 0.3 < (value - np.nanmin(matrix)) / 
                                    (np.nanmax(matrix) - np.nanmin(matrix) + 1e-10) < 0.7 
                              else 'white',
                              fontsize=9)
    
    ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_metrics_table(
    data: List[Dict],
    title: str = "Clustering Results",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create a visual table of metrics.
    
    Args:
        data: List of dicts with metric data
        title: Table title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i, key in enumerate(df.columns):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_all_datasets_grid(
    datasets_results: Dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 16),
) -> plt.Figure:
    """
    Create a comprehensive grid showing all datasets and all algorithms.
    
    Args:
        datasets_results: Dict mapping dataset_name to (X, y_true, [(algo_name, labels), ...])
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    n_datasets = len(datasets_results)
    dataset_names = list(datasets_results.keys())
    
    # Get algorithm names from first dataset
    first_data = list(datasets_results.values())[0]
    algorithm_names = ['Ground Truth'] + [name for name, _ in first_data[2]]
    n_algorithms = len(algorithm_names)
    
    fig, axes = plt.subplots(n_datasets, n_algorithms, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for row_idx, (dataset_name, (X, y_true, results)) in enumerate(datasets_results.items()):
        # Plot ground truth in first column
        ax = axes[row_idx, 0]
        for label in np.unique(y_true):
            mask = y_true == label
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[label % len(colors)]], 
                      s=15, alpha=0.7)
        if row_idx == 0:
            ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
        ax.set_ylabel(dataset_name, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot each algorithm
        for col_idx, (algo_name, labels) in enumerate(results, start=1):
            ax = axes[row_idx, col_idx]
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    ax.scatter(X[mask, 0], X[mask, 1], c='gray', marker='x', 
                              s=10, alpha=0.5)
                else:
                    ax.scatter(X[mask, 0], X[mask, 1], c=[colors[label % len(colors)]], 
                              s=15, alpha=0.7)
            if row_idx == 0:
                ax.set_title(algo_name, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Clustering Algorithm Comparison Across Datasets', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def create_summary_bar_chart(
    metrics_data: Dict[str, Dict[str, float]],
    metric_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create a grouped bar chart comparing algorithms across datasets.
    
    Args:
        metrics_data: Dict {dataset: {algorithm: value}}
        metric_name: Name of metric
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_data).T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} by Algorithm and Dataset', fontsize=14, fontweight='bold')
    ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig
