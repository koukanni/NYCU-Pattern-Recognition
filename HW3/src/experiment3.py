"""
Experiment 3: High-Dimensional Data and Dimensionality Reduction

This module investigates the impact of dimensionality on clustering quality and efficiency.

Procedure:
1. Apply clustering algorithms directly to original features
2. Apply PCA to reduce dimensionality (e.g., 10 and 20 dimensions)
3. Repeat clustering on reduced representations
4. Compare: Runtime, Internal metrics, External metrics
5. Optional: t-SNE plots for qualitative inspection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .datasets import (
    ClusteringDataset,
    standardize_data,
    load_digits_dataset,
    load_iris_dataset,
    load_wine_dataset,
)
from .clustering import (
    get_main_algorithms,
    ClusteringAlgorithm,
    ClusteringResult,
)
from .metrics import compute_all_metrics, ClusteringMetrics


@dataclass
class DimensionalityExperimentResult:
    """Results from a single clustering run with timing."""
    algorithm_name: str
    n_dimensions: int
    dimension_label: str  # e.g., "Original (64)", "PCA-10", "PCA-20"
    runtime_seconds: float
    metrics: ClusteringMetrics
    labels: np.ndarray


def apply_pca(
    X: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA, float]:
    """
    Apply PCA dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_components: Number of principal components
        random_state: Random seed
    
    Returns:
        Tuple of (transformed data, fitted PCA object, explained variance ratio)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    return X_reduced, pca, explained_variance


def run_clustering_with_timing(
    algorithm: ClusteringAlgorithm,
    X: np.ndarray,
    y_true: np.ndarray,
    n_dimensions: int,
    dimension_label: str,
) -> DimensionalityExperimentResult:
    """
    Run a clustering algorithm and measure execution time.
    
    Args:
        algorithm: Clustering algorithm to run
        X: Feature matrix
        y_true: Ground truth labels
        n_dimensions: Number of dimensions in X
        dimension_label: Label for this dimension setting
    
    Returns:
        DimensionalityExperimentResult with timing and metrics
    """
    # Measure runtime
    start_time = time.perf_counter()
    result = algorithm.fit_predict(X)
    end_time = time.perf_counter()
    
    runtime = end_time - start_time
    
    # Compute metrics
    metrics = compute_all_metrics(X, result.labels, y_true)
    
    return DimensionalityExperimentResult(
        algorithm_name=result.algorithm_name,
        n_dimensions=n_dimensions,
        dimension_label=dimension_label,
        runtime_seconds=runtime,
        metrics=metrics,
        labels=result.labels,
    )


def plot_pca_explained_variance(
    X: np.ndarray,
    max_components: int = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot PCA explained variance analysis.
    
    Creates two subplots:
    - Individual explained variance ratio per component
    - Cumulative explained variance ratio
    """
    if max_components is None:
        max_components = min(X.shape[1], 50)
    
    pca = PCA(n_components=max_components)
    pca.fit(X)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    ax1 = axes[0]
    ax1.bar(range(1, max_components + 1), pca.explained_variance_ratio_, 
            alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=11)
    ax1.set_title('Individual Explained Variance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2 = axes[1]
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, max_components + 1), cumulative_variance, 
             'b-o', linewidth=2, markersize=4)
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=11)
    ax2.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Find components for 90% and 95% variance
    n_90 = np.argmax(cumulative_variance >= 0.9) + 1
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    ax2.axvline(x=n_90, color='r', linestyle=':', alpha=0.5)
    ax2.axvline(x=n_95, color='orange', linestyle=':', alpha=0.5)
    
    plt.suptitle(f'PCA Analysis (Original: {X.shape[1]} dimensions)\n'
                 f'90% variance: {n_90} components, 95% variance: {n_95} components',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_tsne_visualization(
    X: np.ndarray,
    y_true: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    dataset_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
    random_state: int = 42,
) -> plt.Figure:
    """
    Create t-SNE visualizations comparing ground truth and clustering results.
    
    Args:
        X: Feature matrix (will be reduced to 2D via t-SNE)
        y_true: Ground truth labels
        labels_dict: Dictionary mapping algorithm names to predicted labels
        dataset_name: Name of dataset for title
        save_path: Path to save figure
        figsize: Figure size
        random_state: Random seed for t-SNE
    
    Returns:
        matplotlib Figure object
    """
    # Apply t-SNE for 2D visualization
    print("  Computing t-SNE embedding (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    n_plots = len(labels_dict) + 1  # +1 for ground truth
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_scatter(ax, X_2d, labels, title):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if label == -1:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c='gray', marker='x',
                          s=20, alpha=0.5, label='Noise')
            else:
                color = colors[label % len(colors)]
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color],
                          s=20, alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Plot ground truth
    plot_scatter(axes[0], X_tsne, y_true, f'Ground Truth\n({len(np.unique(y_true))} classes)')
    
    # Plot each algorithm's results
    for idx, (algo_name, labels) in enumerate(labels_dict.items(), start=1):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        plot_scatter(axes[idx], X_tsne, labels, f'{algo_name}\n({n_clusters} clusters)')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f't-SNE Visualization: {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_dimensionality_comparison(
    results: List[DimensionalityExperimentResult],
    metric_name: str,
    dataset_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot comparison of a metric across different dimensionalities.
    """
    # Organize data by algorithm and dimension
    algorithms = sorted(set(r.algorithm_name for r in results))
    dimensions = sorted(set(r.dimension_label for r in results), 
                       key=lambda x: (0 if 'Original' in x else int(x.split('-')[1]) if '-' in x else 999))
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(algorithms))
    width = 0.8 / len(dimensions)
    
    for i, dim_label in enumerate(dimensions):
        values = []
        for algo in algorithms:
            result = next((r for r in results 
                          if r.algorithm_name == algo and r.dimension_label == dim_label), None)
            if result:
                value = getattr(result.metrics, metric_name, None)
                values.append(value if value is not None else 0)
            else:
                values.append(0)
        
        offset = (i - len(dimensions)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=dim_label, alpha=0.8)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric_name.replace("_", " ").title()} by Dimensionality: {dataset_name}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.split('(')[0].strip() for a in algorithms], rotation=45, ha='right')
    ax.legend(title='Dimensions')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_runtime_comparison(
    results: List[DimensionalityExperimentResult],
    dataset_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot runtime comparison across dimensionalities.
    """
    algorithms = sorted(set(r.algorithm_name for r in results))
    dimensions = sorted(set(r.dimension_label for r in results),
                       key=lambda x: (0 if 'Original' in x else int(x.split('-')[1]) if '-' in x else 999))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(algorithms))
    width = 0.8 / len(dimensions)
    
    for i, dim_label in enumerate(dimensions):
        values = []
        for algo in algorithms:
            result = next((r for r in results
                          if r.algorithm_name == algo and r.dimension_label == dim_label), None)
            values.append(result.runtime_seconds * 1000 if result else 0)  # Convert to ms
        
        offset = (i - len(dimensions)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=dim_label, alpha=0.8)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Runtime (ms)', fontsize=12)
    ax.set_title(f'Runtime Comparison by Dimensionality: {dataset_name}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.split('(')[0].strip() for a in algorithms], rotation=45, ha='right')
    ax.legend(title='Dimensions')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_comprehensive_summary(
    results: List[DimensionalityExperimentResult],
    dataset_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive summary plot with multiple metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    algorithms = sorted(set(r.algorithm_name for r in results))
    dimensions = sorted(set(r.dimension_label for r in results),
                       key=lambda x: (0 if 'Original' in x else int(x.split('-')[1]) if '-' in x else 999))
    
    x = np.arange(len(algorithms))
    width = 0.8 / len(dimensions)
    
    metrics_to_plot = [
        ('ari', 'Adjusted Rand Index (ARI)', True),
        ('nmi', 'Normalized Mutual Information (NMI)', True),
        ('silhouette', 'Silhouette Score', True),
        ('davies_bouldin', 'Davies-Bouldin Index', False),  # Lower is better
        ('runtime_seconds', 'Runtime (ms)', False),
    ]
    
    for ax_idx, (metric_key, metric_title, higher_is_better) in enumerate(metrics_to_plot):
        ax = axes.flatten()[ax_idx]
        
        for i, dim_label in enumerate(dimensions):
            values = []
            for algo in algorithms:
                result = next((r for r in results
                              if r.algorithm_name == algo and r.dimension_label == dim_label), None)
                if result:
                    if metric_key == 'runtime_seconds':
                        value = result.runtime_seconds * 1000  # Convert to ms
                    else:
                        value = getattr(result.metrics, metric_key, None)
                    values.append(value if value is not None else 0)
                else:
                    values.append(0)
            
            offset = (i - len(dimensions)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=dim_label, alpha=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=10)
        ax.set_ylabel(metric_title, fontsize=10)
        ax.set_title(metric_title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([a.split('(')[0].strip()[:10] for a in algorithms], 
                          rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        if ax_idx == 0:
            ax.legend(title='Dimensions', fontsize=8)
    
    # Hide the last subplot (we have 5 metrics, 6 subplots)
    axes.flatten()[5].set_visible(False)
    
    plt.suptitle(f'Dimensionality Reduction Impact: {dataset_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def run_experiment3(
    dataset: ClusteringDataset = None,
    pca_dimensions: List[int] = [10, 20, 30],
    output_dir: Path = Path("outputs/experiment3"),
    random_state: int = 42,
) -> Dict:
    """
    Run Experiment 3: High-Dimensional Data and Dimensionality Reduction.
    
    Args:
        dataset: Dataset to use (default: Digits dataset)
        pca_dimensions: List of PCA dimensions to test
        output_dir: Directory to save outputs
        random_state: Random seed
    
    Returns:
        Dictionary containing all results
    """
    print("=" * 70)
    print("Experiment 3: High-Dimensional Data and Dimensionality Reduction")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use Digits dataset if not provided (recommended in README)
    if dataset is None:
        dataset = load_digits_dataset()
    
    print(f"\nDataset: {dataset.name}")
    print(f"  Samples: {dataset.X.shape[0]}")
    print(f"  Original dimensions: {dataset.X.shape[1]}")
    print(f"  Classes: {dataset.n_clusters}")
    print(f"  {dataset.description}")
    
    # Standardize data
    X_scaled, _ = standardize_data(dataset.X)
    
    # ========================================
    # PCA Analysis
    # ========================================
    print("\n[1/4] PCA Explained Variance Analysis")
    print("-" * 50)
    
    fig = plot_pca_explained_variance(
        X=X_scaled,
        save_path=output_dir / "pca_explained_variance.png",
    )
    plt.close(fig)
    
    # Show variance explained by chosen dimensions
    for n_dim in pca_dimensions:
        _, _, var_explained = apply_pca(X_scaled, n_dim, random_state)
        print(f"  PCA-{n_dim}: {var_explained:.2%} variance explained")
    
    # ========================================
    # Run Clustering at Different Dimensionalities
    # ========================================
    print("\n[2/4] Running Clustering at Different Dimensionalities")
    print("-" * 50)
    
    all_results: List[DimensionalityExperimentResult] = []
    
    # Prepare data at different dimensionalities
    data_versions = [
        (X_scaled, dataset.X.shape[1], f"Original ({dataset.X.shape[1]})"),
    ]
    
    for n_dim in pca_dimensions:
        X_pca, _, var = apply_pca(X_scaled, n_dim, random_state)
        data_versions.append((X_pca, n_dim, f"PCA-{n_dim}"))
    
    # Get algorithms
    algorithms = get_main_algorithms(
        n_clusters=dataset.n_clusters,
        random_state=random_state,
    )
    
    # Run clustering for each data version and algorithm
    for X_data, n_dim, dim_label in data_versions:
        print(f"\n  {dim_label}:")
        
        for algo in algorithms:
            # Create fresh algorithm instance (some algorithms cache state)
            algo_fresh = type(algo).__new__(type(algo))
            algo_fresh.__dict__.update(algo.__dict__)
            
            result = run_clustering_with_timing(
                algorithm=algo_fresh,
                X=X_data,
                y_true=dataset.y,
                n_dimensions=n_dim,
                dimension_label=dim_label,
            )
            all_results.append(result)
            
            ari = result.metrics.ari if result.metrics.ari else 0
            print(f"    {result.algorithm_name}: ARI={ari:.4f}, "
                  f"Runtime={result.runtime_seconds*1000:.2f}ms")
    
    # ========================================
    # Generate Visualizations
    # ========================================
    print("\n[3/4] Generating Visualizations")
    print("-" * 50)
    
    # Comprehensive summary
    fig = plot_comprehensive_summary(
        results=all_results,
        dataset_name=dataset.name,
        save_path=output_dir / "comprehensive_summary.png",
    )
    plt.close(fig)
    
    # Individual metric comparisons
    for metric in ['ari', 'nmi', 'silhouette']:
        fig = plot_dimensionality_comparison(
            results=all_results,
            metric_name=metric,
            dataset_name=dataset.name,
            save_path=output_dir / f"comparison_{metric}.png",
        )
        plt.close(fig)
    
    # Runtime comparison
    fig = plot_runtime_comparison(
        results=all_results,
        dataset_name=dataset.name,
        save_path=output_dir / "runtime_comparison.png",
    )
    plt.close(fig)
    
    # t-SNE visualization (on PCA-reduced data for speed)
    print("\n  Generating t-SNE visualization...")
    X_pca_30, _, _ = apply_pca(X_scaled, 30, random_state)
    
    # Get labels from original dimension clustering
    labels_dict = {}
    for result in all_results:
        if "Original" in result.dimension_label:
            # Simplify algorithm name for display
            simple_name = result.algorithm_name.split('(')[0].strip()
            labels_dict[simple_name] = result.labels
    
    fig = plot_tsne_visualization(
        X=X_pca_30,  # Use PCA-reduced for faster t-SNE
        y_true=dataset.y,
        labels_dict=labels_dict,
        dataset_name=dataset.name,
        save_path=output_dir / "tsne_visualization.png",
        random_state=random_state,
    )
    plt.close(fig)
    
    # ========================================
    # Create Summary Tables
    # ========================================
    print("\n[4/4] Creating Summary Tables")
    print("-" * 50)
    
    # Create detailed results table
    summary_data = []
    for result in all_results:
        row = {
            'Algorithm': result.algorithm_name,
            'Dimensions': result.dimension_label,
            'ARI': f"{result.metrics.ari:.4f}" if result.metrics.ari else "N/A",
            'NMI': f"{result.metrics.nmi:.4f}" if result.metrics.nmi else "N/A",
            'Silhouette': f"{result.metrics.silhouette:.4f}" if result.metrics.silhouette else "N/A",
            'Davies-Bouldin': f"{result.metrics.davies_bouldin:.4f}" if result.metrics.davies_bouldin else "N/A",
            'Runtime (ms)': f"{result.runtime_seconds * 1000:.2f}",
        }
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / "experiment3_results.csv", index=False)
    
    # Create comparison table (algorithms as rows, dimensions as columns for ARI)
    pivot_ari = df_summary.pivot(index='Algorithm', columns='Dimensions', values='ARI')
    pivot_ari.to_csv(output_dir / "ari_by_dimension.csv")
    
    pivot_runtime = df_summary.pivot(index='Algorithm', columns='Dimensions', values='Runtime (ms)')
    pivot_runtime.to_csv(output_dir / "runtime_by_dimension.csv")
    
    # Print summaries
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 RESULTS SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ARI BY DIMENSIONALITY")
    print("=" * 70)
    print(pivot_ari.to_string())
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS AND OBSERVATIONS")
    print("=" * 70)
    
    # Find best performing dimension for each algorithm
    print("\n📊 Best Dimensionality by Algorithm (ARI):")
    for algo in set(r.algorithm_name for r in all_results):
        algo_results = [r for r in all_results if r.algorithm_name == algo]
        best = max(algo_results, key=lambda r: r.metrics.ari if r.metrics.ari else 0)
        print(f"  {algo}: {best.dimension_label} (ARI={best.metrics.ari:.4f})")
    
    # Runtime improvement
    print("\n⏱ Runtime Improvement with Dimensionality Reduction:")
    for algo in set(r.algorithm_name for r in all_results):
        algo_results = [r for r in all_results if r.algorithm_name == algo]
        original = next((r for r in algo_results if "Original" in r.dimension_label), None)
        pca_best = min([r for r in algo_results if "PCA" in r.dimension_label], 
                       key=lambda r: r.runtime_seconds)
        if original and pca_best:
            speedup = original.runtime_seconds / pca_best.runtime_seconds
            print(f"  {algo.split('(')[0].strip()}: {speedup:.2f}x faster with {pca_best.dimension_label}")
    
    print("""
📝 Key Findings:

1. Curse of Dimensionality:
   - High-dimensional data leads to distance concentration
   - Points become equidistant, making clustering harder
   - Dimensionality reduction can recover meaningful structure

2. PCA Benefits:
   - Reduces noise by removing low-variance components
   - Improves computational efficiency
   - May improve clustering quality if signal is in top components

3. PCA Risks:
   - May discard discriminative information in lower components
   - Not all cluster structure aligns with principal components
   - Trade-off between variance explained and cluster separability

4. Algorithm-Specific Observations:
   - K-Means and GMM benefit from noise reduction via PCA
   - Spectral Clustering may be robust to dimensionality with proper kernel
   - DBSCAN requires density parameter re-tuning after PCA
""")
    
    print(f"\n✓ All results saved to: {output_dir}/")
    print("✓ Experiment 3 completed successfully!")
    
    return {
        'results': all_results,
        'summary': df_summary,
        'pivot_ari': pivot_ari,
        'pivot_runtime': pivot_runtime,
    }
