"""
Experiment 2: Hyperparameter Sensitivity and Metric Conflict

This module analyzes how sensitive clustering results are to hyperparameter choices
and reveals conflicts between internal and external evaluation metrics.

Two sub-experiments:
2.1 K-Means: Choice of Number of Clusters (k sweep from 2 to 10)
2.2 DBSCAN: Density Parameters (eps and min_samples sweep)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .datasets import ClusteringDataset, standardize_data, get_synthetic_datasets


@dataclass
class KMeansAnalysisResult:
    """Results from K-Means k sweep analysis."""
    k_values: np.ndarray
    sse_values: np.ndarray  # Sum of Squared Errors (inertia)
    silhouette_values: np.ndarray
    calinski_harabasz_values: np.ndarray
    davies_bouldin_values: np.ndarray
    ari_values: Optional[np.ndarray] = None
    nmi_values: Optional[np.ndarray] = None


@dataclass
class DBSCANAnalysisResult:
    """Results from DBSCAN parameter sweep analysis."""
    eps_values: np.ndarray
    min_samples_values: np.ndarray
    ari_matrix: np.ndarray  # Shape: (len(eps), len(min_samples))
    nmi_matrix: np.ndarray
    n_clusters_matrix: np.ndarray
    noise_ratio_matrix: np.ndarray


def analyze_kmeans_k_selection(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> KMeansAnalysisResult:
    """
    Analyze K-Means performance across different values of k.
    
    Args:
        X: Feature matrix
        y_true: Ground truth labels (optional, for external metrics)
        k_range: Range of k values to test
        random_state: Random seed for reproducibility
    
    Returns:
        KMeansAnalysisResult with all computed metrics
    """
    k_values = np.array(list(k_range))
    n_k = len(k_values)
    
    sse_values = np.zeros(n_k)
    silhouette_values = np.zeros(n_k)
    calinski_harabasz_values = np.zeros(n_k)
    davies_bouldin_values = np.zeros(n_k)
    ari_values = np.zeros(n_k) if y_true is not None else None
    nmi_values = np.zeros(n_k) if y_true is not None else None
    
    for i, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # SSE (Sum of Squared Errors) - inertia
        sse_values[i] = kmeans.inertia_
        
        # Internal metrics
        silhouette_values[i] = silhouette_score(X, labels)
        calinski_harabasz_values[i] = calinski_harabasz_score(X, labels)
        davies_bouldin_values[i] = davies_bouldin_score(X, labels)
        
        # External metrics (if ground truth available)
        if y_true is not None:
            ari_values[i] = adjusted_rand_score(y_true, labels)
            nmi_values[i] = normalized_mutual_info_score(y_true, labels)
    
    return KMeansAnalysisResult(
        k_values=k_values,
        sse_values=sse_values,
        silhouette_values=silhouette_values,
        calinski_harabasz_values=calinski_harabasz_values,
        davies_bouldin_values=davies_bouldin_values,
        ari_values=ari_values,
        nmi_values=nmi_values,
    )


def analyze_dbscan_parameters(
    X: np.ndarray,
    y_true: np.ndarray,
    eps_range: np.ndarray = None,
    min_samples_range: np.ndarray = None,
) -> DBSCANAnalysisResult:
    """
    Analyze DBSCAN performance across different eps and min_samples combinations.
    
    Args:
        X: Feature matrix
        y_true: Ground truth labels
        eps_range: Array of eps values to test
        min_samples_range: Array of min_samples values to test
    
    Returns:
        DBSCANAnalysisResult with all computed metrics
    """
    # Default ranges based on data statistics
    if eps_range is None:
        # Use k-distance based heuristic
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_dist = np.sort(distances[:, -1])
        eps_min = np.percentile(k_dist, 10)
        eps_max = np.percentile(k_dist, 95)
        eps_range = np.linspace(eps_min, eps_max, 15)
    
    if min_samples_range is None:
        min_samples_range = np.array([3, 5, 7, 10, 15, 20])
    
    n_eps = len(eps_range)
    n_min_samples = len(min_samples_range)
    
    ari_matrix = np.zeros((n_eps, n_min_samples))
    nmi_matrix = np.zeros((n_eps, n_min_samples))
    n_clusters_matrix = np.zeros((n_eps, n_min_samples))
    noise_ratio_matrix = np.zeros((n_eps, n_min_samples))
    
    for i, eps in enumerate(eps_range):
        for j, min_samples in enumerate(min_samples_range):
            dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
            labels = dbscan.fit_predict(X)
            
            # Count clusters (excluding noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_clusters_matrix[i, j] = n_clusters
            
            # Noise ratio
            noise_ratio = np.sum(labels == -1) / len(labels)
            noise_ratio_matrix[i, j] = noise_ratio
            
            # External metrics (filter noise for fair comparison)
            mask = labels != -1
            if mask.sum() > 1 and n_clusters >= 1:
                # Only compute ARI/NMI if we have valid clusters
                if n_clusters >= 2:
                    ari_matrix[i, j] = adjusted_rand_score(y_true[mask], labels[mask])
                    nmi_matrix[i, j] = normalized_mutual_info_score(y_true[mask], labels[mask])
                else:
                    ari_matrix[i, j] = 0
                    nmi_matrix[i, j] = 0
            else:
                ari_matrix[i, j] = -1  # Invalid clustering
                nmi_matrix[i, j] = -1
    
    return DBSCANAnalysisResult(
        eps_values=eps_range,
        min_samples_values=min_samples_range,
        ari_matrix=ari_matrix,
        nmi_matrix=nmi_matrix,
        n_clusters_matrix=n_clusters_matrix,
        noise_ratio_matrix=noise_ratio_matrix,
    )


def plot_kmeans_elbow(
    result: KMeansAnalysisResult,
    dataset_name: str,
    true_k: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot K-Means elbow analysis with multiple metrics.
    
    Creates a 2x2 subplot:
    - SSE (Elbow Method)
    - Silhouette Score
    - ARI/NMI (external metrics)
    - Davies-Bouldin / Calinski-Harabasz
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    k_values = result.k_values
    
    # Plot 1: SSE (Elbow Method)
    ax1 = axes[0, 0]
    ax1.plot(k_values, result.sse_values, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('SSE (Inertia)', fontsize=11)
    ax1.set_title('Elbow Method (SSE)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if true_k:
        ax1.axvline(x=true_k, color='r', linestyle='--', label=f'True k={true_k}')
        ax1.legend()
    
    # Plot 2: Silhouette Score
    ax2 = axes[0, 1]
    ax2.plot(k_values, result.silhouette_values, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Silhouette Score', fontsize=11)
    ax2.set_title('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Mark optimal k (max silhouette)
    optimal_k_sil = k_values[np.argmax(result.silhouette_values)]
    ax2.axvline(x=optimal_k_sil, color='orange', linestyle='--', 
                label=f'Optimal k={optimal_k_sil}')
    if true_k:
        ax2.axvline(x=true_k, color='r', linestyle='--', label=f'True k={true_k}')
    ax2.legend()
    
    # Plot 3: External Metrics (ARI/NMI)
    ax3 = axes[1, 0]
    if result.ari_values is not None:
        ax3.plot(k_values, result.ari_values, 'r-o', linewidth=2, markersize=8, label='ARI')
        ax3.plot(k_values, result.nmi_values, 'm-s', linewidth=2, markersize=8, label='NMI')
        ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('External Metrics (ARI & NMI)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        if true_k:
            ax3.axvline(x=true_k, color='r', linestyle='--', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'No ground truth available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('External Metrics (N/A)', fontsize=12, fontweight='bold')
    
    # Plot 4: Davies-Bouldin and Calinski-Harabasz
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1, = ax4.plot(k_values, result.davies_bouldin_values, 'c-o', 
                      linewidth=2, markersize=8, label='Davies-Bouldin (↓)')
    ax4.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax4.set_ylabel('Davies-Bouldin Index', color='c', fontsize=11)
    ax4.tick_params(axis='y', labelcolor='c')
    
    line2, = ax4_twin.plot(k_values, result.calinski_harabasz_values, 'y-s', 
                           linewidth=2, markersize=8, label='Calinski-Harabasz (↑)')
    ax4_twin.set_ylabel('Calinski-Harabasz Index', color='y', fontsize=11)
    ax4_twin.tick_params(axis='y', labelcolor='y')
    
    ax4.set_title('Other Internal Metrics', fontsize=12, fontweight='bold')
    ax4.legend([line1, line2], ['Davies-Bouldin (↓)', 'Calinski-Harabasz (↑)'], loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'K-Means Hyperparameter Analysis: {dataset_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_dbscan_heatmaps(
    result: DBSCANAnalysisResult,
    dataset_name: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Plot DBSCAN parameter sensitivity as heatmaps.
    
    Creates a 2x2 subplot:
    - ARI heatmap
    - NMI heatmap
    - Number of clusters heatmap
    - Noise ratio heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    eps_labels = [f'{e:.2f}' for e in result.eps_values]
    ms_labels = [str(int(m)) for m in result.min_samples_values]
    
    # ARI Heatmap
    ax1 = axes[0, 0]
    # Mask invalid values
    ari_masked = np.ma.masked_where(result.ari_matrix < 0, result.ari_matrix)
    im1 = ax1.imshow(ari_masked, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(ms_labels)))
    ax1.set_yticks(np.arange(len(eps_labels)))
    ax1.set_xticklabels(ms_labels)
    ax1.set_yticklabels(eps_labels)
    ax1.set_xlabel('min_samples', fontsize=11)
    ax1.set_ylabel('eps', fontsize=11)
    ax1.set_title('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations
    for i in range(len(eps_labels)):
        for j in range(len(ms_labels)):
            val = result.ari_matrix[i, j]
            if val >= 0:
                color = 'white' if val < 0.3 or val > 0.7 else 'black'
                ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        color=color, fontsize=8)
    
    # NMI Heatmap
    ax2 = axes[0, 1]
    nmi_masked = np.ma.masked_where(result.nmi_matrix < 0, result.nmi_matrix)
    im2 = ax2.imshow(nmi_masked, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(ms_labels)))
    ax2.set_yticks(np.arange(len(eps_labels)))
    ax2.set_xticklabels(ms_labels)
    ax2.set_yticklabels(eps_labels)
    ax2.set_xlabel('min_samples', fontsize=11)
    ax2.set_ylabel('eps', fontsize=11)
    ax2.set_title('Normalized Mutual Information (NMI)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    for i in range(len(eps_labels)):
        for j in range(len(ms_labels)):
            val = result.nmi_matrix[i, j]
            if val >= 0:
                color = 'white' if val < 0.3 or val > 0.7 else 'black'
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        color=color, fontsize=8)
    
    # Number of Clusters Heatmap
    ax3 = axes[1, 0]
    im3 = ax3.imshow(result.n_clusters_matrix, cmap='Blues', aspect='auto')
    ax3.set_xticks(np.arange(len(ms_labels)))
    ax3.set_yticks(np.arange(len(eps_labels)))
    ax3.set_xticklabels(ms_labels)
    ax3.set_yticklabels(eps_labels)
    ax3.set_xlabel('min_samples', fontsize=11)
    ax3.set_ylabel('eps', fontsize=11)
    ax3.set_title('Number of Clusters Found', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3)
    
    for i in range(len(eps_labels)):
        for j in range(len(ms_labels)):
            val = int(result.n_clusters_matrix[i, j])
            ax3.text(j, i, str(val), ha='center', va='center', 
                    color='white' if val > 5 else 'black', fontsize=8)
    
    # Noise Ratio Heatmap
    ax4 = axes[1, 1]
    im4 = ax4.imshow(result.noise_ratio_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(np.arange(len(ms_labels)))
    ax4.set_yticks(np.arange(len(eps_labels)))
    ax4.set_xticklabels(ms_labels)
    ax4.set_yticklabels(eps_labels)
    ax4.set_xlabel('min_samples', fontsize=11)
    ax4.set_ylabel('eps', fontsize=11)
    ax4.set_title('Noise Ratio', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4)
    
    for i in range(len(eps_labels)):
        for j in range(len(ms_labels)):
            val = result.noise_ratio_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color=color, fontsize=8)
    
    plt.suptitle(f'DBSCAN Parameter Sensitivity: {dataset_name}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_metric_conflict_analysis(
    kmeans_result: KMeansAnalysisResult,
    dataset_name: str,
    true_k: int,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Visualize conflicts between internal and external metrics.
    
    Shows cases where internal metrics suggest different k than external metrics.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    k_values = kmeans_result.k_values
    
    # Normalize all metrics to [0, 1] for comparison
    def normalize(arr, higher_is_better=True):
        arr = np.array(arr)
        if not higher_is_better:
            arr = -arr
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
    
    # Plot normalized metrics
    ax.plot(k_values, normalize(kmeans_result.silhouette_values), 
            'g-o', linewidth=2, label='Silhouette (normalized)')
    ax.plot(k_values, normalize(kmeans_result.davies_bouldin_values, higher_is_better=False), 
            'c-s', linewidth=2, label='Davies-Bouldin (normalized, inverted)')
    
    if kmeans_result.ari_values is not None:
        ax.plot(k_values, normalize(kmeans_result.ari_values), 
                'r-^', linewidth=2, label='ARI (normalized)')
        ax.plot(k_values, normalize(kmeans_result.nmi_values), 
                'm-d', linewidth=2, label='NMI (normalized)')
    
    # Mark true k
    ax.axvline(x=true_k, color='black', linestyle='--', linewidth=2, 
               label=f'True k={true_k}')
    
    # Find optimal k for each metric
    optimal_sil = k_values[np.argmax(kmeans_result.silhouette_values)]
    optimal_db = k_values[np.argmin(kmeans_result.davies_bouldin_values)]
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title(f'Metric Conflict Analysis: {dataset_name}\n'
                 f'(Silhouette optimal: k={optimal_sil}, Davies-Bouldin optimal: k={optimal_db})',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def run_experiment2(
    datasets: List[ClusteringDataset] = None,
    output_dir: Path = Path("outputs/experiment2"),
    random_state: int = 42,
) -> Dict:
    """
    Run Experiment 2: Hyperparameter Sensitivity and Metric Conflict.
    
    Args:
        datasets: List of datasets to use (default: synthetic datasets)
        output_dir: Directory to save outputs
        random_state: Random seed
    
    Returns:
        Dictionary containing all results
    """
    print("=" * 70)
    print("Experiment 2: Hyperparameter Sensitivity and Metric Conflict")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use synthetic datasets if not provided
    if datasets is None:
        datasets = get_synthetic_datasets(random_state=random_state)
    
    results = {
        'kmeans': {},
        'dbscan': {},
    }
    
    # ========================================
    # 2.1 K-Means: Choice of Number of Clusters
    # ========================================
    print("\n[2.1] K-Means: Number of Clusters Analysis")
    print("-" * 50)
    
    for dataset in datasets:
        print(f"\nDataset: {dataset.name}")
        
        # Standardize data
        X_scaled, _ = standardize_data(dataset.X)
        
        # Analyze k selection
        kmeans_result = analyze_kmeans_k_selection(
            X=X_scaled,
            y_true=dataset.y,
            k_range=range(2, 11),
            random_state=random_state,
        )
        results['kmeans'][dataset.name] = kmeans_result
        
        # Find optimal k values
        optimal_sil = kmeans_result.k_values[np.argmax(kmeans_result.silhouette_values)]
        optimal_db = kmeans_result.k_values[np.argmin(kmeans_result.davies_bouldin_values)]
        optimal_ari = kmeans_result.k_values[np.argmax(kmeans_result.ari_values)] if kmeans_result.ari_values is not None else None
        
        print(f"  True k: {dataset.n_clusters}")
        print(f"  Optimal k (Silhouette): {optimal_sil}")
        print(f"  Optimal k (Davies-Bouldin): {optimal_db}")
        if optimal_ari:
            print(f"  Optimal k (ARI): {optimal_ari}")
        
        # Check for metric conflict
        if optimal_sil != dataset.n_clusters:
            print(f"  ⚠ Silhouette suggests k={optimal_sil}, but true k={dataset.n_clusters}")
        
        # Plot elbow analysis
        fig = plot_kmeans_elbow(
            result=kmeans_result,
            dataset_name=dataset.name,
            true_k=dataset.n_clusters,
            save_path=output_dir / f"kmeans_elbow_{dataset.name.lower().replace(' ', '_')}.png",
        )
        plt.close(fig)
        
        # Plot metric conflict analysis
        fig = plot_metric_conflict_analysis(
            kmeans_result=kmeans_result,
            dataset_name=dataset.name,
            true_k=dataset.n_clusters,
            save_path=output_dir / f"metric_conflict_{dataset.name.lower().replace(' ', '_')}.png",
        )
        plt.close(fig)
    
    # ========================================
    # 2.2 DBSCAN: Density Parameters
    # ========================================
    print("\n[2.2] DBSCAN: Density Parameters Analysis")
    print("-" * 50)
    
    for dataset in datasets:
        print(f"\nDataset: {dataset.name}")
        
        # Standardize data
        X_scaled, _ = standardize_data(dataset.X)
        
        # Analyze DBSCAN parameters
        dbscan_result = analyze_dbscan_parameters(
            X=X_scaled,
            y_true=dataset.y,
        )
        results['dbscan'][dataset.name] = dbscan_result
        
        # Find best parameters
        best_idx = np.unravel_index(np.argmax(dbscan_result.ari_matrix), dbscan_result.ari_matrix.shape)
        best_eps = dbscan_result.eps_values[best_idx[0]]
        best_min_samples = int(dbscan_result.min_samples_values[best_idx[1]])
        best_ari = dbscan_result.ari_matrix[best_idx]
        
        print(f"  Best params: eps={best_eps:.3f}, min_samples={best_min_samples}")
        print(f"  Best ARI: {best_ari:.4f}")
        print(f"  Clusters at best params: {int(dbscan_result.n_clusters_matrix[best_idx])}")
        print(f"  Noise ratio at best params: {dbscan_result.noise_ratio_matrix[best_idx]:.2%}")
        
        # Plot DBSCAN heatmaps
        fig = plot_dbscan_heatmaps(
            result=dbscan_result,
            dataset_name=dataset.name,
            save_path=output_dir / f"dbscan_heatmap_{dataset.name.lower().replace(' ', '_')}.png",
        )
        plt.close(fig)
    
    # ========================================
    # Create summary table
    # ========================================
    print("\n[Summary] Creating summary tables...")
    
    # K-Means summary
    kmeans_summary = []
    for ds_name, result in results['kmeans'].items():
        row = {
            'Dataset': ds_name,
            'True k': next(d.n_clusters for d in datasets if d.name == ds_name),
            'Optimal k (Silhouette)': int(result.k_values[np.argmax(result.silhouette_values)]),
            'Optimal k (DB)': int(result.k_values[np.argmin(result.davies_bouldin_values)]),
            'Optimal k (ARI)': int(result.k_values[np.argmax(result.ari_values)]) if result.ari_values is not None else 'N/A',
            'Max ARI': f"{np.max(result.ari_values):.4f}" if result.ari_values is not None else 'N/A',
        }
        kmeans_summary.append(row)
    
    df_kmeans = pd.DataFrame(kmeans_summary)
    df_kmeans.to_csv(output_dir / "kmeans_k_selection_summary.csv", index=False)
    
    # DBSCAN summary
    dbscan_summary = []
    for ds_name, result in results['dbscan'].items():
        best_idx = np.unravel_index(np.argmax(result.ari_matrix), result.ari_matrix.shape)
        row = {
            'Dataset': ds_name,
            'Best eps': f"{result.eps_values[best_idx[0]]:.3f}",
            'Best min_samples': int(result.min_samples_values[best_idx[1]]),
            'Best ARI': f"{result.ari_matrix[best_idx]:.4f}",
            'Clusters Found': int(result.n_clusters_matrix[best_idx]),
            'Noise Ratio': f"{result.noise_ratio_matrix[best_idx]:.2%}",
        }
        dbscan_summary.append(row)
    
    df_dbscan = pd.DataFrame(dbscan_summary)
    df_dbscan.to_csv(output_dir / "dbscan_parameter_summary.csv", index=False)
    
    # Print summaries
    print("\n" + "=" * 70)
    print("K-MEANS K SELECTION SUMMARY")
    print("=" * 70)
    print(df_kmeans.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("DBSCAN PARAMETER SENSITIVITY SUMMARY")
    print("=" * 70)
    print(df_dbscan.to_string(index=False))
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS AND OBSERVATIONS")
    print("=" * 70)
    print("""
📊 Key Findings:

1. K-Means k Selection:
   - Internal metrics (Silhouette, DB) may disagree with external metrics (ARI, NMI)
   - The elbow method often suggests different k than ground truth
   - This highlights the challenge of unsupervised model selection

2. DBSCAN Parameter Sensitivity:
   - DBSCAN is highly sensitive to eps and min_samples
   - Small eps → many noise points (all noise in extreme cases)
   - Large eps → single cluster (everything merged)
   - Optimal parameter region is often narrow
   - Different datasets require different parameter tuning

3. Metric Conflict Implications:
   - Internal metrics optimize cluster compactness/separation
   - External metrics measure alignment with ground truth
   - In practice, ground truth is unavailable, making model selection difficult
""")
    
    print(f"\n✓ All results saved to: {output_dir}/")
    print("✓ Experiment 2 completed successfully!")
    
    return results
