"""
NYCU Pattern Recognition HW3 - Clustering Algorithm Comparison

This script implements Experiment 1: Geometric Structure and Algorithm Assumptions
Evaluates how different clustering algorithms perform under varying cluster geometries.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.datasets import (
    get_synthetic_datasets,
    standardize_data,
    ClusteringDataset,
)
from src.clustering import (
    get_main_algorithms,
    tune_dbscan_eps,
    ClusteringResult,
    DBSCANClustering,
)
from src.metrics import compute_all_metrics, ClusteringMetrics
from src.visualization import (
    plot_clustering_comparison,
    plot_all_datasets_grid,
    plot_metrics_heatmap,
    create_summary_bar_chart,
)


# Configuration
RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs/experiment1")


def run_experiment1():
    """
    Experiment 1: Geometric Structure and Algorithm Assumptions
    
    Evaluates how different algorithms perform under varying cluster geometries
    using synthetic datasets.
    """
    print("=" * 70)
    print("Experiment 1: Geometric Structure and Algorithm Assumptions")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic datasets
    print("\n[1/4] Generating synthetic datasets...")
    datasets = get_synthetic_datasets(random_state=RANDOM_STATE)
    
    for ds in datasets:
        print(f"  - {ds.name}: {ds.X.shape[0]} samples, {ds.n_clusters} clusters")
        print(f"    {ds.description}")
    
    # Store results for visualization
    all_results: Dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]] = {}
    all_metrics: Dict[str, Dict[str, ClusteringMetrics]] = {}
    
    print("\n[2/4] Running clustering algorithms on each dataset...")
    
    for dataset in datasets:
        print(f"\n{'─' * 50}")
        print(f"Dataset: {dataset.name}")
        print(f"{'─' * 50}")
        
        # Standardize features (z-score normalization)
        X_scaled, _ = standardize_data(dataset.X)
        
        # Get algorithms with correct number of clusters
        algorithms = get_main_algorithms(
            n_clusters=dataset.n_clusters, 
            random_state=RANDOM_STATE
        )
        
        # Tune DBSCAN eps for this specific dataset
        # Replace the default DBSCAN with tuned version
        tuned_eps = tune_dbscan_eps(X_scaled, dataset.n_clusters)
        for i, algo in enumerate(algorithms):
            if isinstance(algo, DBSCANClustering):
                algorithms[i] = DBSCANClustering(eps=tuned_eps, min_samples=5)
                print(f"  Tuned DBSCAN eps: {tuned_eps:.3f}")
        
        dataset_results = []
        dataset_metrics = {}
        
        for algo in algorithms:
            # Fit and predict
            result = algo.fit_predict(X_scaled)
            
            # Compute metrics
            metrics = compute_all_metrics(
                X_scaled, 
                result.labels, 
                labels_true=dataset.y
            )
            
            dataset_results.append((result.algorithm_name, result.labels))
            dataset_metrics[result.algorithm_name] = metrics
            
            print(f"\n  {result.algorithm_name}:")
            print(f"    Clusters found: {result.n_clusters_found}")
            print(metrics)
        
        # Store for visualization
        all_results[dataset.name] = (X_scaled, dataset.y, dataset_results)
        all_metrics[dataset.name] = dataset_metrics
    
    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    
    # Individual dataset comparison plots
    for dataset_name, (X, y_true, results) in all_results.items():
        fig = plot_clustering_comparison(
            X=X,
            labels_true=y_true,
            results=results,
            dataset_name=dataset_name,
            save_path=OUTPUT_DIR / f"{dataset_name.lower().replace(' ', '_')}_comparison.png"
        )
        plt.close(fig)
    
    # All datasets grid
    fig = plot_all_datasets_grid(
        datasets_results=all_results,
        save_path=OUTPUT_DIR / "all_datasets_grid.png"
    )
    plt.close(fig)
    
    # Metrics heatmaps
    metric_names = ['silhouette', 'ari', 'nmi', 'davies_bouldin']
    display_names = {
        'silhouette': 'Silhouette Score',
        'ari': 'Adjusted Rand Index (ARI)',
        'nmi': 'Normalized Mutual Information (NMI)',
        'davies_bouldin': 'Davies-Bouldin Index'
    }
    
    for metric in metric_names:
        metrics_data = {}
        for dataset_name, algo_metrics in all_metrics.items():
            metrics_data[dataset_name] = {}
            for algo_name, m in algo_metrics.items():
                value = getattr(m, metric, None)
                metrics_data[dataset_name][algo_name] = value
        
        dataset_names = list(metrics_data.keys())
        algorithm_names = list(next(iter(metrics_data.values())).keys())
        
        fig = plot_metrics_heatmap(
            metrics_data=metrics_data,
            dataset_names=dataset_names,
            algorithm_names=algorithm_names,
            metric_name=display_names[metric],
            save_path=OUTPUT_DIR / f"heatmap_{metric}.png"
        )
        plt.close(fig)
    
    # Create summary table
    print("\n[4/4] Creating summary tables...")
    
    summary_data = []
    for dataset_name, algo_metrics in all_metrics.items():
        for algo_name, metrics in algo_metrics.items():
            row = {
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'Silhouette': f"{metrics.silhouette:.4f}" if metrics.silhouette else "N/A",
                'ARI': f"{metrics.ari:.4f}" if metrics.ari else "N/A",
                'NMI': f"{metrics.nmi:.4f}" if metrics.nmi else "N/A",
                'Davies-Bouldin': f"{metrics.davies_bouldin:.4f}" if metrics.davies_bouldin else "N/A",
            }
            summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = OUTPUT_DIR / "experiment1_results.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 RESULTS SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    # Analysis and observations
    print_analysis(all_metrics)
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("✓ Experiment 1 completed successfully!")
    
    return all_results, all_metrics


def print_analysis(all_metrics: Dict[str, Dict[str, ClusteringMetrics]]):
    """Print analysis and observations based on results."""
    print("\n" + "=" * 70)
    print("ANALYSIS AND OBSERVATIONS")
    print("=" * 70)
    
    # Find best algorithm for each dataset
    print("\n📊 Best Algorithm by ARI (per dataset):")
    for dataset_name, algo_metrics in all_metrics.items():
        best_algo = None
        best_ari = -1
        for algo_name, metrics in algo_metrics.items():
            if metrics.ari is not None and metrics.ari > best_ari:
                best_ari = metrics.ari
                best_algo = algo_name
        if best_algo:
            print(f"  {dataset_name}: {best_algo} (ARI = {best_ari:.4f})")
    
    # Key observations
    print("\n📝 Expected Observations:")
    print("  1. K-Means performs well on Blobs but fails on Moons and Circles")
    print("  2. DBSCAN and Spectral Clustering succeed on non-convex structures")
    print("  3. GMM outperforms K-Means on Anisotropic Blobs (elliptical clusters)")
    print("  4. Hierarchical (Ward) behaves similarly to K-Means on spherical data")


# Import matplotlib at module level for visualization
import matplotlib.pyplot as plt


def main():
    """Main entry point."""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        experiment = sys.argv[1].lower()
    else:
        experiment = "all"
    
    if experiment == "1" or experiment == "exp1":
        run_experiment1()
    elif experiment == "2" or experiment == "exp2":
        from src.experiment2 import run_experiment2
        run_experiment2()
    elif experiment == "3" or experiment == "exp3":
        from src.experiment3 import run_experiment3
        run_experiment3()
    elif experiment == "bonus" or experiment == "bayesian":
        from src.bonus_bayesian import run_all_bonus_experiments
        run_all_bonus_experiments()
    elif experiment == "all":
        print("\n" + "=" * 70)
        print("Running All Experiments (including Bonus)")
        print("=" * 70 + "\n")
        
        run_experiment1()
        
        print("\n" + "=" * 70 + "\n")
        
        from src.experiment2 import run_experiment2
        run_experiment2()
        
        print("\n" + "=" * 70 + "\n")
        
        from src.experiment3 import run_experiment3
        run_experiment3()
        
        print("\n" + "=" * 70 + "\n")
        
        from src.bonus_bayesian import run_all_bonus_experiments
        run_all_bonus_experiments()
    elif experiment == "main":
        # Run only main experiments (1, 2, 3) without bonus
        print("\n" + "=" * 70)
        print("Running Main Experiments (1, 2, 3)")
        print("=" * 70 + "\n")
        
        run_experiment1()
        
        print("\n" + "=" * 70 + "\n")
        
        from src.experiment2 import run_experiment2
        run_experiment2()
        
        print("\n" + "=" * 70 + "\n")
        
        from src.experiment3 import run_experiment3
        run_experiment3()
    else:
        print(f"Unknown experiment: {experiment}")
        print("Usage: python main.py [1|2|3|bonus|main|all]")
        print("  1 or exp1: Run Experiment 1 (Geometric Structure)")
        print("  2 or exp2: Run Experiment 2 (Hyperparameter Sensitivity)")
        print("  3 or exp3: Run Experiment 3 (Dimensionality Reduction)")
        print("  bonus:     Run Bonus Experiments (Bayesian Clustering)")
        print("  main:      Run main experiments (1, 2, 3) without bonus")
        print("  all:       Run all experiments including bonus (default)")
        sys.exit(1)


if __name__ == "__main__":
    main()
