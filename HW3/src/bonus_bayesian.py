"""
Bonus Experiment: Bayesian Clustering

This module demonstrates how Bayesian modeling:
- Handles uncertainty
- Controls model complexity
- Reduces reliance on manual selection of cluster number

Three sub-experiments:
1. Effective Number of Clusters: EM-GMM vs Bayesian GMM
2. Concentration Parameter Sensitivity: α sweep
3. Posterior Uncertainty Visualization: Entropy-based analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import entropy

from .datasets import (
    ClusteringDataset,
    standardize_data,
    get_synthetic_datasets,
    generate_blobs,
)


@dataclass
class BayesianGMMResult:
    """Results from Bayesian GMM analysis."""
    weights: np.ndarray
    effective_k: int
    labels: np.ndarray
    responsibilities: np.ndarray
    ari: float
    nmi: float
    model: BayesianGaussianMixture


@dataclass
class EMGMMResult:
    """Results from EM-GMM analysis."""
    k: int
    labels: np.ndarray
    ari: float
    nmi: float
    bic: float
    aic: float
    model: GaussianMixture


def fit_em_gmm(
    X: np.ndarray,
    y_true: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> EMGMMResult:
    """
    Fit EM-GMM with specified number of components.
    
    Args:
        X: Feature matrix
        y_true: Ground truth labels
        n_components: Number of Gaussian components
        random_state: Random seed
    
    Returns:
        EMGMMResult with fitted model and metrics
    """
    model = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        n_init=5,
    )
    labels = model.fit_predict(X)
    
    return EMGMMResult(
        k=n_components,
        labels=labels,
        ari=adjusted_rand_score(y_true, labels),
        nmi=normalized_mutual_info_score(y_true, labels),
        bic=model.bic(X),
        aic=model.aic(X),
        model=model,
    )


def fit_bayesian_gmm(
    X: np.ndarray,
    y_true: np.ndarray,
    n_components: int = 20,
    weight_concentration_prior: float = 0.01,
    weight_concentration_prior_type: str = 'dirichlet_process',
    weight_threshold: float = 0.01,
    random_state: int = 42,
) -> BayesianGMMResult:
    """
    Fit Bayesian GMM with Dirichlet Process prior.
    
    Args:
        X: Feature matrix
        y_true: Ground truth labels
        n_components: Maximum number of components (Kmax)
        weight_concentration_prior: Concentration parameter α
        weight_concentration_prior_type: 'dirichlet_process' or 'dirichlet_distribution'
        weight_threshold: Threshold for counting effective clusters
        random_state: Random seed
    
    Returns:
        BayesianGMMResult with fitted model and metrics
    """
    model = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        weight_concentration_prior_type=weight_concentration_prior_type,
        weight_concentration_prior=weight_concentration_prior,
        random_state=random_state,
        n_init=3,
        max_iter=200,
    )
    
    labels = model.fit_predict(X)
    responsibilities = model.predict_proba(X)
    
    # Count effective clusters (weights above threshold)
    effective_k = np.sum(model.weights_ > weight_threshold)
    
    return BayesianGMMResult(
        weights=model.weights_,
        effective_k=effective_k,
        labels=labels,
        responsibilities=responsibilities,
        ari=adjusted_rand_score(y_true, labels),
        nmi=normalized_mutual_info_score(y_true, labels),
        model=model,
    )


def compute_point_entropy(responsibilities: np.ndarray) -> np.ndarray:
    """
    Compute entropy of posterior responsibilities for each data point.
    
    High entropy indicates uncertainty in cluster assignment.
    
    Args:
        responsibilities: Posterior probabilities (n_samples, n_components)
    
    Returns:
        Entropy values for each data point
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    probs = np.clip(responsibilities, eps, 1.0)
    
    # Compute entropy: -sum(p * log(p))
    point_entropy = -np.sum(probs * np.log(probs), axis=1)
    
    # Normalize by max possible entropy (uniform distribution)
    n_components = responsibilities.shape[1]
    max_entropy = np.log(n_components)
    
    return point_entropy / max_entropy  # Normalized to [0, 1]


# ========================================
# Bonus Experiment 1: Effective Number of Clusters
# ========================================

def run_bonus_experiment1(
    datasets: List[ClusteringDataset],
    k_range: range = range(2, 15),
    kmax_bayesian: int = 20,
    output_dir: Path = Path("outputs/bonus"),
    random_state: int = 42,
) -> Dict:
    """
    Bonus Experiment 1: Compare EM-GMM and Bayesian GMM on cluster number selection.
    
    Procedure:
    1. Fit EM-GMM with varying K
    2. Fit Bayesian GMM with large Kmax
    3. Compare learned mixture weights
    4. Count effective clusters
    """
    print("\n" + "=" * 70)
    print("Bonus Experiment 1: Effective Number of Clusters")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for dataset in datasets:
        print(f"\nDataset: {dataset.name} (True k={dataset.n_clusters})")
        print("-" * 50)
        
        X_scaled, _ = standardize_data(dataset.X)
        
        # ========================================
        # Part A: EM-GMM with varying K
        # ========================================
        print("  Fitting EM-GMM with varying K...")
        em_results = []
        for k in k_range:
            result = fit_em_gmm(X_scaled, dataset.y, k, random_state)
            em_results.append(result)
            print(f"    K={k}: ARI={result.ari:.4f}, BIC={result.bic:.1f}")
        
        # Find best K by BIC
        best_em = min(em_results, key=lambda r: r.bic)
        print(f"  EM-GMM best K (by BIC): {best_em.k}")
        
        # ========================================
        # Part B: Bayesian GMM
        # ========================================
        print(f"  Fitting Bayesian GMM (Kmax={kmax_bayesian})...")
        bayesian_result = fit_bayesian_gmm(
            X_scaled, dataset.y,
            n_components=kmax_bayesian,
            weight_concentration_prior=0.01,
            random_state=random_state,
        )
        print(f"  Bayesian GMM effective K: {bayesian_result.effective_k}")
        print(f"  Bayesian GMM ARI: {bayesian_result.ari:.4f}")
        
        results[dataset.name] = {
            'em_results': em_results,
            'bayesian_result': bayesian_result,
            'best_em': best_em,
            'true_k': dataset.n_clusters,
        }
        
        # ========================================
        # Visualizations
        # ========================================
        
        # Plot 1: EM-GMM model selection (BIC/AIC)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # BIC/AIC plot
        ax1 = axes[0]
        k_vals = [r.k for r in em_results]
        bic_vals = [r.bic for r in em_results]
        aic_vals = [r.aic for r in em_results]
        ax1.plot(k_vals, bic_vals, 'b-o', linewidth=2, label='BIC')
        ax1.plot(k_vals, aic_vals, 'r-s', linewidth=2, label='AIC')
        ax1.axvline(x=dataset.n_clusters, color='green', linestyle='--', 
                   label=f'True K={dataset.n_clusters}')
        ax1.axvline(x=best_em.k, color='blue', linestyle=':', 
                   label=f'Best BIC K={best_em.k}')
        ax1.set_xlabel('Number of Components (K)', fontsize=11)
        ax1.set_ylabel('Information Criterion', fontsize=11)
        ax1.set_title('EM-GMM Model Selection', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ARI comparison
        ax2 = axes[1]
        ari_vals = [r.ari for r in em_results]
        ax2.plot(k_vals, ari_vals, 'g-o', linewidth=2, label='EM-GMM ARI')
        ax2.axhline(y=bayesian_result.ari, color='purple', linestyle='--',
                   linewidth=2, label=f'Bayesian GMM ARI')
        ax2.axvline(x=dataset.n_clusters, color='green', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Components (K)', fontsize=11)
        ax2.set_ylabel('Adjusted Rand Index', fontsize=11)
        ax2.set_title('Clustering Quality (ARI)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bayesian GMM weights
        ax3 = axes[2]
        weights = bayesian_result.weights
        sorted_weights = np.sort(weights)[::-1]
        colors = ['steelblue' if w > 0.01 else 'lightgray' for w in sorted_weights]
        ax3.bar(range(1, len(weights) + 1), sorted_weights, color=colors, alpha=0.8)
        ax3.axhline(y=0.01, color='red', linestyle='--', label='Threshold (0.01)')
        ax3.set_xlabel('Component (sorted)', fontsize=11)
        ax3.set_ylabel('Mixture Weight', fontsize=11)
        ax3.set_title(f'Bayesian GMM Weights\n(Effective K={bayesian_result.effective_k})',
                     fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'EM-GMM vs Bayesian GMM: {dataset.name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / f"bonus1_effective_k_{dataset.name.lower().replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    
    # Create summary table
    summary_data = []
    for ds_name, data in results.items():
        row = {
            'Dataset': ds_name,
            'True K': data['true_k'],
            'EM-GMM Best K (BIC)': data['best_em'].k,
            'EM-GMM ARI': f"{data['best_em'].ari:.4f}",
            'Bayesian GMM Effective K': data['bayesian_result'].effective_k,
            'Bayesian GMM ARI': f"{data['bayesian_result'].ari:.4f}",
        }
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / "bonus1_summary.csv", index=False)
    
    print("\n" + "=" * 70)
    print("BONUS EXPERIMENT 1 SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    return results


# ========================================
# Bonus Experiment 2: Concentration Parameter Sensitivity
# ========================================

def run_bonus_experiment2(
    datasets: List[ClusteringDataset],
    alpha_range: np.ndarray = None,
    kmax: int = 20,
    output_dir: Path = Path("outputs/bonus"),
    random_state: int = 42,
) -> Dict:
    """
    Bonus Experiment 2: Analyze effect of concentration parameter α.
    
    Procedure:
    1. Use Dirichlet Process-style Bayesian GMM
    2. Sweep concentration parameter α
    3. Record effective number of clusters and ARI
    4. Plot Effective K vs α and ARI vs α
    """
    print("\n" + "=" * 70)
    print("Bonus Experiment 2: Concentration Parameter Sensitivity")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if alpha_range is None:
        # Log-spaced alpha values from very small to large
        alpha_range = np.logspace(-3, 2, 20)  # 0.001 to 100
    
    results = {}
    
    for dataset in datasets:
        print(f"\nDataset: {dataset.name} (True k={dataset.n_clusters})")
        print("-" * 50)
        
        X_scaled, _ = standardize_data(dataset.X)
        
        effective_k_list = []
        ari_list = []
        nmi_list = []
        
        for alpha in alpha_range:
            result = fit_bayesian_gmm(
                X_scaled, dataset.y,
                n_components=kmax,
                weight_concentration_prior=alpha,
                random_state=random_state,
            )
            effective_k_list.append(result.effective_k)
            ari_list.append(result.ari)
            nmi_list.append(result.nmi)
            
            print(f"  α={alpha:.4f}: Effective K={result.effective_k}, ARI={result.ari:.4f}")
        
        results[dataset.name] = {
            'alpha_range': alpha_range,
            'effective_k': np.array(effective_k_list),
            'ari': np.array(ari_list),
            'nmi': np.array(nmi_list),
            'true_k': dataset.n_clusters,
        }
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Effective K vs α
        ax1 = axes[0]
        ax1.semilogx(alpha_range, effective_k_list, 'b-o', linewidth=2, markersize=6)
        ax1.axhline(y=dataset.n_clusters, color='red', linestyle='--',
                   linewidth=2, label=f'True K={dataset.n_clusters}')
        ax1.set_xlabel('Concentration Parameter (α)', fontsize=12)
        ax1.set_ylabel('Effective Number of Clusters', fontsize=12)
        ax1.set_title('Effective K vs α', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ARI vs α
        ax2 = axes[1]
        ax2.semilogx(alpha_range, ari_list, 'g-o', linewidth=2, markersize=6, label='ARI')
        ax2.semilogx(alpha_range, nmi_list, 'm-s', linewidth=2, markersize=6, label='NMI')
        ax2.set_xlabel('Concentration Parameter (α)', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Clustering Quality vs α', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        # Find best α
        best_idx = np.argmax(ari_list)
        best_alpha = alpha_range[best_idx]
        ax1.axvline(x=best_alpha, color='green', linestyle=':', alpha=0.7,
                   label=f'Best α={best_alpha:.3f}')
        ax2.axvline(x=best_alpha, color='green', linestyle=':', alpha=0.7)
        ax1.legend()
        
        plt.suptitle(f'Concentration Parameter Sensitivity: {dataset.name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / f"bonus2_alpha_sensitivity_{dataset.name.lower().replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Concentration Parameter Effects")
    print("=" * 70)
    print("""
📊 Key Observations:

1. Small α (< 0.1):
   - Encourages fewer clusters
   - Prior concentrates probability on few components
   - May underfit if true K is large

2. Large α (> 10):
   - Encourages more clusters
   - More uniform prior over components
   - Approaches EM-GMM behavior

3. Optimal α:
   - Dataset-dependent
   - Typically in range [0.01, 1] for moderate cluster numbers
   - Different from geometry-based parameters (e.g., DBSCAN eps)
""")
    
    return results


# ========================================
# Bonus Experiment 3: Posterior Uncertainty Visualization
# ========================================

def run_bonus_experiment3(
    datasets: List[ClusteringDataset] = None,
    output_dir: Path = Path("outputs/bonus"),
    random_state: int = 42,
) -> Dict:
    """
    Bonus Experiment 3: Visualize posterior uncertainty via entropy.
    
    Procedure:
    1. Compute posterior responsibilities
    2. Calculate entropy per data point
    3. Visualize entropy on 2D toy datasets
    
    High entropy indicates ambiguous/boundary points.
    """
    print("\n" + "=" * 70)
    print("Bonus Experiment 3: Posterior Uncertainty Visualization")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use 2D synthetic datasets for visualization
    if datasets is None:
        datasets = get_synthetic_datasets(random_state=random_state)
    
    results = {}
    
    for dataset in datasets:
        if dataset.X.shape[1] != 2:
            print(f"  Skipping {dataset.name} (not 2D)")
            continue
            
        print(f"\nDataset: {dataset.name}")
        print("-" * 50)
        
        X_scaled, _ = standardize_data(dataset.X)
        
        # Fit Bayesian GMM
        bayesian_result = fit_bayesian_gmm(
            X_scaled, dataset.y,
            n_components=15,
            weight_concentration_prior=0.1,
            random_state=random_state,
        )
        
        # Compute point-wise entropy
        point_entropy = compute_point_entropy(bayesian_result.responsibilities)
        
        results[dataset.name] = {
            'X': X_scaled,
            'y_true': dataset.y,
            'labels': bayesian_result.labels,
            'entropy': point_entropy,
            'responsibilities': bayesian_result.responsibilities,
            'effective_k': bayesian_result.effective_k,
        }
        
        print(f"  Effective K: {bayesian_result.effective_k}")
        print(f"  Mean entropy: {np.mean(point_entropy):.4f}")
        print(f"  High uncertainty points (entropy > 0.5): {np.sum(point_entropy > 0.5)}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        ax1 = axes[0]
        scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dataset.y,
                              cmap='tab10', s=30, alpha=0.7)
        ax1.set_title(f'Ground Truth\n({dataset.n_clusters} clusters)',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        
        # Bayesian GMM clustering
        ax2 = axes[1]
        scatter2 = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=bayesian_result.labels,
                              cmap='tab10', s=30, alpha=0.7)
        ax2.set_title(f'Bayesian GMM Clustering\n(Effective K={bayesian_result.effective_k})',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        
        # Entropy visualization
        ax3 = axes[2]
        scatter3 = ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], c=point_entropy,
                              cmap='YlOrRd', s=30, alpha=0.8, vmin=0, vmax=1)
        cbar = plt.colorbar(scatter3, ax=ax3)
        cbar.set_label('Normalized Entropy', fontsize=10)
        ax3.set_title('Posterior Uncertainty\n(High entropy = ambiguous)',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Feature 1')
        ax3.set_ylabel('Feature 2')
        
        plt.suptitle(f'Posterior Uncertainty Analysis: {dataset.name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / f"bonus3_uncertainty_{dataset.name.lower().replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    
    # Create combined visualization
    n_datasets = len([d for d in datasets if d.X.shape[1] == 2])
    if n_datasets > 0:
        fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 4 * n_datasets))
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        row_idx = 0
        for dataset in datasets:
            if dataset.X.shape[1] != 2:
                continue
            
            data = results[dataset.name]
            X = data['X']
            
            # Ground truth
            axes[row_idx, 0].scatter(X[:, 0], X[:, 1], c=data['y_true'],
                                    cmap='tab10', s=20, alpha=0.7)
            axes[row_idx, 0].set_title(f'{dataset.name}: Ground Truth')
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])
            
            # Clustering
            axes[row_idx, 1].scatter(X[:, 0], X[:, 1], c=data['labels'],
                                    cmap='tab10', s=20, alpha=0.7)
            axes[row_idx, 1].set_title(f'Bayesian GMM (K={data["effective_k"]})')
            axes[row_idx, 1].set_xticks([])
            axes[row_idx, 1].set_yticks([])
            
            # Entropy
            sc = axes[row_idx, 2].scatter(X[:, 0], X[:, 1], c=data['entropy'],
                                         cmap='YlOrRd', s=20, alpha=0.8, vmin=0, vmax=1)
            axes[row_idx, 2].set_title('Uncertainty (Entropy)')
            axes[row_idx, 2].set_xticks([])
            axes[row_idx, 2].set_yticks([])
            
            row_idx += 1
        
        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Normalized Entropy', fontsize=10)
        
        plt.suptitle('Posterior Uncertainty Across Datasets', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        save_path = output_dir / "bonus3_uncertainty_all_datasets.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved combined visualization: {save_path}")
        plt.close(fig)
    
    print("\n" + "=" * 70)
    print("ANALYSIS: Posterior Uncertainty")
    print("=" * 70)
    print("""
📊 Interpretation:

1. Low Entropy (Yellow):
   - High confidence in cluster assignment
   - Points clearly belong to one cluster

2. High Entropy (Red):
   - Uncertain cluster assignment
   - Typically boundary points or overlapping regions
   - Model acknowledges ambiguity

3. Bayesian Advantage:
   - Probabilistic assignments capture uncertainty
   - Hard clustering (K-Means) cannot express this
   - Useful for downstream decision-making
""")
    
    return results


# ========================================
# Main Bonus Experiment Runner
# ========================================

def run_all_bonus_experiments(
    output_dir: Path = Path("outputs/bonus"),
    random_state: int = 42,
) -> Dict:
    """
    Run all bonus experiments for Bayesian clustering.
    
    Args:
        output_dir: Directory to save outputs
        random_state: Random seed
    
    Returns:
        Dictionary containing all results
    """
    print("\n" + "=" * 70)
    print("BONUS: Bayesian Gaussian Mixture Model Experiments")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get synthetic datasets
    datasets = get_synthetic_datasets(random_state=random_state)
    
    results = {}
    
    # Bonus Experiment 1: Effective Number of Clusters
    results['experiment1'] = run_bonus_experiment1(
        datasets=datasets,
        output_dir=output_dir,
        random_state=random_state,
    )
    
    # Bonus Experiment 2: Concentration Parameter Sensitivity
    results['experiment2'] = run_bonus_experiment2(
        datasets=datasets,
        output_dir=output_dir,
        random_state=random_state,
    )
    
    # Bonus Experiment 3: Posterior Uncertainty Visualization
    results['experiment3'] = run_bonus_experiment3(
        datasets=datasets,
        output_dir=output_dir,
        random_state=random_state,
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("BONUS EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"""
✓ All bonus experiment results saved to: {output_dir}/

📁 Generated Files:
  - bonus1_effective_k_*.png: EM-GMM vs Bayesian GMM comparison
  - bonus1_summary.csv: Effective K comparison table
  - bonus2_alpha_sensitivity_*.png: Concentration parameter analysis
  - bonus3_uncertainty_*.png: Posterior entropy visualization
  - bonus3_uncertainty_all_datasets.png: Combined uncertainty view

📝 Key Takeaways:

1. Bayesian GMM automatically infers cluster number:
   - No need to manually specify K
   - Dirichlet Process prior controls complexity

2. Concentration parameter α controls model complexity:
   - Small α → fewer clusters
   - Large α → more clusters
   - Optimal α is dataset-dependent

3. Posterior uncertainty provides valuable information:
   - Identifies ambiguous/boundary points
   - Enables principled decision-making under uncertainty
   - Not available in hard clustering methods
""")
    
    return results
