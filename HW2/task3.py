#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from data_loader import load_dataset


DatasetName = Literal["bankruptcy", "dry_bean", "telescope", "cover_type"]


# Utils
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PR HW2 - Task 3: Nonlinear dimensionality reduction (Isomap) experiments"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bankruptcy", "dry_bean", "telescope", "cover_type"],
        default="dry_bean",
        help="Which dataset to run on (default: dry_bean).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "embedding_viz",  # Experiment 1: 2D embedding visualization
            "dim_sweep_isomap",  # Experiment 2: Isomap dimension sweep (to be implemented)
            "neighbor_sensitivity",  # Experiment 3: n_neighbors sensitivity (to be implemented)
        ],
        required=True,
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/task3",
        help="Directory to save figures / CSVs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for Isomap (default: 10).",
    )
    return parser.parse_args()


def ensure_output_dir(path_str: str) -> Path:
    out_dir = Path(path_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
):
    """
    Split into train/val/test and apply StandardScaler fitted on train only.
    We use the same 60/20/20 split protocol as in Task 2:
    - First split: train+val vs test = 0.8 / 0.2
    - Second split: train vs val = 0.75 / 0.25 of (train+val)
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,  # 0.8 * 0.25 = 0.2  =>  train:val:test = 0.6:0.2:0.2
        random_state=random_state,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_logreg(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )


def build_rbf_svm(random_state: int) -> SVC:
    return SVC(
        kernel="rbf",
        probability=False,
        random_state=random_state,
    )


def evaluate_clf(
    clf,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)

    return {
        "acc": accuracy_score(y_val, y_pred),
        "balanced_acc": balanced_accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
    }


# Experiment 1:
# 2D Embedding Visualization (PCA vs Isomap)
def run_experiment_embedding_viz(
    dataset: DatasetName,
    args: argparse.Namespace,
) -> None:
    """
    Experiment 1:
    Compare 2D embeddings produced by PCA and Isomap on a given dataset.
    This is mainly designed for Dry Bean, but can be run on other datasets as well.
    """
    if dataset != "dry_bean":
        logger.warning(
            "Experiment 1 (embedding_viz) is primarily designed for the Dry Bean dataset, "
            f"but you are running it on {dataset}."
        )

    logger.info(
        f"[Experiment 1] 2D embedding visualization on dataset = {dataset}"
    )

    # Load full dataset (no split; this is for visualization)
    X, y = load_dataset(dataset)
    logger.info(
        f"Loaded dataset {dataset}: X shape={X.shape}, y shape={y.shape}"
    )

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2D PCA
    logger.info("Fitting PCA (2D)...")
    pca = PCA(n_components=2, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)

    # 2D Isomap
    logger.info(f"Fitting Isomap (2D) with n_neighbors={args.n_neighbors}...")
    iso = Isomap(
        n_components=2,
        n_neighbors=args.n_neighbors,
    )
    X_iso = iso.fit_transform(X_scaled)

    out_dir = ensure_output_dir(args.output_dir)

    # Save the embeddings as CSV for later plotting or analysis
    import csv

    pca_csv = out_dir / f"embedding_{dataset}_pca2.csv"
    iso_csv = out_dir / f"embedding_{dataset}_isomap2_k{args.n_neighbors}.csv"

    logger.info(f"Saving PCA 2D embedding to: {pca_csv}")
    with pca_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["z1", "z2", "label"])
        for (z1, z2), label in zip(X_pca, y):
            writer.writerow([z1, z2, label])

    logger.info(f"Saving Isomap 2D embedding to: {iso_csv}")
    with iso_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["z1", "z2", "label"])
        for (z1, z2), label in zip(X_iso, y):
            writer.writerow([z1, z2, label])

    # Optionally, generate a side-by-side scatter plot for report
    try:
        import matplotlib.pyplot as plt

        logger.info("Generating 2D scatter plots for PCA and Isomap...")

        classes = np.unique(y)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # PCA plot
        ax = axes[0]
        for c in classes:
            idx = y == c
            ax.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                s=5,
                alpha=0.6,
                label=str(c),
            )
        ax.set_title("PCA (2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        # Isomap plot
        ax = axes[1]
        for c in classes:
            idx = y == c
            ax.scatter(
                X_iso[idx, 0],
                X_iso[idx, 1],
                s=5,
                alpha=0.6,
                label=str(c),
            )
        ax.set_title(f"Isomap (2D, k={args.n_neighbors})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        # Legend: 只在右圖畫一次，避免擠爆
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(classes), 6),
            fontsize=8,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.90])

        fig_path_pdf = out_dir / f"task3_{dataset}_pca_isomap_2d.pdf"
        fig_path_png = out_dir / f"task3_{dataset}_pca_isomap_2d.png"

        fig.savefig(fig_path_pdf)
        fig.savefig(fig_path_png, dpi=300)

        plt.close(fig)

        logger.success(
            f"Saved 2D embedding figure to: {fig_path_pdf} and {fig_path_png}"
        )

    except ImportError:
        logger.warning(
            "matplotlib is not available; skipped figure generation. "
            "You can still use the saved CSV embeddings for plotting."
        )


# Experiment 2:
# Isomap Dimension Sweep
def run_experiment_dim_sweep_isomap(
    dataset: DatasetName,
    args: argparse.Namespace,
) -> None:
    """
    Isomap dimension sweep experiment.

    - Split dataset into train/val/test using the same 60/20/20 protocol as Task 2.
    - For a range of Isomap dimensions (e.g., k ∈ {2, 5, 10, 20, 30}),
      fit Isomap on the training data (after standardization) and
      evaluate logistic regression on val/test.
    - Save all metrics to CSV.
    """
    if dataset != "bankruptcy":
        logger.warning(
            "Experiment 'dim_sweep_isomap' is mainly designed for the "
            "Taiwanese Bankruptcy dataset, but you are running it on "
            f"{dataset}."
        )

    logger.info(f"[Experiment 2] Isomap dimension sweep on dataset = {dataset}")

    # 1. Load dataset
    X, y = load_dataset(dataset)
    logger.info(f"Loaded dataset {dataset}: X shape={X.shape}, y shape={y.shape}")

    # 2. Split + scale
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(
        X,
        y,
        random_state=args.random_state,
    )

    n_features = X_train.shape[1]

    # 3. Candidate Isomap dimensions (bounded by #features)
    candidate_dims = [2, 5, 10, 20, 30]
    candidate_dims = [k for k in candidate_dims if k <= n_features]

    out_dir = ensure_output_dir(args.output_dir)
    out_csv = out_dir / f"dim_sweep_isomap_{dataset}.csv"

    import csv

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "val_acc",
                "val_balanced_acc",
                "val_f1_macro",
                "test_acc",
                "test_balanced_acc",
                "test_f1_macro",
            ]
        )

        # 4. Baseline: no dimensionality reduction (just logistic regression on scaled space)
        logger.info("Running baseline (no dimensionality reduction)...")
        clf_base = build_logreg(args.random_state)
        metrics_val = evaluate_clf(clf_base, X_train, y_train, X_val, y_val)
        metrics_test = evaluate_clf(
            clf_base,
            np.vstack([X_train, X_val]),
            np.hstack([y_train, y_val]),
            X_test,
            y_test,
        )

        writer.writerow(
            [
                0,
                metrics_val["acc"],
                metrics_val["balanced_acc"],
                metrics_val["f1_macro"],
                metrics_test["acc"],
                metrics_test["balanced_acc"],
                metrics_test["f1_macro"],
            ]
        )

        logger.success(
            "Baseline (no DR) - "
            f"val_bal_acc={metrics_val['balanced_acc']:.4f}, "
            f"test_bal_acc={metrics_test['balanced_acc']:.4f}"
        )

        # 5. For each Isomap dimension, fit Isomap + LogReg
        for k in candidate_dims:
            logger.info(
                f"Running Isomap + Logistic Regression with n_components={k}, "
                f"n_neighbors={args.n_neighbors}..."
            )

            iso = Isomap(
                n_components=k,
                n_neighbors=args.n_neighbors,
            )

            # Fit Isomap on training data only, then transform val/test
            X_train_iso = iso.fit_transform(X_train)
            X_val_iso = iso.transform(X_val)
            X_test_iso = iso.transform(X_test)

            clf = build_logreg(args.random_state)

            metrics_val = evaluate_clf(
                clf,
                X_train_iso,
                y_train,
                X_val_iso,
                y_val,
            )
            metrics_test = evaluate_clf(
                clf,
                np.vstack([X_train_iso, X_val_iso]),
                np.hstack([y_train, y_val]),
                X_test_iso,
                y_test,
            )

            writer.writerow(
                [
                    k,
                    metrics_val["acc"],
                    metrics_val["balanced_acc"],
                    metrics_val["f1_macro"],
                    metrics_test["acc"],
                    metrics_test["balanced_acc"],
                    metrics_test["f1_macro"],
                ]
            )

            logger.success(
                f"k={k}: val_bal_acc={metrics_val['balanced_acc']:.4f}, "
                f"test_bal_acc={metrics_test['balanced_acc']:.4f}"
            )

    logger.success(f"Saved Isomap dimension sweep results to: {out_csv}")


# Experiment 3:
def run_experiment_neighbor_sensitivity(
    dataset: DatasetName,
    args: argparse.Namespace,
) -> None:
    """
    Hyperparameter sensitivity experiment for Isomap.

    - Fix Isomap output dimension (n_components = 10).
    - Vary n_neighbors (e.g., {5, 10, 20, 50}).
    - Evaluate performance of an RBF SVM classifier on val/test for each setting.
    - Also include a baseline RBF SVM without Isomap.
    - Save all metrics to CSV.
    """
    logger.info(
        f"[Experiment 3] Isomap neighbor sensitivity on dataset = {dataset}"
    )

    # 1. Load dataset
    X, y = load_dataset(dataset)
    logger.info(f"Loaded dataset {dataset}: X shape={X.shape}, y shape={y.shape}")

    # 2. Split + scale (same 60/20/20 protocol)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(
        X,
        y,
        random_state=args.random_state,
    )

    out_dir = ensure_output_dir(args.output_dir)
    out_csv = out_dir / f"neighbor_sensitivity_isomap_{dataset}.csv"

    # Choose a reasonable fixed output dimension for Isomap
    n_components = min(10, X_train.shape[1])

    # Candidate neighbor sizes
    candidate_neighbors = [5, 10, 20, 50]

    import csv

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "use_isomap",
                "n_components",
                "n_neighbors",
                "val_acc",
                "val_balanced_acc",
                "val_f1_macro",
                "test_acc",
                "test_balanced_acc",
                "test_f1_macro",
            ]
        )

        # 3. Baseline: RBF SVM without Isomap
        logger.info("Running baseline RBF SVM (no Isomap)...")
        clf_base = build_rbf_svm(args.random_state)
        metrics_val = evaluate_clf(clf_base, X_train, y_train, X_val, y_val)
        metrics_test = evaluate_clf(
            clf_base,
            np.vstack([X_train, X_val]),
            np.hstack([y_train, y_val]),
            X_test,
            y_test,
        )

        writer.writerow(
            [
                0,          # use_isomap = 0
                0,          # n_components
                0,          # n_neighbors
                metrics_val["acc"],
                metrics_val["balanced_acc"],
                metrics_val["f1_macro"],
                metrics_test["acc"],
                metrics_test["balanced_acc"],
                metrics_test["f1_macro"],
            ]
        )

        logger.success(
            "Baseline RBF SVM - "
            f"val_bal_acc={metrics_val['balanced_acc']:.4f}, "
            f"test_bal_acc={metrics_test['balanced_acc']:.4f}"
        )

        # 4. Isomap + RBF SVM for each neighbor setting
        for k_nn in candidate_neighbors:
            logger.info(
                f"Running Isomap (n_components={n_components}, "
                f"n_neighbors={k_nn}) + RBF SVM..."
            )

            iso = Isomap(
                n_components=n_components,
                n_neighbors=k_nn,
            )

            # Fit Isomap on training data only
            X_train_iso = iso.fit_transform(X_train)
            X_val_iso = iso.transform(X_val)
            X_test_iso = iso.transform(X_test)

            clf = build_rbf_svm(args.random_state)

            metrics_val = evaluate_clf(
                clf,
                X_train_iso,
                y_train,
                X_val_iso,
                y_val,
            )
            metrics_test = evaluate_clf(
                clf,
                np.vstack([X_train_iso, X_val_iso]),
                np.hstack([y_train, y_val]),
                X_test_iso,
                y_test,
            )

            writer.writerow(
                [
                    1,              # use_isomap = 1
                    n_components,   # n_components
                    k_nn,           # n_neighbors
                    metrics_val["acc"],
                    metrics_val["balanced_acc"],
                    metrics_val["f1_macro"],
                    metrics_test["acc"],
                    metrics_test["balanced_acc"],
                    metrics_test["f1_macro"],
                ]
            )

            logger.success(
                f"n_neighbors={k_nn}: "
                f"val_bal_acc={metrics_val['balanced_acc']:.4f}, "
                f"test_bal_acc={metrics_test['balanced_acc']:.4f}"
            )

    logger.success(
        f"Saved Isomap neighbor sensitivity results to: {out_csv}"
    )


if __name__ == "__main__":
    args = parse_args()

    logger.info(
        f"Running Task 3 with dataset={args.dataset}, "
        f"experiment={args.experiment}, "
        f"n_neighbors={args.n_neighbors}"
    )

    if args.experiment == "embedding_viz":
        run_experiment_embedding_viz(args.dataset, args)
    elif args.experiment == "dim_sweep_isomap":
        run_experiment_dim_sweep_isomap(args.dataset, args)
    elif args.experiment == "neighbor_sensitivity":
        run_experiment_neighbor_sensitivity(args.dataset, args)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
