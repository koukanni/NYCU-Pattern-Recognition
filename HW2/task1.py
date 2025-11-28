import os
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data_loader import load_dataset
from utils import k_fold_cross_validation, evaluate_classifier


class LDA:
    def __init__(self, n_components=None, priors=None):
        self.model = LinearDiscriminantAnalysis(n_components=n_components, priors=priors)
        logger.info("Initialized Linear Discriminant Analysis Classifier.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.success("Model training completed.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(X)
        y_scores = self.model.predict_proba(X)
        return y_pred, y_scores


def compute_separability(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the separability measure before projection using Fisher's criterion.

    Separability is computed as the ratio of between-class scatter to within-class scatter.
    For multi-class problems, we use the trace of S_B @ inv(S_W) where:
    - S_B is the between-class scatter matrix
    - S_W is the within-class scatter matrix

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)

    Returns:
        separability: A scalar value representing the separability measure
    """
    classes = np.unique(y)
    n_features = X.shape[1]

    # Compute overall mean
    overall_mean = np.mean(X, axis=0)

    # Initialize scatter matrices
    S_W = np.zeros((n_features, n_features))  # Within-class scatter
    S_B = np.zeros((n_features, n_features))  # Between-class scatter

    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]

        # Class mean
        mean_c = np.mean(X_c, axis=0)

        # Within-class scatter for class c
        S_W += (X_c - mean_c).T @ (X_c - mean_c)

        # Between-class scatter for class c
        mean_diff = (mean_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)

    # Compute separability as trace(S_B @ inv(S_W))
    # Add regularization to avoid singular matrix
    regularization = 1e-6
    S_W_reg = S_W + regularization * np.eye(n_features)

    try:
        separability = np.trace(np.linalg.solve(S_W_reg, S_B))
    except np.linalg.LinAlgError:
        logger.warning(
            "Could not compute separability due to singular matrix. Returning 0."
        )
        separability = 0.0

    return separability


def split_data(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets with stratification and feature scaling.
    """

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logger.success("Feature scaling completed.\n")

    return X_train, X_test, y_train, y_test


def baseline(X: np.ndarray, y: np.ndarray, dataset: str):
    # split the dataset
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42
    )

    separability_before = compute_separability(X_train, y_train)

    # Apply LDA
    lda = LDA(n_components=None)
    lda.fit(X_train, y_train)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    separability_after = compute_separability(X_train_lda, y_train)

    y_pred, y_score = lda.predict(X_test)
    evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_scores=y_score,
        dataset_name=dataset,
        classifier_name="LDA",
        results_dir="results/task1/baseline",
    )

    logger.info(f"--- Task 1 ({dataset}) completed ---")
    logger.info("Summary:")
    logger.info(f"Separability before projection: {separability_before:.4f}")
    logger.info(f"Separability after projection: {separability_after:.4f}")
    logger.info("ROC/AUC plots saved to 'results/task1/baseline' dtirectory.")


def dimension_sweep(X: np.ndarray, y: np.ndarray, dataset: str):
    """
    Sweep LDA projection dimensions d = 1, ..., C-1 and
    record separability and test accuracy for each d.
    Results are saved to results/task1/dim_sweep/{dataset}_dim_sweep.csv
    """

    # split the dataset
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42
    )

    classes = np.unique(y_train)
    C = len(classes)
    max_dim = min(C - 1, X_train.shape[1])

    if max_dim < 1:
        logger.error(
            f"dim_sweep not applicable: C={C}, features={X_train.shape[1]}."
        )
        return

    logger.info(
        f"[{dataset}] dim_sweep: number of classes = {C}, "
        f"feature dim = {X_train.shape[1]}, max_dim = {max_dim}"
    )

    results = []

    for d in range(1, max_dim + 1):
        lda = LDA(n_components=d)
        lda.fit(X_train, y_train)

        # Project to LDA subspace for computing separability
        X_train_lda = lda.transform(X_train)
        sep_d = compute_separability(X_train_lda, y_train)

        # Classification: use LDA's decision rule in the original (scaled) space
        y_pred, y_score = lda.predict(X_test)
        acc_d = accuracy_score(y_test, y_pred)

        logger.info(
            f"[{dataset}] dim={d}: separability={sep_d:.4f}, "
            f"accuracy={acc_d:.4f}"
        )

        results.append(
            {
                "dim": d,
                "separability": sep_d,
                "accuracy": acc_d,
            }
        )

    import os

    os.makedirs("results/task1/dim_sweep", exist_ok=True)
    df = pd.DataFrame(results)
    out_path = f"results/task1/dim_sweep/{dataset}_dim_sweep.csv"
    df.to_csv(out_path, index=False)

    logger.success(
        f"[{dataset}] dim_sweep finished. Results saved to '{out_path}'."
    )


def prior_experiment(X: np.ndarray, y: np.ndarray, dataset: str):
    """
    Compare LDA with empirical priors vs balanced priors on an imbalanced dataset
    (here, Taiwanese Bankruptcy).

    Results (accuracy, balanced accuracy, precision, recall, F1, AUC) are saved to
    results/task1/prior/{dataset}_prior.json
    """

    classes, counts = np.unique(y, return_counts=True)
    C = len(classes)
    if C != 2:
        logger.error(
            f"prior_experiment is designed for binary datasets, got C={C}."
        )
        return

    logger.info(
        f"[{dataset}] prior_experiment: classes={classes}, counts={counts}"
    )

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42
    )

    # 1. empirical prior
    lda_emp = LDA(n_components=None, priors=None)
    lda_emp.fit(X_train, y_train)
    y_pred_emp, y_score_emp = lda_emp.predict(X_test)

    # 2. balanced prior
    balanced_prior = np.ones(C) / C
    lda_bal = LDA(n_components=None, priors=balanced_prior)
    lda_bal.fit(X_train, y_train)
    y_pred_bal, y_score_bal = lda_bal.predict(X_test)

    metrics = {}

    metrics["empirical"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_emp)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_emp)),
        "precision": float(precision_score(y_test, y_pred_emp)),
        "recall": float(recall_score(y_test, y_pred_emp)),
        "f1": float(f1_score(y_test, y_pred_emp)),
        "auc": float(roc_auc_score(y_test, y_score_emp[:, 1])),
    }

    metrics["balanced"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_bal)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_bal)),
        "precision": float(precision_score(y_test, y_pred_bal)),
        "recall": float(recall_score(y_test, y_pred_bal)),
        "f1": float(f1_score(y_test, y_pred_bal)),
        "auc": float(roc_auc_score(y_test, y_score_bal[:, 1])),
    }

    # save results
    os.makedirs("results/task1/prior", exist_ok=True)
    out_path = f"results/task1/prior/{dataset}_prior.json"

    import json

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[{dataset}] prior_experiment metrics:\n{metrics}")
    logger.success(
        f"[{dataset}] prior_experiment finished. Results saved to '{out_path}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Classifier with dimension reduction techniques"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["telescope", "bankruptcy", "cover_type", "dry_bean"],
        help="Dataset to load (e.g., telescope, bankruptcy, cover_type, dry_bean)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "dim_sweep", "prior"],
        help="Experiment mode: baseline or dimension sweep or prior",
    )

    args = parser.parse_args()

    # Load dataset
    X, y = load_dataset(args.dataset)
    logger.info(
        f"Loaded {args.dataset} dataset: X shape={X.shape}, y shape={y.shape}\n"
    )

    # run the selected experiment mode
    if args.mode == "baseline":
        baseline(X, y, args.dataset)

    elif args.mode == "dim_sweep":
        dimension_sweep(X, y, args.dataset)

    elif args.mode == "prior":
        if args.dataset != "bankruptcy":
            logger.error(
                "Prior mode is only applicable to the bankruptcy dataset."
            )
            exit(1)

        prior_experiment(X, y, args.dataset)
