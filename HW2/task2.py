import argparse
from pathlib import Path
from typing import Tuple, Literal, Dict

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from data_loader import load_dataset


DatasetName = Literal["bankruptcy", "dry_bean"]


# Utils
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PR HW2 - Task 2: PCA experiments"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bankruptcy", "dry_bean"],
        required=True,
        help="Which dataset to run on.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "explained_variance",  # Experiment 1
            "dim_sweep",  # Experiment 2
            "clf_compare",  # Experiment 3
        ],
        required=True,
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/task2",
        help="Directory to save CSVs / logs for the report.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio (for train/val/test splitting).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.25,
        help="Validation ratio within train+val "
        "(e.g. 0.25 means train:val:test = 0.6:0.2:0.2).",
    )
    return parser.parse_args()


def ensure_output_dir(path_str: str) -> Path:
    out_dir = Path(path_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Split into train/val/test and apply StandardScaler fitted on train only.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
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
) -> Dict[str, float]:
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)

    return {
        "acc": accuracy_score(y_val, y_pred),
        "balanced_acc": balanced_accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
    }


# Experiment 1:
# Explained variance curve
def run_experiment_explained_variance(
    dataset: DatasetName,
    args: argparse.Namespace,
) -> None:
    logger.info(f"[Experiment 1] Explained variance on dataset = {dataset}")

    X, y = load_dataset(dataset)
    logger.info(
        f"Loaded dataset {dataset}: X shape={X.shape}, y shape={y.shape}"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = X_scaled.shape[1]
    pca = PCA(n_components=n_features)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    out_dir = ensure_output_dir(args.output_dir)
    out_csv = out_dir / f"explained_variance_{dataset}.csv"

    import csv

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "explained_var_ratio", "cumulative_var_ratio"])
        for k, (e, c) in enumerate(zip(explained, cumulative), start=1):
            writer.writerow([k, e, c])

    logger.success(f"Saved explained variance curve to: {out_csv}")


# Experiment 2:
# Dimension sweep on Bankruptcy + LogReg
def run_experiment_dim_sweep(
    dataset: DatasetName,
    args: argparse.Namespace,
) -> None:
    if dataset != "bankruptcy":
        logger.warning(
            "Experiment 2 (dim_sweep) is mainly designed for the "
            "Taiwanese Bankruptcy dataset. You are running it on "
            f"{dataset}, which is fine but less emphasized in the report."
        )

    logger.info(f"[Experiment 2] PCA dimension sweep on dataset = {dataset}")

    X, y = load_dataset(dataset)
    logger.info(
        f"Loaded dataset {dataset}: X shape={X.shape}, y shape={y.shape}"
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    n_features = X_train.shape[1]

    candidate_dims = [2, 5, 10, 20, 30, 40, 60, 80]
    candidate_dims = [k for k in candidate_dims if k <= n_features]

    out_dir = ensure_output_dir(args.output_dir)
    out_csv = out_dir / f"dim_sweep_{dataset}.csv"

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

        # Baseline: no PCA
        logger.info("Running baseline (no PCA)...")
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
            f"Baseline (no PCA) - "
            f"val_acc={metrics_val['acc']:.4f}, "
            f"val_balanced_acc={metrics_val['balanced_acc']:.4f}"
        )

        # PCA + LogReg for each k
        for k in candidate_dims:
            logger.info(f"Running PCA + LogReg with k={k}...")
            pca = PCA(n_components=k, random_state=args.random_state)
            X_train_pca = pca.fit_transform(X_train)
            X_val_pca = pca.transform(X_val)
            X_test_pca = pca.transform(X_test)

            clf = build_logreg(args.random_state)
            metrics_val = evaluate_clf(
                clf, X_train_pca, y_train, X_val_pca, y_val
            )
            metrics_test = evaluate_clf(
                clf,
                np.vstack([X_train_pca, X_val_pca]),
                np.hstack([y_train, y_val]),
                X_test_pca,
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
                f"k={k}: val_acc={metrics_val['acc']:.4f}, "
                f"val_balanced_acc={metrics_val['balanced_acc']:.4f}"
            )

    logger.success(f"Saved dimension sweep results to: {out_csv}")


# Experiment 3:
# Compare PCA effects across classifiers & datasets
def run_experiment_classifier_compare(
    dataset: DatasetName,
    args: argparse.Namespace,
) -> None:
    logger.info(f"[Experiment 3] Classifier comparison on dataset = {dataset}")

    X, y = load_dataset(dataset)
    logger.info(
        f"Loaded dataset {dataset}: X shape={X.shape}, y shape={y.shape}"
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    out_dir = ensure_output_dir(args.output_dir)
    out_csv = out_dir / f"classifier_compare_{dataset}.csv"

    import csv

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "classifier",
                "use_pca",
                "pca_dim",
                "val_acc",
                "val_balanced_acc",
                "val_f1_macro",
                "test_acc",
                "test_balanced_acc",
                "test_f1_macro",
            ]
        )

        # 1. Baseline: LogReg / RBF-SVM without PCA
        for clf_name in ["logreg", "rbf_svm"]:
            logger.info(f"Running baseline without PCA for {clf_name}...")

            if clf_name == "logreg":
                clf = build_logreg(args.random_state)
            else:
                clf = build_rbf_svm(args.random_state)

            metrics_val = evaluate_clf(clf, X_train, y_train, X_val, y_val)
            metrics_test = evaluate_clf(
                clf,
                np.vstack([X_train, X_val]),
                np.hstack([y_train, y_val]),
                X_test,
                y_test,
            )

            writer.writerow(
                [
                    clf_name,
                    0,
                    0,
                    metrics_val["acc"],
                    metrics_val["balanced_acc"],
                    metrics_val["f1_macro"],
                    metrics_test["acc"],
                    metrics_test["balanced_acc"],
                    metrics_test["f1_macro"],
                ]
            )

        # 2. PCA + classifier
        logger.info("Fitting PCA (95% variance) for classifier comparison...")
        pca = PCA(
            n_components=0.95, svd_solver="full", random_state=args.random_state
        )
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        k_eff = X_train_pca.shape[1]

        logger.info(f"Effective PCA dimension for 95% variance: k={k_eff}")

        for clf_name in ["logreg", "rbf_svm"]:
            logger.info(f"Running {clf_name} with PCA (k={k_eff})...")

            if clf_name == "logreg":
                clf = build_logreg(args.random_state)
            else:
                clf = build_rbf_svm(args.random_state)

            metrics_val = evaluate_clf(
                clf, X_train_pca, y_train, X_val_pca, y_val
            )
            metrics_test = evaluate_clf(
                clf,
                np.vstack([X_train_pca, X_val_pca]),
                np.hstack([y_train, y_val]),
                X_test_pca,
                y_test,
            )

            writer.writerow(
                [
                    clf_name,
                    1,
                    k_eff,
                    metrics_val["acc"],
                    metrics_val["balanced_acc"],
                    metrics_val["f1_macro"],
                    metrics_test["acc"],
                    metrics_test["balanced_acc"],
                    metrics_test["f1_macro"],
                ]
            )

    logger.success(f"Saved classifier comparison results to: {out_csv}")


if __name__ == "__main__":
    args = parse_args()

    logger.info(
        f"Running Task 2 with dataset={args.dataset}, "
        f"experiment={args.experiment}"
    )

    if args.experiment == "explained_variance":
        run_experiment_explained_variance(args.dataset, args)
    elif args.experiment == "dim_sweep":
        run_experiment_dim_sweep(args.dataset, args)
    elif args.experiment == "clf_compare":
        run_experiment_classifier_compare(args.dataset, args)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
