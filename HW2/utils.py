import os
import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def k_fold_cross_validation(classifier_class, X, y, k=5, random_state=42):
    """
    Perform k-fold cross validation on the given classifier.

    Args:
        classifier_class (class): The classifier class to instantiate.
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        k (int): Number of folds
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing fold results and average metrics
    """
    logger.info(
        f"Starting {k}-fold cross validation for {classifier_class.__name__}..."
    )

    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx}/{k}")
        logger.info(f"{'='*60}")

        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(
            f"Train size: {len(X_train)}, Validation size: {len(X_val)}"
        )

        classifier = classifier_class()
        classifier.train(X_train, y_train)

        # Make predictions
        y_pred, y_scores = classifier.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        logger.success(f"Fold {fold_idx} Accuracy: {accuracy:.4f}")

        # Store results
        fold_results.append(
            {
                "fold": fold_idx,
                "accuracy": accuracy,
                "y_true": y_val,
                "y_pred": y_pred,
                "y_scores": y_scores,
            }
        )

    # Calculate average metrics
    accuracies = [result["accuracy"] for result in fold_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    logger.info(f"\n{'='*60}")
    logger.info("Cross Validation Results Summary")
    logger.info(f"{'='*60}")
    logger.success(
        f"Average Accuracy: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})"
    )
    logger.info(
        f"Individual fold accuracies: {[f'{acc:.4f}' for acc in accuracies]}"
    )

    return {
        "fold_results": fold_results,
        "avg_accuracy": avg_accuracy,
        "std_accuracy": std_accuracy,
        "accuracies": accuracies,
    }


def _plot_confusion_matrix(
    cm: np.ndarray, dataset_name: str, classifier_name: str, results_dir: str
):
    """
    Plot and save confusion matrix heatmap.
    """
    try:
        plt.figure(figsize=(8, 6))

        num_classes = cm.shape[0]

        show_percent = True
        if num_classes > 10:
            show_percent = False

        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        annot_labels = (
            np.asarray(
                [
                    f"{val}\n({perc:.1%})" if show_percent else f"{val}"
                    for val, perc in zip(cm.flatten(), cm_percent.flatten())
                ]
            )
        ).reshape(cm.shape)

        sns.heatmap(cm, annot=annot_labels, fmt="", cmap="Blues", cbar=True)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {classifier_name} on {dataset_name}")

        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(
            results_dir, f"CM_{dataset_name}_{classifier_name}.png"
        )
        plt.savefig(save_path)
        logger.info(f"Confusion Matrix heatmap saved to {save_path}")
        plt.close()

    except Exception as e:
        logger.error("Failed to plot confusion matrix heatmap.")
        logger.exception(e)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    dataset_name: str,
    classifier_name: str,
    results_dir: str = "results",
) -> None:
    """
    Evaluate classifier performance and generate reports.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels from the model
        y_scores (np.ndarray): Model discriminant function values (e.g., probabilities)
        dataset_name (str): Dataset name (used for logs and filenames)
        classifier_name (str): Classifier name (used for logs and filenames)
        results_dir (str): Directory to save plots (e.g., ROC curve)
    """

    logger.info(
        f"--- Evaluation for [{classifier_name}] on [{dataset_name}] ---"
    )

    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logger.success(f"Accuracy: {accuracy:.4f}")
    
    try:
        logger.info("Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)

        # _plot_confusion_matrix(cm, dataset_name, classifier_name, results_dir)

        logger.info(f"\n{cm}")

        logger.info("Classification Report:")
        report = classification_report(y_true, y_pred, zero_division=0)
        logger.info(f"\n{report}")

        num_classes = len(np.unique(y_true))

        if num_classes == 2:
            logger.info("Two-class dataset detected. Calculating ROC/AUC...")

            scores_for_roc = y_scores[:, 1]

            fpr, tpr, thresholds = roc_curve(y_true, scores_for_roc)

            roc_auc = auc(fpr, tpr)
            logger.success(f"Area Under Curve (AUC): {roc_auc:.4f}")

            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {roc_auc:.4f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {classifier_name} on {dataset_name}")
            plt.legend(loc="lower right")

            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(
                results_dir, f"ROC_{dataset_name}_{classifier_name}.png"
            )
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
            plt.close()

        else:
            logger.info(
                f"Skipping ROC/AUC calculation ({num_classes} classes detected)."
            )

    except Exception as e:
        logger.error("An error occurred during evaluation.")
        logger.exception(e)
        raise

    logger.info(f"--- End of Evaluation for [{classifier_name}] ---")
