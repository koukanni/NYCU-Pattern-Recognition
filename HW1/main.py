import argparse
import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import sys

from data_loader import load_dataset
from evaluation import evaluate_classifier

from classifier import MLP, NaiveBayes, KNN, RandomForest, SVM, XGBoost

CLASSIFIER_MAP = {
    'MLP': MLP,
    'NaiveBayes': NaiveBayes,
    'KNN': KNN,
    'RandomForest': RandomForest,
    'SVM': SVM,
    'XGBoost': XGBoost
}


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
    logger.info(f"Starting {k}-fold cross validation for {classifier_class.__name__}...")

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
            f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

        classifier = classifier_class()
        classifier.train(X_train, y_train)

        # Make predictions
        y_pred, y_scores = classifier.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        logger.success(f"Fold {fold_idx} Accuracy: {accuracy:.4f}")

        # Store results
        fold_results.append({
            'fold': fold_idx,
            'accuracy': accuracy,
            'y_true': y_val,
            'y_pred': y_pred,
            'y_scores': y_scores
        })

    # Calculate average metrics
    accuracies = [result['accuracy'] for result in fold_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    logger.info(f"\n{'='*60}")
    logger.info("Cross Validation Results Summary")
    logger.info(f"{'='*60}")
    logger.success(f"Average Accuracy: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
    logger.info(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in accuracies]}")

    return {
        'fold_results': fold_results,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'accuracies': accuracies
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Classifier with k-fold cross validation')
    parser.add_argument('--model', type=str, required=True,
                        help=f'Classifier to use. Available: {list(CLASSIFIER_MAP.keys())}')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to load (e.g., iris, breast_cancer, bank_note, glass)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds for cross validation (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Load dataset
    X, y = load_dataset(args.dataset)
    logger.info(f"Loaded {args.dataset} dataset: X shape={X.shape}, y shape={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifier_class = CLASSIFIER_MAP.get(args.model)
    if classifier_class is None:
        logger.error(f"Unknown model: '{args.model}'")
        logger.info(f"Available models: {list(CLASSIFIER_MAP.keys())}")
        sys.exit(1)

    logger.info(f"Selected model: {args.model}")

    cv_results = k_fold_cross_validation(
        classifier_class, X_train, y_train, k=args.k, random_state=args.random_state
    )

    logger.info(f"\n{'='*60}")
    logger.info("Training final model on entire dataset for evaluation...")
    logger.info(f"{'='*60}")

    classifier = classifier_class()
    classifier.train(X_train, y_train)
    y_pred_final, y_scores_final = classifier.predict(X_test)

    evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred_final,
        y_scores=y_scores_final,
        dataset_name=args.dataset,
        classifier_name=args.model
    )