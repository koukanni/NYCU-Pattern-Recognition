from sklearn.ensemble import RandomForestClassifier
import numpy as np
from loguru import logger

from data_loader import load_dataset
from classifier import NaiveBayes
from main import k_fold_cross_validation, evaluate_classifier


if __name__ == "__main__":
    dataset = "bankruptcy"

    model = RandomForestClassifier()
    logger.info("Initialized Random Forest Classifier.")

    X, y = load_dataset(dataset)
    model.fit(X, y)

    importances = model.feature_importances_
    top_20_indices = np.argsort(importances)[::-1][:20]

    logger.info(f"Top 20 most important feature indices: {top_20_indices}")

    X_top20 = X[:, top_20_indices]
    logger.info(f"Created new dataset with top 20 features: X shape={X_top20.shape}")

    logger.info("Running K-fold CV for NaiveBayes on TOP 20 FEATURES...")
    nb_top20_cv_results = k_fold_cross_validation(
        NaiveBayes, X_top20, y
    )

    logger.info("Evaluating final NaiveBayes model on TOP 20 FEATURES...")
    nb_top20_final = NaiveBayes()
    nb_top20_final.train(X_top20, y)
    y_pred_nb_top20, y_scores_nb_top20 = nb_top20_final.predict(X_top20)

    evaluate_classifier(
        y_true=y,
        y_pred=y_pred_nb_top20,
        y_scores=y_scores_nb_top20,
        dataset_name="bankruptcy_top20",
        classifier_name="NaiveBayes"
    )