import numpy as np
from loguru import logger
from typing import Tuple

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

class MLP:
    def __init__(self):
        self.model = MLPClassifier()
        logger.info("Initialized Multi-layer Perceptron Classifier.")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(X)
        y_scores = self.model.predict_proba(X)
        logger.info("Prediction completed.")
        return y_pred, y_scores

class NaiveBayes:
    def __init__(self):
        self.model = GaussianNB()
        logger.info("Initialized Gaussian Naive Bayes Classifier.")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(X)
        y_scores = self.model.predict_proba(X)
        logger.info("Prediction completed.")
        return y_pred, y_scores

class KNN:
    def __init__(self):
        self.model = KNeighborsClassifier()
        logger.info("Initialized K-Nearest Neighbors Classifier.")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(X)
        y_scores = self.model.predict_proba(X)
        logger.info("Prediction completed.")
        return y_pred, y_scores


class RandomForest:    
    def __init__(self):
        self.model = RandomForestClassifier()
        logger.info("Initialized Random Forest Classifier.")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(X)
        y_scores = self.model.predict_proba(X)
        logger.info("Prediction completed.")
        return y_pred, y_scores


class SVM:
    def __init__(self):
        self.model = SVC(probability=True)
        logger.info("Initialized Support Vector Machine Classifier.")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(X)
        y_scores = self.model.predict_proba(X)
        logger.info("Prediction completed.")
        return y_pred, y_scores


class XGBoost:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        logger.info("Initialized XGBoost Classifier.")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.model.predict(X)
        y_scores = self.model.predict_proba(X)
        logger.info("Prediction completed.")
        return y_pred, y_scores
