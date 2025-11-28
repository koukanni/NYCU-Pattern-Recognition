import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional

def _plot_confusion_matrix(cm: np.ndarray, 
                           dataset_name: str, 
                           classifier_name: str, 
                           results_dir: str):
    """
    Plot and save confusion matrix heatmap.
    """
    try:
        plt.figure(figsize=(8, 6))
        
        num_classes = cm.shape[0]
        
        show_percent = True
        if num_classes > 10:
            show_percent = False

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        annot_labels = (
            np.asarray([
                f"{val}\n({perc:.1%})" if show_percent else f"{val}"
                for val, perc in zip(cm.flatten(), cm_percent.flatten())
            ])
        ).reshape(cm.shape)

        sns.heatmap(
            cm, 
            annot=annot_labels, 
            fmt='',
            cmap='Blues',
            cbar=True
        )
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {classifier_name} on {dataset_name}')
        
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"CM_{dataset_name}_{classifier_name}.png")
        plt.savefig(save_path)
        logger.info(f"Confusion Matrix heatmap saved to {save_path}")
        plt.close()

    except Exception as e:
        logger.error("Failed to plot confusion matrix heatmap.")
        logger.exception(e)


def evaluate_classifier(y_true: np.ndarray, 
                      y_pred: np.ndarray, 
                      y_scores: np.ndarray, 
                      dataset_name: str, 
                      classifier_name: str,
                      results_dir: str = "results") -> None:
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
    
    logger.info(f"--- Evaluation for [{classifier_name}] on [{dataset_name}] ---")
    
    try:
        logger.info("Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        
        _plot_confusion_matrix(cm, dataset_name, classifier_name, results_dir)

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
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {classifier_name} on {dataset_name}')
            plt.legend(loc="lower right")
            
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, f"ROC_{dataset_name}_{classifier_name}.png")
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
            plt.close()
        
        else:
            logger.info(f"Skipping ROC/AUC calculation ({num_classes} classes detected).")
            
    except Exception as e:
        logger.error("An error occurred during evaluation.")
        logger.exception(e)
        raise
        
    logger.info(f"--- End of Evaluation for [{classifier_name}] ---")