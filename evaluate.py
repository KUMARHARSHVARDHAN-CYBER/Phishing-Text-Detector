"""
Evaluation Module for Machine Learning Models.

This module provides the ModelEvaluator class to compute key classification
performance metrics including Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:
    """
    Utility class for evaluating classification models with support for
    both string labels ('ham'/'spam') and numeric labels (0/1).
    """

    @staticmethod
    def evaluate(
        y_true: Union[list, np.ndarray],
        y_pred: Union[list, np.ndarray],
        pos_label: Optional[Union[str, int]] = None,
        labels: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Computes evaluation metrics between ground truth and predictions.
        """
        # Auto-detect pos_label if not specified
        if pos_label is None:
            unique_labels = list(set(y_true))
            if 'spam' in unique_labels:
                pos_label = 'spam'
            elif 'phishing' in unique_labels:
                pos_label = 'phishing'
            elif 1 in unique_labels:
                pos_label = 1
            else:
                pos_label = unique_labels[0] if len(unique_labels) > 0 else 1

        # Determine average strategy for multi-class vs binary
        is_binary = len(set(y_true).union(set(y_pred))) <= 2

        acc = float(accuracy_score(y_true, y_pred))
        
        if is_binary and pos_label is not None:
            precision = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            recall = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        else:
            precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        if labels is not None:
            cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        else:
            cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Confusion Matrix': cm
        }

    @staticmethod
    def display_metrics(metrics: Dict[str, Any]) -> None:
        """
        Pretty prints evaluation metrics to the console.
        """
        print("-" * 40)
        print("          EVALUATION RESULTS            ")
        print("-" * 40)
        for key, value in metrics.items():
            if key == 'Confusion Matrix':
                print(f"{key}:\n{np.array(value)}")
            elif isinstance(value, float):
                print(f"{key:<18}: {value * 100:.2f}% ({value:.4f})")
            else:
                print(f"{key:<18}: {value}")
        print("-" * 40)


if __name__ == "__main__":
    print("=== Testing ModelEvaluator (String Labels) ===")
    y_test_str = ['ham', 'spam', 'ham', 'ham', 'spam', 'spam', 'ham', 'spam']
    y_pred_str = ['ham', 'spam', 'ham', 'spam', 'spam', 'spam', 'ham', 'ham']

    metrics_str = ModelEvaluator.evaluate(y_test_str, y_pred_str, pos_label='spam', labels=['ham', 'spam'])
    ModelEvaluator.display_metrics(metrics_str)

    print("\n=== Testing ModelEvaluator (Numeric Labels) ===")
    y_test_num = [0, 1, 0, 0, 1, 1, 0, 1]
    y_pred_num = [0, 1, 0, 1, 1, 1, 0, 0]

    metrics_num = ModelEvaluator.evaluate(y_test_num, y_pred_num)
    ModelEvaluator.display_metrics(metrics_num)

    print("\nAll ModelEvaluator checks completed successfully!")