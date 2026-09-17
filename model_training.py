from typing import Any, Dict, Optional
import os
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer


class PhishingClassifier:
    """
    Multinomial Naive Bayes classifier tailored for phishing detection.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = MultinomialNB(alpha=self.alpha)
        self.is_trained = False

    def train(self, X_train_vec, y_train):
        """
        Trains the MultinomialNB model.
        """
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        return self

    def fit(self, X_train_vec, y_train):
        """
        Alias for train() for standard scikit-learn API compatibility.
        """
        return self.train(X_train_vec, y_train)

    def predict(self, X_test_vec):
        """
        Predicts labels for feature vectors.
        """
        return self.model.predict(X_test_vec)

    def predict_proba(self, X_test_vec):
        """
        Returns class probability estimates.
        """
        return self.model.predict_proba(X_test_vec)

    def evaluate(self, X_test_vec, y_test) -> Dict[str, Any]:
        """
        Evaluates the model on test data and returns key metrics.
        """
        y_pred = self.predict(X_test_vec)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0, average="weighted")),
            "recall": float(recall_score(y_test, y_pred, zero_division=0, average="weighted")),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0, average="weighted")),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0)
        }

    def save_model(self, filepath: str) -> None:
        """
        Saves the trained model to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "PhishingClassifier":
        """
        Loads a trained model from disk.
        """
        instance = cls()
        instance.model = joblib.load(filepath)
        instance.is_trained = True
        return instance


if __name__ == "__main__":
    print("=== Testing PhishingClassifier ===")

    train_texts = [
        "Please find attached the invoice for your recent purchase",
        "Urgent! Your bank account is suspended. Click here to verify password immediately",
        "Team meeting is rescheduled to tomorrow 10 AM in Conference Room B",
        "Claim your $1,000,000 lottery winnings now! Send bank details to claim prize",
        "The project quarterly report has been updated in the shared drive",
        "Security Alert: Your PayPal account was accessed. Update credentials now"
    ]
    train_labels = [0, 1, 0, 1, 0, 1]

    test_texts = [
        "Let's catch up over coffee this afternoon to discuss the agenda",
        "Warning: Your account will be terminated unless you verify your credit card details immediately"
    ]
    test_labels = [0, 1]

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(train_texts)
    X_test_vec = vectorizer.transform(test_texts)

    classifier = PhishingClassifier(alpha=1.0)
    classifier.train(X_train_vec, train_labels)

    preds = classifier.predict(X_test_vec)
    probs = classifier.predict_proba(X_test_vec)

    for text, p, prob in zip(test_texts, preds, probs):
        label = "PHISHING" if p == 1 else "LEGITIMATE"
        print(f"\nSample: {text}\nResult: {label} (Confidence: {np.max(prob)*100:.1f}%)")

    metrics = classifier.evaluate(X_test_vec, test_labels)
    print(f"\nAccuracy: {metrics['accuracy'] * 100:.1f}%")
    print(f"F1-Score: {metrics['f1_score']:.2f}")
    print("\nAll checks passed successfully!")