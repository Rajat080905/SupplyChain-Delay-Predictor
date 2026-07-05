"""
evaluation.py

Evaluates trained models.

Author: Rajat
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


class ModelEvaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test, threshold=0.5):

        if hasattr(self.model, "predict_proba"):

            probs = self.model.predict_proba(X_test)[:, 1]
            predictions = (probs >= threshold).astype(int)

        else:

            predictions = self.model.predict(X_test)
            probs = predictions

        accuracy = accuracy_score(y_test, predictions)

        print("\nAccuracy :", round(accuracy, 4))
        print()

        print(classification_report(
            y_test,
            predictions
        ))

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            probs
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
            "predictions": predictions,
            "probabilities": probs
        }
