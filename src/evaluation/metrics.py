"""Binary classification metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def compute_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities)
    scores = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
    predictions = (scores >= threshold).astype(int)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    result["auroc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")
    return result
