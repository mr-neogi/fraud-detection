
"""Metrics, validation-based threshold selection, and report generation."""
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

def choose_threshold(y_true, scores, min_precision=0.10):
    """Maximize recall among thresholds meeting a minimum precision floor."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    eligible = np.where(p >= min_precision)[0]
    if len(eligible):
        best = eligible[np.argmax(r[eligible])]
    else:
        # Fall back to the best F1 threshold if the requested floor is infeasible.
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        best = int(np.argmax(f1))
    return float(thresholds[best])

def evaluate_scores(y_true, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, pred, target_names=["Legit", "Fraud"],
            output_dict=True, zero_division=0
        ),
        "predictions": pred
    }
