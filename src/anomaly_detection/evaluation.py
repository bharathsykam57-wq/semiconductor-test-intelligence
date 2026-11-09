
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_anomaly_predictions(y_true, y_pred):
    """
    Evaluate anomaly predictions against recorded failure labels.

    Both inputs must contain binary labels:
        0 or False: No recorded failure / not anomalous
        1 or True: Recorded failure / anomalous

    Returns a dictionary of evaluation metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("True labels must be binary.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Predicted labels must be binary.")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "total_devices": len(y_true),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }
