
import pytest

from src.anomaly_detection.evaluation import (
    evaluate_anomaly_predictions,
)


def test_evaluation_metrics():
    """Verify confusion matrix and classification metrics."""

    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]

    metrics = evaluate_anomaly_predictions(y_true, y_pred)

    assert metrics["total_devices"] == 6
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1_score"] == pytest.approx(2 / 3)


def test_perfect_predictions():
    """Verify metrics when all predictions are correct."""

    metrics = evaluate_anomaly_predictions(
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_mismatched_lengths():
    """Reject inputs with different lengths."""

    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_anomaly_predictions(
            [0, 1, 0],
            [0, 1],
        )


def test_empty_inputs():
    """Reject empty evaluation datasets."""

    with pytest.raises(ValueError, match="empty"):
        evaluate_anomaly_predictions([], [])


def test_non_binary_labels():
    """Reject labels other than 0 and 1."""

    with pytest.raises(ValueError, match="binary"):
        evaluate_anomaly_predictions(
            [0, 1, 2],
            [0, 1, 1],
        )


def test_non_binary_predictions():
    """Reject predictions other than 0 and 1."""

    with pytest.raises(ValueError, match="binary"):
        evaluate_anomaly_predictions(
            [0, 1, 1],
            [0, 2, 1],
        )

