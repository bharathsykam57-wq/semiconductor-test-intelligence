
import polars as pl
import pytest

from src.anomaly_detection.wafer_evaluation import (
    evaluate_leave_one_wafer_out,
)


def make_sample_wafers():
    """Create two synthetic wafers with 10 devices each."""

    rows = []

    for wafer in ["WAFER_001", "WAFER_002"]:
        for i in range(10):
            rows.append({
                "lot_id": "LOT_001",
                "wafer_id": wafer,
                "device_id": f"{wafer}_DEVICE_{i:03d}",
                "failure_rate": 1.0 if i == 9 else 0.0,
                "mean_leakage_current_ua": 5.0 + i,
                "max_leakage_current_ua": 7.0 + i,
                "max_propagation_delay_ns": 0.8 + i * 0.1,
                "std_propagation_delay_ns": 0.05 + i * 0.01,
                "mean_power_mw": 10.0 + i,
                "std_power_mw": 1.0 + i * 0.1,
            })

    return pl.DataFrame(rows)


def test_wafer_evaluation_output():
    """Verify all wafers are evaluated and metrics are returned."""

    df = make_sample_wafers()

    per_wafer, overall = evaluate_leave_one_wafer_out(
        df,
        contamination=0.10,
        random_state=42,
    )

    assert len(per_wafer) == 2
    assert overall["total_devices"] == 20

    assert (
        overall["true_positives"]
        + overall["true_negatives"]
        + overall["false_positives"]
        + overall["false_negatives"]
    ) == 20

    assert 0 <= overall["precision"] <= 1
    assert 0 <= overall["recall"] <= 1
    assert 0 <= overall["f1_score"] <= 1


def test_missing_columns():
    """Reject datasets missing required columns."""

    df = make_sample_wafers().drop(
        "max_leakage_current_ua"
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_leave_one_wafer_out(df)


def test_single_wafer():
    """Reject datasets containing only one wafer."""

    df = make_sample_wafers().filter(
        pl.col("wafer_id") == "WAFER_001"
    )

    with pytest.raises(ValueError, match="At least two wafers"):
        evaluate_leave_one_wafer_out(df)


def test_invalid_contamination():
    """Reject invalid contamination settings."""

    df = make_sample_wafers()

    with pytest.raises(ValueError, match="Contamination"):
        evaluate_leave_one_wafer_out(
            df,
            contamination=0.0,
        )
