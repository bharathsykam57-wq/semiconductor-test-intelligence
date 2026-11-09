import numpy as np
import polars as pl
import pytest

from src.anomaly_detection.isolation_forest import (
    detect_device_anomalies,
)


def make_sample_devices():
    """Create a small synthetic device feature table."""

    return pl.DataFrame({
        "device_id": [f"D{i}" for i in range(20)],
        "mean_leakage_current_ua": np.linspace(1.0, 5.0, 20),
        "max_propagation_delay_ns": np.linspace(0.5, 1.5, 20),
        "mean_power_mw": np.linspace(10.0, 20.0, 20),
        "std_power_mw": np.linspace(1.0, 3.0, 20),
        "max_leakage_current_ua": [
            10.0 + i for i in range(20)
        ],
        "std_propagation_delay_ns": [
            0.05 + i * 0.01 for i in range(20)
        ],
    })


def test_isolation_forest_output():
    """Verify output shape, columns, and valid anomaly scores."""

    df = make_sample_devices()

    result = detect_device_anomalies(df, contamination=0.1)

    assert result.height == df.height
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert result["is_anomaly"].dtype == pl.Boolean
    assert np.isfinite(result["anomaly_score"].to_numpy()).all()


def test_isolation_forest_missing_features():
    """Missing required features should raise an error."""

    df = pl.DataFrame({
        "device_id": ["D1", "D2"],
    })

    with pytest.raises(ValueError, match="Missing required feature columns"):
        detect_device_anomalies(df)


def test_isolation_forest_invalid_contamination():
    """Contamination must be between 0 and 0.5."""

    df = make_sample_devices()

    with pytest.raises(ValueError, match="Contamination"):
        detect_device_anomalies(df, contamination=0.0)

    with pytest.raises(ValueError, match="Contamination"):
        detect_device_anomalies(df, contamination=0.6)


def test_isolation_forest_non_finite_features():
    """NaN measurements should be rejected."""

    df = make_sample_devices().with_columns(
        pl.when(pl.col("device_id") == "D0")
        .then(float("nan"))
        .otherwise(pl.col("mean_power_mw"))
        .alias("mean_power_mw")
    )

    with pytest.raises(ValueError, match="finite numerical values"):
        detect_device_anomalies(df)