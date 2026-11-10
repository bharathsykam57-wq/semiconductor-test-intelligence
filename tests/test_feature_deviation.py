
import numpy as np
import polars as pl
import pytest

from src.anomaly_detection.isolation_forest import (
    FEATURE_COLUMNS,
)
from src.explainability.feature_deviation import (
    calculate_feature_deviations,
    interpret_feature_deviation,
)


def make_sample_devices():
    """Create a small synthetic device dataset."""

    data = {
        "device_id": [
            "DEVICE_001",
            "DEVICE_002",
            "DEVICE_003",
        ],
    }

    for feature in FEATURE_COLUMNS:
        data[feature] = [10.0, 20.0, 30.0]

    return pl.DataFrame(data)


def test_feature_deviation_output():
    """Verify the expected feature columns and rows."""

    df = make_sample_devices()

    result = calculate_feature_deviations(
        df,
        "DEVICE_003",
    )

    assert result.height == len(FEATURE_COLUMNS)

    assert set(result.columns) == {
        "feature",
        "device_value",
        "dataset_mean",
        "dataset_std",
        "z_score",
        "absolute_z_score",
    }


def test_z_score_calculation():
    """Verify the standardized deviation formula."""

    df = make_sample_devices()

    result = calculate_feature_deviations(
        df,
        "DEVICE_003",
    )

    expected_z = (30.0 - 20.0) / np.std(
        [10.0, 20.0, 30.0]
    )

    assert np.allclose(
        result["z_score"].to_numpy(),
        expected_z,
    )


def test_missing_device():
    """Reject device IDs that do not exist."""

    df = make_sample_devices()

    with pytest.raises(
        ValueError,
        match="exactly one device",
    ):
        calculate_feature_deviations(
            df,
            "UNKNOWN_DEVICE",
        )


def test_missing_feature_column():
    """Reject missing electrical features."""

    df = make_sample_devices().drop(
        FEATURE_COLUMNS[0]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        calculate_feature_deviations(
            df,
            "DEVICE_001",
        )


def test_empty_dataset():
    """Reject empty datasets."""

    df = make_sample_devices().head(0)

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        calculate_feature_deviations(
            df,
            "DEVICE_001",
        )


def test_non_finite_features():
    """Reject infinite or NaN feature values."""

    df = make_sample_devices().with_columns(
        pl.when(pl.col("device_id") == "DEVICE_002")
        .then(float("inf"))
        .otherwise(pl.col(FEATURE_COLUMNS[0]))
        .alias(FEATURE_COLUMNS[0])
    )

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        calculate_feature_deviations(
            df,
            "DEVICE_001",
        )


def test_constant_feature():
    """Verify zero standard deviation is handled."""

    df = make_sample_devices().with_columns(
        pl.lit(5.0).alias(FEATURE_COLUMNS[0])
    )

    result = calculate_feature_deviations(
        df,
        "DEVICE_003",
    )

    constant_feature = result.filter(
        pl.col("feature") == FEATURE_COLUMNS[0]
    )

    assert constant_feature["z_score"][0] == 0.0



@pytest.mark.parametrize(
    "z_score, expected",
    [
        (0.0, "Within one standard deviation"),
        (0.9, "Within one standard deviation"),
        (1.0, "Moderately unusual measurement"),
        (-1.5, "Moderately unusual measurement"),
        (2.0, "Highly unusual measurement"),
        (-2.5, "Highly unusual measurement"),
        (3.0, "Extremely unusual measurement"),
        (-4.0, "Extremely unusual measurement"),
    ],
)
def test_interpret_feature_deviation(z_score, expected):
    """Verify interpretation thresholds and negative values."""

    result = interpret_feature_deviation(z_score)

    assert result == expected
