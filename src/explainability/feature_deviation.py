
import numpy as np
import polars as pl

from src.anomaly_detection.isolation_forest import (
    FEATURE_COLUMNS,
)


def calculate_feature_deviations(
    df: pl.DataFrame,
    device_id: str,
) -> pl.DataFrame:
    """
    Compare one device's electrical features against
    the full dataset using standardized deviations.

    This is descriptive analysis, not model attribution.
    """

    required_columns = ["device_id", *FEATURE_COLUMNS]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.is_empty():
        raise ValueError("Dataset cannot be empty.")

    device = df.filter(
        pl.col("device_id") == device_id
    )

    if device.height != 1:
        raise ValueError(
            "Device ID must identify exactly one device."
        )

    feature_matrix = df.select(
        FEATURE_COLUMNS
    ).to_numpy()

    if not np.isfinite(feature_matrix).all():
        raise ValueError(
            "Feature values must be finite."
        )

    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)

    device_values = device.select(
        FEATURE_COLUMNS
    ).to_numpy()[0]

    deviations = np.divide(
        device_values - means,
        stds,
        out=np.zeros_like(means, dtype=float),
        where=stds > 0,
    )

    return pl.DataFrame({
        "feature": FEATURE_COLUMNS,
        "device_value": device_values,
        "dataset_mean": means,
        "dataset_std": stds,
        "z_score": deviations,
        "absolute_z_score": np.abs(deviations),
    }).sort(
        "absolute_z_score",
        descending=True,
    )



def interpret_feature_deviation(z_score: float) -> str:
    """
    Interpret the magnitude of a standardized deviation.
    """

    magnitude = abs(z_score)

    if magnitude >= 3:
        return "Extremely unusual measurement"
    elif magnitude >= 2:
        return "Highly unusual measurement"
    elif magnitude >= 1:
        return "Moderately unusual measurement"
    else:
        return "Within one standard deviation"
