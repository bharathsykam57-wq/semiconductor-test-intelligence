
import polars as pl
import pytest

from src.reporting.device_anomaly_visualization import (
    generate_device_anomaly_chart,
)


def make_sample_devices():
    """Create a small synthetic anomaly detection dataset."""

    return pl.DataFrame({
        "device_id": [
            "DEVICE_001",
            "DEVICE_002",
            "DEVICE_003",
            "DEVICE_004",
            "DEVICE_005",
        ],
        "anomaly_score": [
            -0.15,
            -0.08,
            0.12,
            -0.04,
            0.25,
        ],
        "is_anomaly": [
            False,
            False,
            True,
            False,
            True,
        ],
    })


def test_chart_created(tmp_path):
    """Verify that the chart is generated."""

    df = make_sample_devices()
    output_path = tmp_path / "device_anomalies.png"

    result = generate_device_anomaly_chart(
        df,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_chart_is_valid_png(tmp_path):
    """Verify that the generated image has a PNG signature."""

    df = make_sample_devices()
    output_path = tmp_path / "chart.png"

    generate_device_anomaly_chart(df, output_path)

    with output_path.open("rb") as file:
        signature = file.read(8)

    assert signature == b"\x89PNG\r\n\x1a\n"


def test_missing_columns(tmp_path):
    """Reject datasets missing required columns."""

    df = make_sample_devices().drop("anomaly_score")

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        generate_device_anomaly_chart(
            df,
            tmp_path / "chart.png",
        )


def test_empty_dataset(tmp_path):
    """Reject empty device datasets."""

    df = make_sample_devices().head(0)

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        generate_device_anomaly_chart(
            df,
            tmp_path / "chart.png",
        )


def test_non_finite_scores(tmp_path):
    """Reject infinite anomaly scores."""

    df = make_sample_devices().with_columns(
        pl.when(pl.col("device_id") == "DEVICE_003")
        .then(float("inf"))
        .otherwise(pl.col("anomaly_score"))
        .alias("anomaly_score")
    )

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        generate_device_anomaly_chart(
            df,
            tmp_path / "chart.png",
        )


def test_null_anomaly_flags(tmp_path):
    """Reject missing anomaly flags."""

    df = make_sample_devices().with_columns(
        pl.when(pl.col("device_id") == "DEVICE_003")
        .then(None)
        .otherwise(pl.col("is_anomaly"))
        .alias("is_anomaly")
    )

    with pytest.raises(
        ValueError,
        match="cannot contain nulls",
    ):
        generate_device_anomaly_chart(
            df,
            tmp_path / "chart.png",
        )
