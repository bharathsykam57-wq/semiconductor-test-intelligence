
import polars as pl
import pytest

from src.wafer_analytics.wafer_summary import (
    calculate_wafer_summary,
    assess_wafer_sample_sizes,
)


def make_sample_data():
    """Create synthetic data for two wafers."""

    return pl.DataFrame({
        "lot_id": [
            "LOT_001",
            "LOT_001",
            "LOT_001",
            "LOT_001",
        ],
        "wafer_id": [
            "WAFER_001",
            "WAFER_001",
            "WAFER_002",
            "WAFER_002",
        ],
        "device_id": [
            "DEVICE_001",
            "DEVICE_002",
            "DEVICE_003",
            "DEVICE_004",
        ],
        "mean_leakage_current_ua": [
            10.0, 20.0, 30.0, 40.0
        ],
        "max_propagation_delay_ns": [
            2.0, 4.0, 6.0, 8.0
        ],
        "mean_power_mw": [
            100.0, 200.0, 300.0, 400.0
        ],
        "failure_rate": [
            0.0, 0.5, 0.0, 1.0
        ],
    })


def test_wafer_summary_row_count():
    df = make_sample_data()

    result = calculate_wafer_summary(df)

    assert result.height == 2


def test_device_count():
    df = make_sample_data()

    result = calculate_wafer_summary(df)

    assert result["device_count"].to_list() == [2, 2]


def test_average_leakage_current():
    df = make_sample_data()

    result = calculate_wafer_summary(df)

    assert result["avg_leakage_current_ua"].to_list() == [
        15.0, 35.0
    ]


def test_average_propagation_delay():
    df = make_sample_data()

    result = calculate_wafer_summary(df)

    assert result[
        "avg_max_propagation_delay_ns"
    ].to_list() == [3.0, 7.0]


def test_average_power():
    df = make_sample_data()

    result = calculate_wafer_summary(df)

    assert result["avg_power_mw"].to_list() == [
        150.0, 350.0
    ]


def test_average_failure_rate():
    df = make_sample_data()

    result = calculate_wafer_summary(df)

    assert result[
        "avg_device_failure_rate"
    ].to_list() == [0.25, 0.5]


def test_missing_column():
    df = make_sample_data().drop(
        "mean_power_mw"
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        calculate_wafer_summary(df)


def test_empty_dataset():
    df = make_sample_data().head(0)

    with pytest.raises(
        ValueError,
        match="Dataset cannot be empty",
    ):
        calculate_wafer_summary(df)


def test_multiple_lots():
    df = make_sample_data().with_columns(
        pl.when(pl.col("wafer_id") == "WAFER_002")
        .then(pl.lit("LOT_002"))
        .otherwise(pl.col("lot_id"))
        .alias("lot_id")
    )

    result = calculate_wafer_summary(df)

    assert result.height == 2
    assert result["lot_id"].to_list() == [
        "LOT_001",
        "LOT_002",
    ]


def test_sample_size_classification():
    """Verify classification above and below the threshold."""

    df = pl.DataFrame({
        "device_count": [100, 30, 1],
    })

    result = assess_wafer_sample_sizes(df)

    assert result["sample_size_status"].to_list() == [
        "Meets sample-size guideline",
        "Meets sample-size guideline",
        "Limited sample size",
    ]


def test_custom_sample_size_threshold():
    """Verify a configurable minimum device count."""

    df = pl.DataFrame({
        "device_count": [20, 50],
    })

    result = assess_wafer_sample_sizes(
        df,
        min_devices=50,
    )

    assert result["sample_size_status"].to_list() == [
        "Limited sample size",
        "Meets sample-size guideline",
    ]


def test_invalid_minimum_device_count():
    """Reject nonpositive thresholds."""

    df = pl.DataFrame({
        "device_count": [100],
    })

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        assess_wafer_sample_sizes(
            df,
            min_devices=0,
        )


def test_missing_device_count_column():
    """Reject missing required columns."""

    df = pl.DataFrame({
        "wafer_id": ["WAFER_001"],
    })

    with pytest.raises(
        ValueError,
        match="Missing required column",
    ):
        assess_wafer_sample_sizes(df)


def test_null_device_count():
    """Reject missing device counts."""

    df = pl.DataFrame({
        "device_count": [100, None],
    })

    with pytest.raises(
        ValueError,
        match="cannot contain null",
    ):
        assess_wafer_sample_sizes(df)


def test_negative_device_count():
    """Reject negative device counts."""

    df = pl.DataFrame({
        "device_count": [100, -1],
    })

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        assess_wafer_sample_sizes(df)
