
import polars as pl
import pytest

from src.yield_analytics.yield_summary import (
    calculate_wafer_yield,
    calculate_lot_yield,
)


def make_sample_data():
    """Create synthetic devices across two wafers."""

    return pl.DataFrame({
        "lot_id": [
            "LOT_001", "LOT_001",
            "LOT_001", "LOT_001",
        ],
        "wafer_id": [
            "WAFER_001", "WAFER_001",
            "WAFER_002", "WAFER_002",
        ],
        "device_id": [
            "DEVICE_001", "DEVICE_002",
            "DEVICE_003", "DEVICE_004",
        ],
        "failure_rate": [
            0.0, 0.5, 0.0, 0.0,
        ],
    })


def test_wafer_count():
    result = calculate_wafer_yield(make_sample_data())
    assert result.height == 2


def test_total_devices():
    result = calculate_wafer_yield(make_sample_data())
    assert result["total_devices"].to_list() == [2, 2]


def test_passing_devices():
    result = calculate_wafer_yield(make_sample_data())
    assert result["passing_devices"].to_list() == [1, 2]


def test_failing_devices():
    result = calculate_wafer_yield(make_sample_data())
    assert result["failing_devices"].to_list() == [1, 0]


def test_yield_percentage():
    result = calculate_wafer_yield(make_sample_data())
    assert result["yield_percent"].to_list() == [50.0, 100.0]


def test_missing_column():
    df = make_sample_data().drop("failure_rate")

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        calculate_wafer_yield(df)


def test_empty_dataset():
    df = make_sample_data().head(0)

    with pytest.raises(
        ValueError,
        match="Dataset cannot be empty",
    ):
        calculate_wafer_yield(df)


@pytest.mark.parametrize(
    "invalid_value",
    [-0.1, 1.1, float("nan"), None],
)
def test_invalid_failure_rate(invalid_value):
    df = make_sample_data().with_columns(
        pl.when(pl.col("device_id") == "DEVICE_001")
        .then(pl.lit(invalid_value))
        .otherwise(pl.col("failure_rate"))
        .alias("failure_rate")
    )

    with pytest.raises(
        ValueError,
        match="failure_rate must be between 0 and 1",
    ):
        calculate_wafer_yield(df)


def test_duplicate_device():
    df = pl.concat([
        make_sample_data(),
        make_sample_data().head(1),
    ])

    with pytest.raises(
        ValueError,
        match="Duplicate device identifiers",
    ):
        calculate_wafer_yield(df)



def make_sample_wafer_yields():
    """Synthetic wafer results across two lots."""
    return pl.DataFrame({
        "lot_id": ["LOT_001", "LOT_001", "LOT_002"],
        "total_devices": [100, 10, 20],
        "passing_devices": [90, 5, 18],
        "failing_devices": [10, 5, 2],
    })


def test_lot_yield_row_count():
    result = calculate_lot_yield(make_sample_wafer_yields())
    assert result.height == 2


def test_lot_device_counts():
    result = calculate_lot_yield(make_sample_wafer_yields())

    assert result["total_devices"].to_list() == [110, 20]
    assert result["passing_devices"].to_list() == [95, 18]
    assert result["failing_devices"].to_list() == [15, 2]


def test_lot_wafer_counts():
    result = calculate_lot_yield(make_sample_wafer_yields())
    assert result["wafer_count"].to_list() == [2, 1]


def test_weighted_lot_yield():
    result = calculate_lot_yield(make_sample_wafer_yields())

    assert result["yield_percent"][0] == pytest.approx(
        95 / 110 * 100
    )
    assert result["yield_percent"][1] == pytest.approx(90.0)


def test_lot_yield_missing_column():
    df = make_sample_wafer_yields().drop("passing_devices")

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        calculate_lot_yield(df)


def test_lot_yield_empty_dataset():
    df = make_sample_wafer_yields().head(0)

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        calculate_lot_yield(df)


def test_lot_yield_inconsistent_counts():
    df = make_sample_wafer_yields().with_columns(
        pl.when(pl.col("total_devices") == 100)
        .then(99)
        .otherwise(pl.col("total_devices"))
        .alias("total_devices")
    )

    with pytest.raises(
        ValueError,
        match="must sum to total devices",
    ):
        calculate_lot_yield(df)


def test_lot_yield_negative_counts():
    df = make_sample_wafer_yields().with_columns(
        pl.lit(-1).alias("failing_devices")
    )

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        calculate_lot_yield(df)


def test_lot_yield_null_counts():
    df = make_sample_wafer_yields().with_columns(
        pl.lit(None, dtype=pl.Int64).alias("passing_devices")
    )

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        calculate_lot_yield(df)
