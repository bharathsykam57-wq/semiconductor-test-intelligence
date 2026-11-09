import polars as pl
import pytest

from src.feature_engineering.device_statistics import (
    calculate_device_statistics,
)


def test_calculate_device_statistics():
    """Verify aggregation of semiconductor measurements."""

    df = pl.DataFrame({
        "lot_id": ["L1"] * 6,
        "wafer_id": ["W1"] * 6,
        "device_id": ["D1"] * 3 + ["D2"] * 3,
        "leakage_current_ua": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "propagation_delay_ns": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        "power_mw": [80.0, 90.0, 100.0, 100.0, 120.0, 140.0],
        "test_status": [
            "PASS", "PASS", "FAIL",
            "PASS", "FAIL", "FAIL",
        ],
    })

    result = calculate_device_statistics(df)

    assert result.height == 2
    assert result["measurement_count"].to_list() == [3, 3]
    assert result["mean_leakage_current_ua"].to_list() == [3.0, 6.0]
    assert result["max_propagation_delay_ns"].to_list() == [3.0, 4.5]
    assert result["mean_power_mw"].to_list() == [90.0, 120.0]

    assert result["failure_rate"][0] == pytest.approx(1 / 3)
    assert result["failure_rate"][1] == pytest.approx(2 / 3)

    assert result["std_power_mw"][0] == pytest.approx(10.0)
    assert result["std_power_mw"][1] == pytest.approx(20.0)


def test_device_statistics_missing_column():
    """Missing required measurement columns should raise an error."""

    df = pl.DataFrame({
        "lot_id": ["L1"],
        "wafer_id": ["W1"],
        "device_id": ["D1"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_device_statistics(df)


def test_device_statistics_single_measurement():
    """Standard deviation is null for a single measurement."""

    df = pl.DataFrame({
        "lot_id": ["L1"],
        "wafer_id": ["W1"],
        "device_id": ["D1"],
        "leakage_current_ua": [2.0],
        "propagation_delay_ns": [3.0],
        "power_mw": [90.0],
        "test_status": ["PASS"],
    })

    result = calculate_device_statistics(df)

    assert result.height == 1
    assert result["measurement_count"][0] == 1
    assert result["std_power_mw"][0] is None
    assert result["failure_rate"][0] == 0.0