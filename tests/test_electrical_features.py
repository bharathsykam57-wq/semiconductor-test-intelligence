import polars as pl
import pytest

from src.feature_engineering.electrical import add_calculated_power


def test_add_calculated_power():
    """Verify electrical power is calculated correctly."""

    df = pl.DataFrame({
        "voltage_v": [0.8, 0.9, 1.0],
        "current_ma": [100.0, 120.0, 150.0],
    })

    result = add_calculated_power(df)

    assert result["calculated_power_mw"].to_list() == [
        80.0,
        108.0,
        150.0,
    ]


def test_add_calculated_power_missing_voltage():
    """Missing voltage should raise an error."""

    df = pl.DataFrame({
        "current_ma": [100.0],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        add_calculated_power(df)


def test_add_calculated_power_missing_current():
    """Missing current should raise an error."""

    df = pl.DataFrame({
        "voltage_v": [0.9],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        add_calculated_power(df)