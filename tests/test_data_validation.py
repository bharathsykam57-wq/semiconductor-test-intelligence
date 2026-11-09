import polars as pl
import pytest

from src.data_validation.schema import (
    REQUIRED_COLUMNS,
    validate_required_columns,
    count_missing_values,
    count_non_finite_values,
)


def test_valid_schema():
    """A dataset containing all required columns should pass."""

    df = pl.DataFrame({
        column: [] for column in REQUIRED_COLUMNS
    })

    validate_required_columns(df)


def test_missing_required_column():
    """A missing required column should raise ValueError."""

    df = pl.DataFrame({
        column: []
        for column in REQUIRED_COLUMNS
        if column != "voltage_v"
    })

    with pytest.raises(
        ValueError,
        match="voltage_v",
    ):
        validate_required_columns(df)


def test_multiple_missing_columns():
    """All missing required columns should be reported."""

    missing = {"voltage_v", "temperature_c"}

    df = pl.DataFrame({
        column: []
        for column in REQUIRED_COLUMNS
        if column not in missing
    })

    with pytest.raises(ValueError) as error:
        validate_required_columns(df)

    assert "voltage_v" in str(error.value)
    assert "temperature_c" in str(error.value)
    
def test_count_missing_values_complete_data():
    """Complete measurements should have zero null counts."""

    df = pl.DataFrame({
        "device_id": ["D001", "D002"],
        "voltage_v": [0.8, 0.9],
        "leakage_current_ua": [12.5, 14.2],
    })

    result = count_missing_values(df)

    assert result == {
        "device_id": 0,
        "voltage_v": 0,
        "leakage_current_ua": 0,
    }


def test_count_missing_values_incomplete_data():
    """Missing measurements should be counted per column."""

    df = pl.DataFrame({
        "device_id": ["D001", "D002", "D003"],
        "voltage_v": [0.8, None, 1.0],
        "leakage_current_ua": [12.5, None, None],
    })

    result = count_missing_values(df)

    assert result == {
        "device_id": 0,
        "voltage_v": 1,
        "leakage_current_ua": 2,
    }
    
def test_non_finite_values_valid_data():
    """Valid numerical measurements should have zero non-finite values."""

    df = pl.DataFrame({
        "device_id": ["D001", "D002"],
        "voltage_v": [0.8, 1.0],
        "leakage_current_ua": [12.5, 14.2],
    })

    result = count_non_finite_values(df)

    assert result == {
        "voltage_v": 0,
        "leakage_current_ua": 0,
    }


def test_non_finite_values_nan_and_infinity():
    """NaN and positive/negative infinity should be detected."""

    df = pl.DataFrame({
        "device_id": ["D001", "D002", "D003"],
        "voltage_v": [0.8, float("nan"), 1.0],
        "leakage_current_ua": [
            12.5,
            float("inf"),
            float("-inf"),
        ],
    })

    result = count_non_finite_values(df)

    assert result == {
        "voltage_v": 1,
        "leakage_current_ua": 2,
    }


def test_non_finite_values_excludes_nulls():
    """Null values should not be counted as NaN or infinity."""

    df = pl.DataFrame({
        "voltage_v": [0.8, None, float("nan")],
        "current_ma": [10.0, None, float("inf")],
    })

    result = count_non_finite_values(df)

    assert result == {
        "voltage_v": 1,
        "current_ma": 1,
    }