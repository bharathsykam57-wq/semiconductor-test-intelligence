import polars as pl
import pytest

from src.data_validation.schema import (
    REQUIRED_COLUMNS,
    validate_required_columns,
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