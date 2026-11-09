from pathlib import Path

import polars as pl
import pytest

from src.data_ingestion.reader import load_parquet


def test_load_parquet_success(tmp_path: Path):
    """Verify that a valid Parquet file loads correctly."""

    test_file = tmp_path / "sample.parquet"

    expected = pl.DataFrame(
        {
            "device_id": ["DEVICE_001", "DEVICE_002"],
            "voltage_v": [0.8, 1.0],
            "current_ma": [10.0, 12.0],
        }
    )

    expected.write_parquet(test_file)

    actual = load_parquet(test_file)

    assert actual.equals(expected)


def test_load_parquet_missing_file(tmp_path: Path):
    """Verify that a missing file produces a clear error."""

    missing_file = tmp_path / "missing.parquet"

    with pytest.raises(
        FileNotFoundError,
        match="Parquet file not found",
    ):
        load_parquet(missing_file)