"""Integration tests for STIP synthetic data generation."""

import pandas as pd

from src.data_generation.generator import (
    generate_dataset,
    load_config,
)
from src.data_generation.failure_simulation import FailureCategory


def test_generator_smoke_test(tmp_path):
    """Verify that the real generator produces valid test records."""

    config = load_config("configs/data_generation.yaml")

    output_dir = tmp_path / "generated_data"

    result = generate_dataset(
        config,
        output_dir=output_dir,
        max_devices=10,
    )

    expected_rows = (
        10 * config["dataset"]["measurements_per_device"]
    )

    assert result["devices"] == 10
    assert result["records"] == expected_rows

    parquet_files = sorted(output_dir.glob("*.parquet"))

    assert parquet_files, "Generator produced no Parquet files"

    df = pd.concat(
        [pd.read_parquet(path) for path in parquet_files],
        ignore_index=True,
    )

    assert len(df) == expected_rows
    assert df["device_id"].nunique() == 10
    assert df["test_id"].is_unique

    valid_categories = {
        category.value for category in FailureCategory
    }

    assert set(df["failure_category"]).issubset(valid_categories)

    assert df["failure_category"].notna().all()
