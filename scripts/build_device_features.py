
"""
Build device-level features from synthetic semiconductor measurements.

Example:
python -m scripts.build_device_features \
    --input-dir data/smoke_test \
    --output-file data/smoke_test_features/device_features.parquet
"""

import argparse
from pathlib import Path

import polars as pl

from src.feature_engineering.device_statistics import (
    calculate_device_statistics,
)


def build_device_features(
    input_dir: str | Path,
    output_file: str | Path,
) -> pl.DataFrame:
    input_path = Path(input_dir)
    output_path = Path(output_file)

    parquet_files = sorted(input_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {input_path}"
        )

    # Read all generated measurement chunks.
    raw_data = pl.concat(
        [pl.read_parquet(file) for file in parquet_files],
        how="vertical",
    )

    # Aggregate raw measurements into one row per device.
    device_features = calculate_device_statistics(raw_data)

    # Create the output directory if necessary.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save model-ready device features.
    device_features.write_parquet(output_path)

    print("FEATURE ENGINEERING COMPLETED")
    print(f"Input files: {len(parquet_files)}")
    print(f"Raw measurements: {raw_data.height:,}")
    print(f"Devices: {device_features.height:,}")
    print(f"Output: {output_path.resolve()}")

    return device_features


def main():
    parser = argparse.ArgumentParser(
        description="Build semiconductor device-level features."
    )

    parser.add_argument(
        "--input-dir",
        default="data/raw",
    )

    parser.add_argument(
        "--output-file",
        default=(
            "data/processed/device_features/"
            "device_features.parquet"
        ),
    )

    args = parser.parse_args()

    build_device_features(
        input_dir=args.input_dir,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
