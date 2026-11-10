
import polars as pl

from scripts.build_device_features import build_device_features


def test_build_device_features(tmp_path):
    # Create a small synthetic input dataset.
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    raw_data = pl.DataFrame({
        "lot_id": ["LOT_001"] * 4,
        "wafer_id": ["WAFER_001"] * 4,
        "device_id": ["DEV_001", "DEV_001", "DEV_002", "DEV_002"],
        "leakage_current_ua": [0.4, 0.6, 0.8, 1.0],
        "propagation_delay_ns": [1.1, 1.2, 1.3, 1.4],
        "power_mw": [10.0, 12.0, 14.0, 16.0],
        "test_status": ["PASS", "PASS", "PASS", "FAIL"],
    })

    raw_data.write_parquet(input_dir / "part-00000.parquet")

    output_file = tmp_path / "processed" / "device_features.parquet"

    # Run the pipeline.
    result = build_device_features(
        input_dir=input_dir,
        output_file=output_file,
    )

    # Check that the output file exists.
    assert output_file.exists()

    # Check the number of devices.
    assert result.height == 2

    # Check that all measurements were preserved.
    assert result["measurement_count"].sum() == 4

    # Check the saved dataset.
    saved_data = pl.read_parquet(output_file)
    assert saved_data.height == 2

    # Check device failure rates.
    failure_rates = dict(
        zip(
            saved_data["device_id"].to_list(),
            saved_data["failure_rate"].to_list(),
        )
    )

    assert failure_rates["DEV_001"] == 0.0
    assert failure_rates["DEV_002"] == 0.5


def test_build_device_features_missing_input(tmp_path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    output_file = tmp_path / "output.parquet"

    try:
        build_device_features(input_dir, output_file)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError for empty input directory."
        )
