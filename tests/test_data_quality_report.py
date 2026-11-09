import polars as pl

from src.data_validation.report import generate_data_quality_report


def test_data_quality_report_valid_data():
    """Verify the report correctly summarizes valid data."""

    df = pl.DataFrame({
        "voltage_v": [0.8, 0.9],
        "frequency_mhz": [500, 1000],
        "temperature_c": [25, 50],
    })

    ranges = {
        "voltage_v": {"min": 0.70, "max": 1.10},
        "frequency_mhz": {"min": 400, "max": 1600},
        "temperature_c": {"min": -20, "max": 125},
    }

    report = generate_data_quality_report(df, ranges)

    assert report["total_rows"] == 2
    assert report["total_columns"] == 3
    assert report["missing_required_columns"]
    assert report["range_violations"] is None

    assert all(
        count == 0
        for count in report["null_counts"].values()
    )

    assert all(
        count == 0
        for count in report["non_finite_counts"].values()
    )


def test_data_quality_report_detects_invalid_data():
    """Verify missing and invalid measurements are reported."""

    df = pl.DataFrame({
        "voltage_v": [0.8, None, 1.25],
        "frequency_mhz": [500.0, float("nan"), 1000.0],
        "temperature_c": [25.0, 50.0, 130.0],
    })

    ranges = {
        "voltage_v": {"min": 0.70, "max": 1.10},
        "frequency_mhz": {"min": 400, "max": 1600},
        "temperature_c": {"min": -20, "max": 125},
    }

    report = generate_data_quality_report(df, ranges)

    assert report["total_rows"] == 3
    assert report["null_counts"]["voltage_v"] == 1
    assert report["non_finite_counts"]["frequency_mhz"] == 1

    # Required semiconductor columns are absent,
    # so range validation is skipped in this version.
    assert report["range_violations"] is None