import polars as pl

from src.data_validation.schema import (
    REQUIRED_COLUMNS,
    count_missing_values,
    count_non_finite_values,
    validate_measurement_ranges,
)


def generate_data_quality_report(
    df: pl.DataFrame,
    ranges: dict[str, dict[str, float]],
) -> dict:
    """
    Generate a semiconductor test-data quality report.

    Returns counts of missing columns, null values,
    non-finite values, and out-of-range measurements.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    null_counts = count_missing_values(df)
    non_finite_counts = count_non_finite_values(df)

    # Range validation requires the specified columns.
    if missing_columns:
        range_violations = None
    else:
        range_violations = validate_measurement_ranges(
            df, ranges
        )

    return {
        "total_rows": df.height,
        "total_columns": df.width,
        "missing_required_columns": missing_columns,
        "null_counts": null_counts,
        "non_finite_counts": non_finite_counts,
        "range_violations": range_violations,
    }