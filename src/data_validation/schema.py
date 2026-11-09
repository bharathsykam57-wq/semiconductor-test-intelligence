import polars as pl


REQUIRED_COLUMNS = [
    "lot_id",
    "wafer_id",
    "device_id",
    "test_id",
    "test_condition_id",
    "voltage_v",
    "frequency_mhz",
    "temperature_c",
    "current_ma",
    "power_mw",
    "leakage_current_ua",
    "propagation_delay_ns",
    "performance_score",
    "test_status",
    "failure_category",
]


def validate_required_columns(df: pl.DataFrame) -> None:
    """
    Check whether all required semiconductor test columns exist.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )
        
def count_missing_values(df: pl.DataFrame) -> dict[str, int]:
    """
    Count null values in each column of a semiconductor dataset.

    Returns
    -------
    dict[str, int]
        Column names mapped to their null counts.
    """

    return {
        column: df[column].null_count()
        for column in df.columns
    }
    
def count_non_finite_values(df: pl.DataFrame) -> dict[str, int]:
    """
    Count NaN and infinite values in numeric columns.

    Null values are excluded because they are handled
    separately by count_missing_values().
    """

    numeric_columns = [
        column
        for column, dtype in df.schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]

    return {
        column: df.select(
            (
                pl.col(column).is_nan().fill_null(False)
                | pl.col(column).is_infinite().fill_null(False)
            ).sum()
        ).item()
        for column in numeric_columns
    }
    
def validate_measurement_ranges(
    df: pl.DataFrame,
    ranges: dict[str, dict[str, float]],
) -> dict[str, int]:
    """
    Count measurements outside configured operating ranges.

    Returns a dictionary mapping each checked column
    to its number of out-of-range measurements.

    Null and non-finite values are handled separately.
    """

    violations = {}

    for column, limits in ranges.items():
        if column not in df.columns:
            raise ValueError(
                f"Missing column for range validation: {column}"
            )

        minimum = limits["min"]
        maximum = limits["max"]

        if minimum > maximum:
            raise ValueError(
                f"Invalid range for {column}: min exceeds max"
            )

        invalid_count = df.select(
            (
                pl.col(column).is_finite()
                & (
                    (pl.col(column) < minimum)
                    | (pl.col(column) > maximum)
                )
            )
            .fill_null(False)
            .sum()
        ).item()

        violations[column] = int(invalid_count)

    return violations