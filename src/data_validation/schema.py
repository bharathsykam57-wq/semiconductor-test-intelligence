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