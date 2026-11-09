import polars as pl


def calculate_device_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate semiconductor test measurements at device level.

    Returns one row per unique lot, wafer, and device.
    """

    required_columns = [
        "lot_id",
        "wafer_id",
        "device_id",
        "leakage_current_ua",
        "propagation_delay_ns",
        "power_mw",
        "test_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return (
        df.group_by(["lot_id", "wafer_id", "device_id"])
        .agg(
            [
                pl.len().alias("measurement_count"),

                pl.col("leakage_current_ua")
                .mean()
                .alias("mean_leakage_current_ua"),

                pl.col("propagation_delay_ns")
                .max()
                .alias("max_propagation_delay_ns"),

                pl.col("power_mw")
                .mean()
                .alias("mean_power_mw"),

                pl.col("power_mw")
                .std()
                .alias("std_power_mw"),

                (
                    (pl.col("test_status") == "FAIL")
                    .cast(pl.Float64)
                    .mean()
                ).alias("failure_rate"),
            ]
        )
        .sort(["lot_id", "wafer_id", "device_id"])
    )