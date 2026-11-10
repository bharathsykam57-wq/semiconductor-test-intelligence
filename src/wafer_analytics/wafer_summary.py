
import polars as pl


def calculate_wafer_summary(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate wafer-level engineering statistics
    from synthetic device features.
    """

    required_columns = [
        "lot_id",
        "wafer_id",
        "device_id",
        "mean_leakage_current_ua",
        "max_propagation_delay_ns",
        "mean_power_mw",
        "failure_rate",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.is_empty():
        raise ValueError("Dataset cannot be empty.")

    return (
        df.group_by(["lot_id", "wafer_id"])
        .agg(
            [
                pl.col("device_id")
                .n_unique()
                .alias("device_count"),

                pl.col("mean_leakage_current_ua")
                .mean()
                .alias("avg_leakage_current_ua"),

                pl.col("max_propagation_delay_ns")
                .mean()
                .alias("avg_max_propagation_delay_ns"),

                pl.col("mean_power_mw")
                .mean()
                .alias("avg_power_mw"),

                pl.col("failure_rate")
                .mean()
                .alias("avg_device_failure_rate"),
            ]
        )
        .sort(["lot_id", "wafer_id"])
    )



def assess_wafer_sample_sizes(
    wafer_summary: pl.DataFrame,
    min_devices: int = 30,
) -> pl.DataFrame:
    """
    Label wafers according to a configurable
    minimum device-count guideline.

    This is a descriptive guideline, not a
    statistically validated sample-size requirement.
    """

    if min_devices < 1:
        raise ValueError(
            "Minimum device count must be positive."
        )

    if "device_count" not in wafer_summary.columns:
        raise ValueError(
            "Missing required column: device_count"
        )

    if wafer_summary["device_count"].null_count() > 0:
        raise ValueError(
            "Device counts cannot contain null values."
        )

    if (
        wafer_summary
        .filter(pl.col("device_count") < 0)
        .height > 0
    ):
        raise ValueError(
            "Device counts cannot be negative."
        )

    return wafer_summary.with_columns(
        pl.when(
            pl.col("device_count") >= min_devices
        )
        .then(pl.lit("Meets sample-size guideline"))
        .otherwise(pl.lit("Limited sample size"))
        .alias("sample_size_status")
    )
