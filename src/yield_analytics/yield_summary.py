
import polars as pl


def calculate_wafer_yield(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate synthetic device-level yield per wafer.

    A device passes when failure_rate == 0.
    This is a project-specific proxy definition.
    """

    required_columns = [
        "lot_id",
        "wafer_id",
        "device_id",
        "failure_rate",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.is_empty():
        raise ValueError("Dataset cannot be empty.")

    if df.select(
        pl.any_horizontal(
            pl.col("failure_rate").is_null(),
            pl.col("failure_rate").is_nan(),
            pl.col("failure_rate") < 0,
            pl.col("failure_rate") > 1,
        ).any()
    ).item():
        raise ValueError(
            "failure_rate must be between 0 and 1 "
            "and cannot contain null or NaN values."
        )

    if df.select(
        pl.struct(["lot_id", "wafer_id", "device_id"])
        .is_duplicated()
        .any()
    ).item():
        raise ValueError(
            "Duplicate device identifiers within a wafer."
        )

    return (
        df.with_columns(
            (pl.col("failure_rate") == 0)
            .alias("is_passing")
        )
        .group_by(["lot_id", "wafer_id"])
        .agg([
            pl.len().alias("total_devices"),

            pl.col("is_passing")
            .sum()
            .alias("passing_devices"),
        ])
        .with_columns(
            (
                pl.col("total_devices")
                - pl.col("passing_devices")
            ).alias("failing_devices"),

            (
                pl.col("passing_devices")
                / pl.col("total_devices")
                * 100
            ).alias("yield_percent"),
        )
        .sort(["lot_id", "wafer_id"])
    )



def calculate_lot_yield(
    wafer_yield: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate lot-level yield from wafer-level results.

    Yield is weighted by device counts, not
    calculated as an unweighted average of wafers.
    """

    required_columns = [
        "lot_id",
        "total_devices",
        "passing_devices",
        "failing_devices",
    ]

    missing = [
        column
        for column in required_columns
        if column not in wafer_yield.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if wafer_yield.is_empty():
        raise ValueError("Wafer yield data cannot be empty.")

    for column in [
        "total_devices",
        "passing_devices",
        "failing_devices",
    ]:
        if (
            wafer_yield[column].null_count() > 0
            or wafer_yield.filter(
                pl.col(column) < 0
            ).height > 0
        ):
            raise ValueError(
                f"{column} must contain nonnegative, "
                "non-null counts."
            )

    if wafer_yield.filter(
        pl.col("total_devices")
        != (
            pl.col("passing_devices")
            + pl.col("failing_devices")
        )
    ).height > 0:
        raise ValueError(
            "Passing and failing counts must sum "
            "to total devices."
        )

    return (
        wafer_yield
        .group_by("lot_id")
        .agg([
            pl.len().alias("wafer_count"),
            pl.col("total_devices")
            .sum()
            .alias("total_devices"),
            pl.col("passing_devices")
            .sum()
            .alias("passing_devices"),
            pl.col("failing_devices")
            .sum()
            .alias("failing_devices"),
        ])
        .with_columns(
            (
                pl.col("passing_devices")
                / pl.col("total_devices")
                * 100
            ).alias("yield_percent")
        )
        .sort("lot_id")
    )
