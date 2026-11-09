import polars as pl


def add_calculated_power(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate electrical power from voltage and current.

    Formula:
        Power (mW) = Voltage (V) × Current (mA)

    Adds a new column:
        calculated_power_mw
    """

    required_columns = ["voltage_v", "current_ma"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df.with_columns(
        (
            pl.col("voltage_v") * pl.col("current_ma")
        ).alias("calculated_power_mw")
    )