
import pandas as pd

from src.wafer_map_features.wafer_statistics import (
    calculate_wafer_statistics,
)


def extract_batch_features(df, max_wafers=None):
    """
    Extract numerical features from wafer maps.

    Parameters
    ----------
    df : pandas.DataFrame
        WM-811K dataset containing a waferMap column.

    max_wafers : int or None
        Maximum number of wafers to process.

    Returns
    -------
    pandas.DataFrame
        One row per wafer with numerical features.
    """

    if "waferMap" not in df.columns:
        raise ValueError("Dataset must contain a waferMap column.")

    if max_wafers is not None and max_wafers < 0:
        raise ValueError("max_wafers must be non-negative.")

    records = []

    selected = (
        df if max_wafers is None
        else df.iloc[:max_wafers]
    )

    for wafer_id, wafer_map in selected["waferMap"].items():
        features = calculate_wafer_statistics(wafer_map)

        features["wafer_id"] = wafer_id

        records.append(features)

    return pd.DataFrame.from_records(records)
