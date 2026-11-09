import numpy as np
import polars as pl

from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = [
    "mean_leakage_current_ua",
    "max_propagation_delay_ns",
    "mean_power_mw",
    "std_power_mw",
]


def detect_device_anomalies(
    df: pl.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Detect unusual semiconductor devices using Isolation Forest.

    Returns the original DataFrame with two additional columns:
        anomaly_score: Higher values indicate greater abnormality.
        is_anomaly: True if the device is flagged as anomalous.
    """

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    if df.height < 2:
        raise ValueError(
            "At least two devices are required for anomaly detection."
        )

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "Contamination must be greater than 0 and at most 0.5."
        )

    features = df.select(FEATURE_COLUMNS).to_numpy()

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature columns must contain only finite numerical values."
        )

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    predictions = model.fit_predict(features)

    # Negate decision_function so larger scores mean more anomalous.
    anomaly_scores = -model.decision_function(features)

    return df.with_columns(
        [
            pl.Series("anomaly_score", anomaly_scores),
            pl.Series("is_anomaly", predictions == -1),
        ]
    )