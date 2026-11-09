
import polars as pl
from sklearn.ensemble import IsolationForest

from src.anomaly_detection.isolation_forest import FEATURE_COLUMNS
from src.anomaly_detection.evaluation import (
    evaluate_anomaly_predictions,
)


def evaluate_leave_one_wafer_out(
    df: pl.DataFrame,
    contamination: float = 0.10,
    random_state: int = 42,
):
    """
    Evaluate Isolation Forest using leave-one-wafer-out validation.

    Each wafer is held out once, while the model is trained on
    devices belonging to the remaining wafers.

    Returns:
        per_wafer_results: list of dictionaries
        overall_metrics: dictionary
    """
    required_columns = [
        "lot_id",
        "wafer_id",
        "device_id",
        "failure_rate",
        *FEATURE_COLUMNS,
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not 0 < contamination <= 0.5:
        raise ValueError("Contamination must be greater than 0 and at most 0.5.")

    wafer_groups = (
        df.select(["lot_id", "wafer_id"])
        .unique()
        .sort(["lot_id", "wafer_id"])
        .iter_rows(named=True)
    )

    wafers = list(wafer_groups)

    if len(wafers) < 2:
        raise ValueError("At least two wafers are required.")

    per_wafer_results = []
    all_true = []
    all_pred = []

    for wafer in wafers:
        is_test_wafer = (
            (pl.col("lot_id") == wafer["lot_id"])
            & (pl.col("wafer_id") == wafer["wafer_id"])
        )

        train = df.filter(~is_test_wafer)
        test = df.filter(is_test_wafer)

        if train.height < 2:
            raise ValueError(
                "At least two training devices are required."
            )

        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(train.select(FEATURE_COLUMNS).to_numpy())

        y_true = (test["failure_rate"] > 0).to_numpy()
        y_pred = (
            model.predict(
                test.select(FEATURE_COLUMNS).to_numpy()
            ) == -1
        )

        metrics = evaluate_anomaly_predictions(y_true, y_pred)

        per_wafer_results.append({
            "lot_id": wafer["lot_id"],
            "wafer_id": wafer["wafer_id"],
            **metrics,
        })

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    overall_metrics = evaluate_anomaly_predictions(
        all_true,
        all_pred,
    )

    return per_wafer_results, overall_metrics
