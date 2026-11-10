
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def generate_device_anomaly_chart(
    df: pl.DataFrame,
    output_path,
):
    """
    Visualize device-level Isolation Forest anomaly scores.

    Requires anomaly_score and is_anomaly columns.
    """

    required = ["anomaly_score", "is_anomaly"]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.is_empty():
        raise ValueError("Device dataset cannot be empty.")

    scores = df["anomaly_score"].to_numpy()
    flags = df["is_anomaly"].to_numpy()

    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores must be finite.")

    if df["is_anomaly"].null_count() > 0:
        raise ValueError("Anomaly flags cannot contain nulls.")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_flags = flags[order]

    positions = np.arange(len(sorted_scores))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(
        positions[~sorted_flags],
        sorted_scores[~sorted_flags],
        label="Not flagged",
        alpha=0.65,
        s=18,
    )

    ax.scatter(
        positions[sorted_flags],
        sorted_scores[sorted_flags],
        label="Flagged anomaly",
        alpha=0.85,
        s=25,
    )

    ax.axhline(
        y=0,
        linestyle="--",
        label="Decision threshold",
    )

    ax.set_title(
        "Device-Level Isolation Forest Anomaly Scores"
    )
    ax.set_xlabel("Devices ranked by anomaly score")
    ax.set_ylabel("Anomaly score")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
