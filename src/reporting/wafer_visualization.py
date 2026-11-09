
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def generate_wafer_performance_chart(json_path, output_path):
    """
    Generate a wafer-level precision, recall and F1-score chart
    from an anomaly evaluation JSON report.
    """

    with Path(json_path).open("r", encoding="utf-8") as file:
        report = json.load(file)

    wafers = report["evaluation"]["per_wafer_results"]

    if not wafers:
        raise ValueError("No wafer results available.")

    labels = [w["wafer_id"] for w in wafers]
    precision = [w["precision"] * 100 for w in wafers]
    recall = [w["recall"] * 100 for w in wafers]
    f1 = [w["f1_score"] * 100 for w in wafers]

    positions = np.arange(len(wafers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        positions - width,
        precision,
        width,
        label="Precision",
    )
    ax.bar(
        positions,
        recall,
        width,
        label="Recall",
    )
    ax.bar(
        positions + width,
        f1,
        width,
        label="F1-score",
    )

    ax.set_title("Wafer-Level Anomaly Detection Performance")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Wafer")
    ax.set_ylim(0, 105)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
