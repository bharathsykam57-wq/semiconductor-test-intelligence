
import json
from datetime import datetime, timezone
from pathlib import Path


def generate_anomaly_report(
    per_wafer_results,
    overall_metrics,
    output_path,
    contamination=0.10,
    random_state=42,
):
    """
    Generate a JSON engineering report for wafer-level
    anomaly detection evaluation.
    """

    report = {
        "report_type": "wafer_level_anomaly_evaluation",
        "data_source": "synthetic",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": {
            "algorithm": "IsolationForest",
            "contamination": contamination,
            "random_state": random_state,
        },
        "evaluation": {
            "method": "leave_one_wafer_out",
            "positive_label": "device_has_recorded_failure",
            "overall_metrics": overall_metrics,
            "per_wafer_results": per_wafer_results,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report
