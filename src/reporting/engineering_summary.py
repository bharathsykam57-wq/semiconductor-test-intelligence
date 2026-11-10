
import json
from pathlib import Path


def generate_engineering_summary(json_path, output_path):
    """
    Convert a wafer-level anomaly evaluation JSON report
    into a human-readable Markdown engineering summary.
    """

    with Path(json_path).open("r", encoding="utf-8") as file:
        report = json.load(file)

    model = report["model"]
    evaluation = report["evaluation"]
    overall = evaluation["overall_metrics"]
    wafers = evaluation["per_wafer_results"]

    lines = [
        "# Semiconductor Anomaly Detection — Engineering Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Data source: {report['data_source']}",
        f"- Model: {model['algorithm']}",
        f"- Evaluation method: {evaluation['method']}",
        f"- Devices evaluated: {overall['total_devices']}",
        f"- Wafers evaluated: {len(wafers)}",
        "",
        "## 2. Overall Model Performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {overall['precision']:.2%} |",
        f"| Recall | {overall['recall']:.2%} |",
        f"| F1-score | {overall['f1_score']:.2%} |",
        f"| True positives | {overall['true_positives']} |",
        f"| True negatives | {overall['true_negatives']} |",
        f"| False positives | {overall['false_positives']} |",
        f"| False negatives | {overall['false_negatives']} |",
        "",
        "## 3. Per-Wafer Performance",
        "",
        "| Wafer | Devices | Precision | Recall | F1-score |",
        "|---|---:|---:|---:|---:|",
    ]

    for wafer in wafers:
        lines.append(
            f"| {wafer['wafer_id']} "
            f"| {wafer['total_devices']} "
            f"| {wafer['precision']:.2%} "
            f"| {wafer['recall']:.2%} "
            f"| {wafer['f1_score']:.2%} |"
        )

    lines.extend([
        "",
        "## 4. Engineering Interpretation",
        "",
        (
            f"The model flagged {overall['true_positives']} devices "
            "with recorded failures and missed "
            f"{overall['false_negatives']} failure-associated devices."
        ),
        "",
        (
            "The anomaly detector is intended to prioritize devices "
            "for engineering investigation, not to replace "
            "semiconductor pass/fail testing."
        ),
        "",
        "## 5. Limitations",
        "",
        "- All evaluation data is synthetic.",
        "- Results do not establish performance on real manufacturing data.",
        "- Evaluation uses recorded failures as reference labels.",
        "- All evaluated wafers belong to the same manufacturing lot.",
        "- Model configuration was explored using this dataset.",
        "- Performance on unseen manufacturing lots remains unknown.",
        "",
        "## 6. Model Configuration",
        "",
        f"- Algorithm: {model['algorithm']}",
        f"- Contamination: {model['contamination']}",
        f"- Random state: {model['random_state']}",
        f"- Evaluation: {evaluation['method']}",
        "",
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_path = output_path.parent / "figures" / "wafer_performance.png"

    if chart_path.is_file():
        lines.extend([
            "## 7. Wafer Performance Visualization",
            "",
            "The chart compares precision, recall, and F1-score",
            "across the evaluated synthetic wafers.",
            "",
            "![Wafer-level model performance](figures/wafer_performance.png)",
            "",
        ])
        
    device_chart_path = (
        output_path.parent
        / "figures"
        / "device_anomaly_scores.png"
    )

    if device_chart_path.is_file():
        lines.extend([
            "",
            "## 8. Device-Level Anomaly Score Distribution",
            "",
            (
                "The following chart shows device-level Isolation "
                "Forest anomaly scores from the synthetic dataset."
            ),
            "",
            (
                "Devices are ranked by anomaly score. "
                "Higher scores indicate more unusual electrical "
                "behavior, and scores above zero are flagged."
            ),
            "",
            (
                "These are exploratory in-sample anomaly scores, "
                "not held-out evaluation predictions."
            ),
            "",
            "![Device anomaly scores](figures/device_anomaly_scores.png)",
            "",
        ])


    output_path.write_text("\n".join(lines), encoding="utf-8")

    return output_path
