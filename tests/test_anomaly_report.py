
import json

from src.reporting.anomaly_report import generate_anomaly_report


def sample_evaluation_results():
    """Create sample wafer-level evaluation results."""

    per_wafer = [
        {
            "lot_id": "LOT_001",
            "wafer_id": "LOT_001_WAFER_001",
            "total_devices": 100,
            "true_positives": 7,
            "true_negatives": 91,
            "false_positives": 0,
            "false_negatives": 2,
            "precision": 1.0,
            "recall": 7 / 9,
            "f1_score": 0.875,
        }
    ]

    overall = {
        "total_devices": 100,
        "true_positives": 7,
        "true_negatives": 91,
        "false_positives": 0,
        "false_negatives": 2,
        "precision": 1.0,
        "recall": 7 / 9,
        "f1_score": 0.875,
    }

    return per_wafer, overall


def test_report_file_created(tmp_path):
    """Verify that the JSON report is created."""

    per_wafer, overall = sample_evaluation_results()

    output_path = tmp_path / "anomaly_report.json"

    generate_anomaly_report(
        per_wafer,
        overall,
        output_path,
    )

    assert output_path.exists()


def test_report_contains_correct_metrics(tmp_path):
    """Verify that evaluation metrics are saved correctly."""

    per_wafer, overall = sample_evaluation_results()

    output_path = tmp_path / "report.json"

    generate_anomaly_report(
        per_wafer,
        overall,
        output_path,
    )

    with output_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    assert (
        report["evaluation"]["overall_metrics"]["true_positives"]
        == 7
    )
    assert (
        report["evaluation"]["overall_metrics"]["false_negatives"]
        == 2
    )
    assert len(report["evaluation"]["per_wafer_results"]) == 1


def test_report_metadata(tmp_path):
    """Verify model configuration and data provenance."""

    per_wafer, overall = sample_evaluation_results()

    report = generate_anomaly_report(
        per_wafer,
        overall,
        tmp_path / "report.json",
        contamination=0.10,
        random_state=42,
    )

    assert report["data_source"] == "synthetic"
    assert report["report_type"] == "wafer_level_anomaly_evaluation"
    assert report["model"]["algorithm"] == "IsolationForest"
    assert report["model"]["contamination"] == 0.10
    assert report["model"]["random_state"] == 42
    assert report["evaluation"]["method"] == "leave_one_wafer_out"
    assert "generated_at_utc" in report


def test_creates_nested_directories(tmp_path):
    """Verify that missing output directories are created."""

    per_wafer, overall = sample_evaluation_results()

    output_path = (
        tmp_path
        / "engineering"
        / "anomaly"
        / "report.json"
    )

    generate_anomaly_report(
        per_wafer,
        overall,
        output_path,
    )

    assert output_path.exists()
    assert output_path.parent.is_dir()
