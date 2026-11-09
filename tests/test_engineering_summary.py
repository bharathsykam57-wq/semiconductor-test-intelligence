
import json

from src.reporting.engineering_summary import (
    generate_engineering_summary,
)


def make_sample_report(tmp_path):
    """Create a sample JSON evaluation report."""

    report = {
        "data_source": "synthetic",
        "model": {
            "algorithm": "IsolationForest",
            "contamination": 0.10,
            "random_state": 42,
        },
        "evaluation": {
            "method": "leave_one_wafer_out",
            "overall_metrics": {
                "total_devices": 100,
                "true_positives": 7,
                "true_negatives": 91,
                "false_positives": 0,
                "false_negatives": 2,
                "precision": 1.0,
                "recall": 7 / 9,
                "f1_score": 0.875,
            },
            "per_wafer_results": [
                {
                    "wafer_id": "LOT_001_WAFER_001",
                    "total_devices": 100,
                    "precision": 1.0,
                    "recall": 7 / 9,
                    "f1_score": 0.875,
                }
            ],
        },
    }

    json_path = tmp_path / "evaluation.json"
    json_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    return json_path


def test_markdown_report_created(tmp_path):
    """Verify that the Markdown report is created."""

    json_path = make_sample_report(tmp_path)
    output_path = tmp_path / "engineering_report.md"

    result = generate_engineering_summary(
        json_path,
        output_path,
    )

    assert output_path.exists()
    assert result == output_path


def test_report_contains_metrics(tmp_path):
    """Verify that the report contains correct metrics."""

    json_path = make_sample_report(tmp_path)
    output_path = tmp_path / "report.md"

    generate_engineering_summary(json_path, output_path)

    content = output_path.read_text(encoding="utf-8")

    assert "100.00%" in content
    assert "77.78%" in content
    assert "87.50%" in content
    assert "True positives | 7" in content
    assert "False negatives | 2" in content


def test_report_contains_wafer_details(tmp_path):
    """Verify that wafer-level results are included."""

    json_path = make_sample_report(tmp_path)
    output_path = tmp_path / "report.md"

    generate_engineering_summary(json_path, output_path)

    content = output_path.read_text(encoding="utf-8")

    assert "LOT_001_WAFER_001" in content
    assert "Per-Wafer Performance" in content


def test_report_contains_limitations(tmp_path):
    """Verify that important limitations are documented."""

    json_path = make_sample_report(tmp_path)
    output_path = tmp_path / "report.md"

    generate_engineering_summary(json_path, output_path)

    content = output_path.read_text(encoding="utf-8")

    assert "All evaluation data is synthetic" in content
    assert "unseen manufacturing lots" in content
    assert "engineering investigation" in content


def test_creates_output_directory(tmp_path):
    """Verify that missing directories are created."""

    json_path = make_sample_report(tmp_path)

    output_path = (
        tmp_path
        / "reports"
        / "engineering"
        / "summary.md"
    )

    generate_engineering_summary(json_path, output_path)

    assert output_path.exists()
    
    

def test_report_embeds_existing_chart(tmp_path):
    """Verify an existing chart is linked correctly in Markdown."""

    json_path = make_sample_report(tmp_path)

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    chart_path = figures_dir / "wafer_performance.png"
    chart_path.write_bytes(b"sample chart data")

    output_path = tmp_path / "engineering_report.md"

    generate_engineering_summary(
        json_path,
        output_path,
    )

    content = output_path.read_text(encoding="utf-8")

    expected_link = (
        "![Wafer-level model performance]"
        "(figures/wafer_performance.png)"
    )

    assert "## 7. Wafer Performance Visualization" in content
    assert expected_link in content

