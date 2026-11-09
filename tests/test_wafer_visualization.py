
import json

import pytest

from src.reporting.wafer_visualization import (
    generate_wafer_performance_chart,
)


def make_sample_report(tmp_path, wafers=None):
    """Create a sample JSON report for testing."""

    if wafers is None:
        wafers = [
            {
                "wafer_id": "LOT_001_WAFER_001",
                "precision": 1.0,
                "recall": 0.80,
                "f1_score": 0.89,
            },
            {
                "wafer_id": "LOT_001_WAFER_002",
                "precision": 1.0,
                "recall": 0.75,
                "f1_score": 0.86,
            },
        ]

    report = {
        "evaluation": {
            "per_wafer_results": wafers,
        }
    }

    json_path = tmp_path / "evaluation.json"
    json_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    return json_path


def test_chart_created(tmp_path):
    """Verify that a PNG chart is generated."""

    json_path = make_sample_report(tmp_path)
    output_path = tmp_path / "wafer_performance.png"

    result = generate_wafer_performance_chart(
        json_path,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_chart_is_valid_png(tmp_path):
    """Verify the output has a valid PNG signature."""

    json_path = make_sample_report(tmp_path)
    output_path = tmp_path / "chart.png"

    generate_wafer_performance_chart(
        json_path,
        output_path,
    )

    with output_path.open("rb") as file:
        signature = file.read(8)

    assert signature == b"\x89PNG\r\n\x1a\n"


def test_creates_nested_directories(tmp_path):
    """Verify missing output folders are created."""

    json_path = make_sample_report(tmp_path)

    output_path = (
        tmp_path
        / "reports"
        / "figures"
        / "wafer_performance.png"
    )

    generate_wafer_performance_chart(
        json_path,
        output_path,
    )

    assert output_path.exists()
    assert output_path.parent.is_dir()


def test_empty_wafer_results(tmp_path):
    """Verify that empty wafer results are rejected."""

    json_path = make_sample_report(
        tmp_path,
        wafers=[],
    )

    with pytest.raises(
        ValueError,
        match="No wafer results available",
    ):
        generate_wafer_performance_chart(
            json_path,
            tmp_path / "chart.png",
        )
