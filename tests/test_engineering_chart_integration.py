
import json

from src.reporting.engineering_summary import (
    generate_engineering_summary,
)


def make_report(tmp_path, create_wafer=True, create_device=True):
    """Generate a report with optional chart files."""

    report_dir = tmp_path / "reports"
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True)

    if create_wafer:
        (figures_dir / "wafer_performance.png").touch()

    if create_device:
        (figures_dir / "device_anomaly_scores.png").touch()

    source_json = tmp_path / "anomaly_evaluation.json"
    source_json.write_text(
        (
            __import__("pathlib").Path(
                "reports/anomaly_evaluation.json"
            ).read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )

    output_path = report_dir / "engineering_summary.md"

    generate_engineering_summary(
        source_json,
        output_path,
    )

    return output_path.read_text(encoding="utf-8")


def test_both_charts_embedded(tmp_path):
    """Both chart links should appear when files exist."""

    content = make_report(tmp_path)

    assert (
        "![Wafer-level model performance]"
        "(figures/wafer_performance.png)"
    ) in content

    assert (
        "![Device anomaly scores]"
        "(figures/device_anomaly_scores.png)"
    ) in content


def test_device_chart_not_embedded_when_missing(tmp_path):
    """Missing device chart should not create a broken image link."""

    content = make_report(
        tmp_path,
        create_device=False,
    )

    assert (
        "![Device anomaly scores]"
        "(figures/device_anomaly_scores.png)"
    ) not in content


def test_wafer_chart_not_embedded_when_missing(tmp_path):
    """Missing wafer chart should not create a broken image link."""

    content = make_report(
        tmp_path,
        create_wafer=False,
    )

    assert (
        "![Wafer-level model performance]"
        "(figures/wafer_performance.png)"
    ) not in content
