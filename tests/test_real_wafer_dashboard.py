import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.wafer_map_features.process_labeled_batch import process_frame
from src.classification.wafer_classifier import train_classifier
from dashboard.real_wafers import category_name, feature_name, yield_chart, confusion_chart


APP_PATH = Path(__file__).resolve().parents[1] / 'dashboard/app.py'


def real_app(directory):
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state['analysis_workflow'] = 'Real WM-811K wafer maps'
    app.session_state['wafer_artifact_directory'] = str(directory)
    return app.run()


@pytest.fixture(scope='module')
def fixture_artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp('dashboard_artifacts')
    frame = pd.DataFrame({'waferMap': [np.array([[1, 2], [1, 1]]) for _ in range(40)],
        'failureType': ['Center', 'none'] * 20, 'lotName': [f'lot{i // 4}' for i in range(40)]})
    process_frame(frame, directory / 'features.parquet')
    features = pd.read_parquet(directory / 'features.parquet')
    # Tiny existing model fixture only; full-data models are never trained here.
    train_classifier(features, directory / 'model', trees=2, n_jobs=1)
    return directory


def test_real_view_missing_artifacts(tmp_path):
    app = real_app(tmp_path)
    assert not app.exception
    assert any('Real WM-811K' in item.value for item in app.header)
    assert app.info


def test_real_view_with_fixture_artifacts(fixture_artifacts):
    app = real_app(fixture_artifacts)
    assert not app.exception
    assert not app.error
    assert any('Defect classification' in item.value for item in app.subheader)
    assert any('Confusion matrix' in item.value for item in app.subheader)
    assert [tab.label for tab in app.tabs] == ['Dataset & wafer maps', 'Model evaluation', 'Features & limitations']
    report = json.loads((fixture_artifacts / 'model/evaluation.json').read_text())
    assert next(item.value for item in app.metric if item.label == 'Accuracy') == f"{report['metrics']['accuracy']:.1%}"
    assert next(item.value for item in app.metric if item.label == 'Macro F1') == f"{report['metrics']['macro_f1']:.3f}"
    assert any('Artifact provenance' == item.label and not item.proto.expanded for item in app.expander)


def test_split_selection_uses_selected_artifact(fixture_artifacts, tmp_path):
    shutil.copytree(fixture_artifacts, tmp_path, dirs_exist_ok=True)
    # Synthetic UI fixture: distinct scores prove the selector changes its source.
    report = json.loads((tmp_path / 'model/evaluation.json').read_text())
    report['split'] = 'metadata'
    report['metrics']['macro_f1'] = 0.123
    (tmp_path / 'model_official').mkdir()
    (tmp_path / 'model_official/evaluation.json').write_text(json.dumps(report))
    app = real_app(tmp_path)
    app.selectbox[0].set_value('Official split · model_official').run()
    assert not app.exception and not app.error
    assert next(item.value for item in app.metric if item.label == 'Macro F1') == '0.123'
    assert any('Official Training/Test assignment' in item.value for item in app.caption)
    assert any('Lot-leakage audit: 0' in item.value for item in app.caption)


def test_leaking_artifact_is_not_presented(fixture_artifacts, tmp_path):
    shutil.copytree(fixture_artifacts, tmp_path, dirs_exist_ok=True)
    path = tmp_path / 'model/evaluation.json'
    report = json.loads(path.read_text())
    report['lot_overlap'] = 1
    path.write_text(json.dumps(report))
    app = real_app(tmp_path)
    assert not app.exception
    assert any('shared training/test lots' in item.value for item in app.error)
    assert not app.metric


def test_yield_axis_preserves_units_and_excludes_untested_maps():
    chart = yield_chart(pd.DataFrame({'total_dies': [10, 10, 0], 'yield_rate': [0.5, 1.0, 0]}))
    spec = chart.to_dict()
    assert spec['encoding']['x']['axis']['format'] == '.0%'
    assert spec['encoding']['x']['scale']['domain'] == [0, 1]
    assert chart.data.Wafers.sum() == 2
    assert chart.data.loc[chart.data.Wafers.gt(0), 'Yield'].tolist() == pytest.approx([0.5, 0.95])


def test_confusion_chart_preserves_saved_counts(fixture_artifacts):
    report = json.loads((fixture_artifacts / 'model/evaluation.json').read_text())
    chart = confusion_chart(report)
    spec = chart.to_dict()
    records = next(iter(spec['datasets'].values()))
    assert [record['Wafers'] for record in records] == [value for row in report['confusion_matrix'] for value in row]
    assert sum(record['Wafers'] for record in records) == report['test_rows']
    assert category_name('none') == 'Normal'
    assert category_name('Edge-Loc') == 'Edge localized'
    assert feature_name('nw_failure_rate') == 'NW failure rate'
