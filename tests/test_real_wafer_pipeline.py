import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.wafer_map_features.wafer_statistics import calculate_wafer_statistics
from src.wafer_map_features.select_labeled_wafers import extract_label, select_labeled_wafers
from src.wafer_map_features.batch_features import extract_batch_features
from src.wafer_map_features.process_labeled_batch import process_frame
from src.classification.wafer_classifier import FEATURE_COLUMNS, train_classifier, split_wafers


@pytest.mark.parametrize('wafer', [[], np.zeros((0, 3)), [1, 2], [[3]], [[np.nan]]])
def test_invalid_maps(wafer):
    with pytest.raises(ValueError):
        calculate_wafer_statistics(wafer)


def test_counts_geometry_and_spatial():
    wafer = np.ones((5, 5), dtype=int)
    wafer[2, 2] = 2
    f = calculate_wafer_statistics(wafer)
    assert f['total_dies'] == 25
    assert f['yield_rate'] == 24 / 25
    assert f['center_failure_density'] == 1 / 5
    assert f['edge_failure_density'] == 0
    assert f['failure_centroid_x'] == f['failure_centroid_y'] == 0
    assert f['inner_failure_fraction'] == 1
    assert f['adjacent_failure_fraction'] == 0
    edge = np.ones((5, 5), dtype=int)
    edge[0, :2] = 2
    f = calculate_wafer_statistics(edge)
    assert f['edge_failure_density'] > 0
    assert f['failure_centroid_x'] < 0 and f['failure_centroid_y'] == -1
    assert f['adjacent_failure_fraction'] == 1
    assert f['nw_failure_rate'] == 0.5


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (3, 8)])
def test_zero_tested_and_rectangular(shape):
    for value in [0, 1, 2]:
        f = calculate_wafer_statistics(np.full(shape, value))
        assert np.isfinite(list(f.values())).all()
        assert f['passing_dies'] + f['failing_dies'] == f['total_dies']
        assert 0 <= f['failure_rate'] <= 1


@pytest.mark.parametrize('value,label', [('Center', 'Center'), ([[b'none']], 'none'),
    (np.array([['Scratch']]), 'Scratch'), ([], 'unlabeled'), (None, 'unlabeled'),
    (np.nan, 'unlabeled'), ('unknown', 'unlabeled'), ([['']], 'unlabeled')])
def test_label_normalization(value, label):
    assert extract_label(value) == label


def fixture_frame():
    return pd.DataFrame({'waferMap': [np.array([[1, 2], [0, 1]]) if i % 2 else np.ones((3, 3), dtype=int) for i in range(48)],
        'failureType': [[['Center']] if i % 2 else [['none']] for i in range(48)],
        'lotName': [f'lot{i // 4}' for i in range(48)],
        'trianTestLabel': [[['Training' if i < 32 else 'Test']] for i in range(48)]}, index=range(100, 148))


def test_selection_and_batch():
    frame = fixture_frame()
    frame.at[100, 'failureType'] = []
    selected = select_labeled_wafers(frame)
    assert 100 not in selected.index and len(selected) == 47
    features = extract_batch_features(selected, max_wafers=3)
    assert features.wafer_id.tolist() == [101, 102, 103]
    with pytest.raises(ValueError):
        select_labeled_wafers(pd.DataFrame())
    with pytest.raises(ValueError):
        extract_batch_features(frame, max_wafers=-1)


def test_parquet_model_and_leakage(tmp_path):
    frame = fixture_frame()
    output = tmp_path / 'features.parquet'
    manifest = process_frame(frame, output, batch_size=7)
    features = pd.read_parquet(output)
    assert manifest['selected_rows'] == 48
    assert features.wafer_id.is_unique
    assert features[FEATURE_COLUMNS].notna().all().all()
    assert (features.passing_dies + features.failing_dies == features.total_dies).all()
    report = train_classifier(features, tmp_path / 'model', trees=5, n_jobs=1)
    assert sum(map(sum, report['confusion_matrix'])) == report['test_rows']
    partitions = pd.read_parquet(tmp_path / 'model/partitions.parquet')
    assert not set(partitions.query("partition == 'train'").lotName) & set(partitions.query("partition == 'test'").lotName)
    bundle = joblib.load(tmp_path / 'model/model.joblib')
    assert len(bundle['model'].predict(features[FEATURE_COLUMNS])) == 48
    assert json.loads((tmp_path / 'model/evaluation.json').read_text())['lot_overlap'] == 0
    features.loc[0, 'trianTestLabel'] = 'Test'
    train, test, removed = split_wafers(features, split='metadata')
    assert removed == 3
    assert not set(features.iloc[train].lotName) & set(features.iloc[test].lotName)
    features.loc[0, 'failureType'] = 'unlabeled'
    with pytest.raises(ValueError):
        train_classifier(features, tmp_path / 'bad', trees=2)


def test_sampling_repeatable(tmp_path):
    a, b = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    process_frame(fixture_frame(), a, per_class=5, seed=7, batch_size=3)
    process_frame(fixture_frame(), b, per_class=5, seed=7, batch_size=4)
    pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b))
    assert len(pd.read_parquet(a)) == 10
    with pytest.raises(ValueError):
        process_frame(fixture_frame(), a, batch_size=0)
