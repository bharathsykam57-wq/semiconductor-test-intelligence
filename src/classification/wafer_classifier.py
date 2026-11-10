"""Lot-disjoint or metadata-held-out WM-811K random forest baseline."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from src.wafer_map_features.wafer_statistics import calculate_wafer_statistics
from src.wafer_map_features.select_labeled_wafers import DEFECT_LABELS

FEATURE_COLUMNS = list(calculate_wafer_statistics(np.ones((2, 2))).keys())


def split_wafers(frame, *, split='grouped', seed=42, test_size=0.2):
    if frame['lotName'].isna().any() or frame['lotName'].astype(str).str.strip().eq('').any():
        raise ValueError('Nonempty lotName is required to audit lot leakage')
    groups = frame['lotName'].astype(str)
    if split == 'grouped':
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(frame, groups=groups))
        removed = 0
    elif split == 'metadata':
        metadata = frame['trianTestLabel'].str.lower()
        if not metadata.isin(['training', 'test']).all():
            raise ValueError('metadata split requires Training/Test labels on every row')
        test = np.flatnonzero(metadata.eq('test'))
        candidate = metadata.eq('training')
        # Preserve official test membership, purge any training lot also in test.
        train = np.flatnonzero(candidate & ~groups.isin(groups.iloc[test]))
        removed = int(candidate.sum() - len(train))
    else:
        raise ValueError('split must be grouped or metadata')
    if not len(train) or not len(test):
        raise ValueError('Split has an empty training or test partition')
    overlap = set(groups.iloc[train]) & set(groups.iloc[test])
    if overlap:
        raise ValueError('Lot leakage detected')
    return train, test, removed


def train_classifier(frame, output_dir, *, split='grouped', seed=42, test_size=0.2,
                     trees=200, min_samples_leaf=2, n_jobs=2):
    required = set(FEATURE_COLUMNS + ['wafer_id', 'failureType', 'lotName'])
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if frame.empty or frame['wafer_id'].isna().any() or not frame['wafer_id'].is_unique:
        raise ValueError('Expected nonempty unique wafer identifiers')
    if not frame['failureType'].isin(DEFECT_LABELS).all():
        raise ValueError('Only known labeled wafers can be used for training')
    x = frame[FEATURE_COLUMNS].astype(float)
    if np.isinf(x.to_numpy()).any():
        raise ValueError('Infinite feature values')
    train, test, removed = split_wafers(frame, split=split, seed=seed, test_size=test_size)
    y = frame['failureType']
    if y.iloc[train].nunique() < 2:
        raise ValueError('Training requires at least two classes')
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          RandomForestClassifier(n_estimators=trees, min_samples_leaf=min_samples_leaf,
                              class_weight='balanced_subsample', random_state=seed, n_jobs=n_jobs))
    model.fit(x.iloc[train], y.iloc[train])
    prediction = model.predict(x.iloc[test])
    dummy = DummyClassifier(strategy='most_frequent').fit(x.iloc[train], y.iloc[train])
    labels = sorted(y.unique())

    def metrics(pred):
        return {'accuracy': accuracy_score(y.iloc[test], pred),
                'macro_f1': f1_score(y.iloc[test], pred, labels=labels, average='macro', zero_division=0),
                'weighted_f1': f1_score(y.iloc[test], pred, labels=labels, average='weighted', zero_division=0)}

    official_overlap = None
    if 'trianTestLabel' in frame:
        metadata = frame['trianTestLabel'].str.lower()
        official_overlap = len(set(frame.loc[metadata.eq('training'), 'lotName']) &
                               set(frame.loc[metadata.eq('test'), 'lotName']))
    report = {'schema_version': 1, 'split': split, 'seed': seed, 'test_size_requested': test_size,
              'parameters': {'trees': trees, 'min_samples_leaf': min_samples_leaf, 'n_jobs': n_jobs},
              'train_rows': len(train), 'test_rows': len(test), 'purged_training_rows': removed,
              'train_lots': int(frame.iloc[train]['lotName'].nunique()),
              'test_lots': int(frame.iloc[test]['lotName'].nunique()), 'lot_overlap': 0,
              'official_metadata_overlapping_lots': official_overlap,
              'metrics': metrics(prediction), 'baseline': metrics(dummy.predict(x.iloc[test])),
              'labels': labels, 'confusion_matrix': confusion_matrix(y.iloc[test], prediction, labels=labels).tolist(),
              'per_class': classification_report(y.iloc[test], prediction, labels=labels, output_dict=True, zero_division=0),
              'class_distribution': {name: y.iloc[ids].value_counts().to_dict() for name, ids in [('train', train), ('test', test)]},
              'classes_missing_from_train': sorted(set(labels) - set(y.iloc[train])),
              'classes_missing_from_test': sorted(set(labels) - set(y.iloc[test])),
              'feature_importance': dict(zip(FEATURE_COLUMNS, model[-1].feature_importances_.tolist()))}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_columns': FEATURE_COLUMNS, 'report': report}, output / 'model.joblib')
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2))
    partitions = frame[['wafer_id', 'lotName', 'failureType']].copy()
    partitions['partition'] = 'purged'
    partitions.loc[partitions.index[train], 'partition'] = 'train'
    partitions.loc[partitions.index[test], 'partition'] = 'test'
    partitions.to_parquet(output / 'partitions.parquet', index=False)
    pd.DataFrame({'wafer_id': frame.iloc[test]['wafer_id'].to_numpy(), 'actual': y.iloc[test].to_numpy(),
                  'predicted': prediction}).to_parquet(output / 'predictions.parquet', index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/wm811k/features.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/wm811k/model'))
    parser.add_argument('--split', choices=['grouped', 'metadata'], default='grouped')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--trees', type=int, default=200)
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    parser.add_argument('--n-jobs', type=int, default=2)
    args = parser.parse_args()
    report = train_classifier(pd.read_parquet(args.input), args.output_dir, split=args.split, seed=args.seed,
                              test_size=args.test_size, trees=args.trees, min_samples_leaf=args.min_samples_leaf, n_jobs=args.n_jobs)
    print(json.dumps({'metrics': report['metrics'], 'baseline': report['baseline'],
                      'train_rows': report['train_rows'], 'test_rows': report['test_rows']}, indent=2))


if __name__ == '__main__':
    main()
