"""Reproducible labeled WM-811K extraction without die-level flattening."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.wafer_map_ingestion.loader import DEFAULT_PATH, load_wm811k
from src.wafer_map_features.batch_features import extract_batch_features
from src.wafer_map_features.select_labeled_wafers import select_labeled_wafers, extract_label

DEFAULT_OUTPUT = Path('data/processed/wm811k/features.parquet')


def process_frame(frame, output, *, per_class=None, seed=42, batch_size=1000):
    if batch_size <= 0 or (per_class is not None and per_class <= 0):
        raise ValueError('batch_size and per_class must be positive')
    selected = select_labeled_wafers(frame)
    if not selected.index.is_unique:
        raise ValueError('Wafer identifiers must be unique')
    if per_class is not None:
        selected = selected.groupby('defect_label', group_keys=False).sample(
            frac=1, random_state=seed).groupby('defect_label', group_keys=False).head(per_class)
    if selected.empty:
        raise ValueError('No labeled wafers selected')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.parquet.tmp')
    writer = None
    examples = {}
    try:
        for start in range(0, len(selected), batch_size):
            batch = selected.iloc[start:start + batch_size]
            features = extract_batch_features(batch)
            features['failureType'] = batch['defect_label'].to_numpy()
            features['lotName'] = batch['lotName'].astype('string').to_numpy() if 'lotName' in batch else pd.NA
            features['lotName'] = features['lotName'].astype('string')
            features['trianTestLabel'] = batch['trianTestLabel'].map(
                lambda value: str(np.asarray(value).reshape(-1)[0]) if np.asarray(value).size else ''
            ).to_numpy() if 'trianTestLabel' in batch else ''
            table = pa.Table.from_pandas(features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
            print(f'Extracted {min(start + batch_size, len(selected)):,}/{len(selected):,}', flush=True)
        writer.close()
        writer = None
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    for label, group in selected.groupby('defect_label'):
        examples[label] = np.asarray(group.iloc[0]['waferMap'])
    np.savez_compressed(output.with_suffix('.examples.npz'), **examples)
    manifest = {'source_rows': len(frame), 'labeled_rows': len(select_labeled_wafers(frame)),
                'selected_rows': len(selected), 'per_class_cap': per_class, 'seed': seed,
                'class_distribution': selected['defect_label'].value_counts().to_dict(),
                'schema_version': 1}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--per-class', type=int, default=None, help='Random cap per class; default all labeled')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=1000)
    args = parser.parse_args()
    frame = load_wm811k(args.input)
    print(json.dumps(process_frame(frame, args.output, per_class=args.per_class,
                                  seed=args.seed, batch_size=args.batch_size), indent=2))


if __name__ == '__main__':
    main()
