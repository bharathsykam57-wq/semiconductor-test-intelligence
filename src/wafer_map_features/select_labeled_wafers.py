"""Normalize legacy nested labels while retaining source wafer identifiers."""
import numpy as np

DEFECT_LABELS = ('Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none')


def extract_label(value):
    if value is None:
        return 'unlabeled'
    array = np.asarray(value)
    if array.size != 1:
        return 'unlabeled'
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode('utf-8')
    label = str(item).strip()
    return label if label in DEFECT_LABELS else 'unlabeled'


def select_labeled_wafers(df):
    if 'failureType' not in df:
        raise ValueError('Missing failureType column.')
    labels = df['failureType'].map(extract_label)
    selected = df.loc[labels != 'unlabeled'].copy()
    selected['defect_label'] = labels.loc[selected.index]
    return selected
