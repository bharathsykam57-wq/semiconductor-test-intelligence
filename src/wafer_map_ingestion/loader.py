"""Loading trusted, legacy WM-811K pickle files (pickle can execute code)."""
import sys
from pathlib import Path
import pandas as pd
import pandas.core.indexes
import pandas.core.indexes.base
from pandas.compat import pickle_compat

DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'data/external/wm811k/LSWMD.pkl'


def load_wm811k(path=DEFAULT_PATH):
    """Load one legacy pickle; this format cannot be streamed by wafer."""
    sys.modules.setdefault('pandas.indexes', pandas.core.indexes)
    sys.modules.setdefault('pandas.indexes.base', pandas.core.indexes.base)
    with open(path, 'rb') as handle:
        frame = pickle_compat.Unpickler(handle, encoding='latin1').load()
    if not isinstance(frame, pd.DataFrame) or 'waferMap' not in frame:
        raise ValueError('Expected a DataFrame with waferMap column')
    if not frame.index.is_unique:
        raise ValueError('Source wafer indices must be unique')
    return frame
