
import sys
from pathlib import Path

import pandas as pd
import pandas.core.indexes
import pandas.core.indexes.base
from pandas.compat import pickle_compat

from src.wafer_map_features.batch_features import (
    extract_batch_features,
)

# Compatibility with the legacy WM-811K pickle format
sys.modules["pandas.indexes"] = pandas.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data/external/wm811k/LSWMD.pkl"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data/processed/wm811k_100_features.parquet"
)


def extract_label(value):
    """Extract the defect label from a WM-811K label array."""
    if len(value) > 0 and len(value[0]) > 0:
        return str(value[0][0])

    return "unlabeled"


def main():
    print("Loading WM-811K dataset...", flush=True)

    # Load the original dataset
    with open(DATA_PATH, "rb") as file:
        df = pickle_compat.Unpickler(
            file, encoding="latin1"
        ).load()

    print(f"Loaded {len(df):,} wafers")

    # Extract numerical features
    print("\nExtracting features from 100 real wafers...")

    features = extract_batch_features(
        df,
        max_wafers=100,
    )

    # Attach the original defect labels
    features["failureType"] = (
        df.loc[features["wafer_id"], "failureType"]
        .apply(extract_label)
        .to_numpy()
    )

    # Validate the extracted features
    assert len(features) == 100
    assert "failure_rate" in features.columns

    assert (
        features["passing_dies"] + features["failing_dies"]
        == features["total_dies"]
    ).all()

    assert features["wafer_id"].is_unique
    assert features["failureType"].notna().all()

    # Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save features and labels
    features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # Display results
    print("\nFEATURE EXTRACTION RESULTS")
    print("--------------------------")
    print(features.head().to_string(index=False))

    print("\nDEFECT LABEL DISTRIBUTION")
    print("-------------------------")
    print(features["failureType"].value_counts())

    print("\nPIPELINE SUMMARY")
    print("----------------")
    print(f"Wafers processed: {len(features)}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nREAL BATCH PROCESSING WITH LABELS PASSED")


if __name__ == "__main__":
    main()
