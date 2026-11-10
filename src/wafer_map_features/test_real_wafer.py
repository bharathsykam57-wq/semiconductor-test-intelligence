
import sys
import pandas as pd
import pandas.core.indexes
import pandas.core.indexes.base

from pathlib import Path
from pandas.compat import pickle_compat

from src.wafer_map_features.wafer_statistics import (
    calculate_wafer_statistics,
)

# Compatibility with the older WM-811K pickle format
sys.modules["pandas.indexes"] = pandas.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base

dataset_path = (
    Path(__file__).resolve().parents[2]
    / "data/external/wm811k/LSWMD.pkl"
)

def main():
    print("Loading real WM-811K dataset...")

    with open(dataset_path, "rb") as file:
        df = pickle_compat.Unpickler(
            file, encoding="latin1"
        ).load()

    print("Dataset loaded successfully.")

    # Select the first real wafer
    wafer_map = df["waferMap"].iloc[0]

    # Extract numerical features
    features = calculate_wafer_statistics(wafer_map)

    print("\nREAL WAFER FEATURES")
    print("-------------------")

    for name, value in features.items():
        print(f"{name}: {value}")

    print("\nREAL WAFER FEATURE EXTRACTION PASSED")


if __name__ == "__main__":
    main()
