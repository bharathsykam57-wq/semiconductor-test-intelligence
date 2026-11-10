
from pathlib import Path
import gc
import os
import sys

import numpy as np
import pandas as pd
import psutil

# ==================================================
# Compatibility with older pandas Pickle files
# ==================================================

import pandas.core.indexes
import pandas.core.indexes.base

sys.modules["pandas.indexes"] = pandas.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base


# ==================================================
# Project configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "wm811k"
    / "LSWMD.pkl"
)


# ==================================================
# Memory monitoring
# ==================================================

def show_memory(stage):
    process = psutil.Process(os.getpid())

    memory_gb = (
        process.memory_info().rss / (1024 ** 3)
    )

    print(f"{stage}: {memory_gb:.2f} GB RAM", flush=True)


# ==================================================
# Dataset inspection
# ==================================================

def main():

    print("WM-811K DATASET INSPECTION", flush=True)
    print("-------------------------", flush=True)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    file_size_gb = (
        DATA_PATH.stat().st_size / (1024 ** 3)
    )

    print(f"Dataset: {DATA_PATH.name}")
    print(f"File size: {file_size_gb:.2f} GB")

    show_memory("Before loading")

    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------

    print("\nLoading dataset...", flush=True)

    from pandas.compat import pickle_compat

    with open(DATA_PATH, "rb") as file:
        df = pickle_compat.Unpickler(
            file,
            encoding="latin1",
        ).load()

    show_memory("After loading")

    # ----------------------------------------------
    # Basic structure
    # ----------------------------------------------

    print("\nDATASET STRUCTURE")
    print("-----------------")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nTotal wafer records:", len(df))

    # ----------------------------------------------
    # Wafer map inspection
    # ----------------------------------------------

    if "waferMap" in df.columns:

        print("\nWAFER MAP INFORMATION")
        print("---------------------")

        first_map = np.asarray(df["waferMap"].iloc[0])

        print("First wafer-map shape:", first_map.shape)

        print("First wafer-map values:")
        print(np.unique(first_map))

        # Count die observations without flattening
        # every wafer map into a large new array.

        print("\nCounting die observations...", flush=True)

        total_die_positions = 0
        total_tested_dies = 0

        for wafer_map in df["waferMap"]:
            wafer_array = np.asarray(wafer_map)

            total_die_positions += wafer_array.size

            # In WM-811K, zero typically represents
            # an unoccupied/non-die position.
            # Values 1 and 2 represent tested dies.
            total_tested_dies += np.count_nonzero(
                wafer_array
            )

        print(
            "Total die-map positions:",
            f"{total_die_positions:,}"
        )

        print(
            "Total nonzero die observations:",
            f"{total_tested_dies:,}"
        )

        if total_tested_dies >= 3_000_000:
            print(
                "\n3 MILLION OBSERVATION TARGET: PASSED"
            )
        else:
            print(
                "\n3 MILLION OBSERVATION TARGET: "
                "NOT REACHED"
            )

    # ----------------------------------------------
    # Defect labels
    # ----------------------------------------------

    if "failureType" in df.columns:

        print("\nDEFECT LABEL INFORMATION")
        print("------------------------")

        print(
            "Failure type column available:",
            True
        )

        print(
            "Example failure label:",
            df["failureType"].iloc[0]
        )

        print("\nFailure type distribution:")

        labels = df["failureType"].apply(
            lambda value: str(value[0][0])
            if len(value) > 0 and len(value[0]) > 0
            else "unlabeled"
        )

        print(labels.value_counts())

    # ----------------------------------------------
    # Cleanup
    # ----------------------------------------------

    print("\nCleaning up...", flush=True)

    del df
    gc.collect()

    show_memory("After cleanup")

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()
