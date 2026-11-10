
import numpy as np
from src.wafer_map_features.spatial import spatial_features


def calculate_wafer_statistics(wafer_map):
    """
    Extract numerical features from a WM-811K wafer map.

    Die values:
        0 = empty position
        1 = passing die
        2 = failing die

    Returns a dictionary of wafer-level features.
    """

    wafer = np.asarray(wafer_map)

    if wafer.ndim != 2:
        raise ValueError("Wafer map must be two-dimensional.")

    if wafer.size == 0:
        raise ValueError("Wafer map cannot be empty.")

    if not np.isin(wafer, [0, 1, 2]).all():
        raise ValueError("Unexpected die values in wafer map.")

    # ------------------------------------------
    # Die counts
    # ------------------------------------------

    passing_dies = int(np.count_nonzero(wafer == 1))
    failing_dies = int(np.count_nonzero(wafer == 2))

    total_dies = passing_dies + failing_dies

    # ------------------------------------------
    # Yield and failure rate
    # ------------------------------------------

    if total_dies > 0:
        yield_rate = passing_dies / total_dies
        failure_rate = failing_dies / total_dies
    else:
        yield_rate = 0.0
        failure_rate = 0.0

    # ------------------------------------------
    # Wafer geometry
    # ------------------------------------------

    wafer_height, wafer_width = wafer.shape

    # ------------------------------------------
    # Return features
    # ------------------------------------------

    return {
        "wafer_height": int(wafer_height),
        "wafer_width": int(wafer_width),
        "total_dies": total_dies,
        "passing_dies": passing_dies,
        "failing_dies": failing_dies,
        "yield_rate": float(yield_rate),
        "failure_rate": float(failure_rate),
        **spatial_features(wafer),
    }
