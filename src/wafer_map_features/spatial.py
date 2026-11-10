"""Dimension-normalized spatial features; definitions in docs/wafer_features.md."""
import numpy as np


def spatial_features(wafer):
    tested, failed = wafer > 0, wafer == 2
    h, w = wafer.shape
    y, x = np.indices(wafer.shape, dtype=float)
    x = (x - (w - 1) / 2) / max((w - 1) / 2, 1)
    y = (y - (h - 1) / 2) / max((h - 1) / 2, 1)
    radius = np.hypot(x, y)

    def density(mask):
        count = np.count_nonzero(tested & mask)
        return float(np.count_nonzero(failed & mask) / count) if count else 0.0

    result = {
        'center_failure_density': density(radius <= 0.5),
        'edge_failure_density': density(radius >= 0.8),
    }
    for name, mask in {'nw': (x < 0) & (y < 0), 'ne': (x >= 0) & (y < 0),
                       'sw': (x < 0) & (y >= 0), 'se': (x >= 0) & (y >= 0)}.items():
        result[f'{name}_failure_rate'] = density(mask)
    n = np.count_nonzero(failed)
    result.update({
        'failure_centroid_x': float(x[failed].mean()) if n else 0.0,
        'failure_centroid_y': float(y[failed].mean()) if n else 0.0,
        'failure_radius_mean': float(radius[failed].mean()) if n else 0.0,
        'failure_radius_std': float(radius[failed].std()) if n else 0.0,
        'failure_spread': float(x[failed].var() + y[failed].var()) if n else 0.0,
    })
    for name, mask in {'inner': radius <= 0.5, 'middle': (radius > 0.5) & (radius < 0.8),
                       'outer': radius >= 0.8}.items():
        result[f'{name}_failure_fraction'] = float(np.count_nonzero(failed & mask) / n) if n else 0.0
    adjacent = np.zeros_like(failed)
    adjacent[1:] |= failed[:-1]
    adjacent[:-1] |= failed[1:]
    adjacent[:, 1:] |= failed[:, :-1]
    adjacent[:, :-1] |= failed[:, 1:]
    result['adjacent_failure_fraction'] = float(np.count_nonzero(failed & adjacent) / n) if n else 0.0
    return result
