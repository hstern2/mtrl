from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from trl.objectives.pareto import nsga2_sort


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute 2D hypervolume indicator.

    Args:
        points: (n, 2) array of objective values (maximized).
        ref_point: (2,) reference point (should be dominated by all Pareto points).

    Returns:
        Hypervolume value.
    """
    # Filter points that dominate the reference
    mask = np.all(points > ref_point, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0

    # Sort by first objective
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    hv = 0.0
    prev_y = ref_point[1]
    for i in range(len(pts)):
        hv += (pts[i, 0] - ref_point[0]) * (pts[i, 1] - prev_y)
        prev_y = pts[i, 1]

    return float(hv)


def pareto_front_indices(scores: np.ndarray) -> list[int]:
    """Return indices of Pareto-optimal points."""
    fronts, _ = nsga2_sort(scores)
    return fronts[0] if fronts else []
