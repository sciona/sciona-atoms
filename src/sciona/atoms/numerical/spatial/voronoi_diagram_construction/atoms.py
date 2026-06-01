from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_voronoi_raw,
    witness_clip_voronoi_boundaries,
)

@register_atom(witness_compute_voronoi_raw, name="compute_voronoi_raw")
@icontract.require(lambda points: points.ndim == 2, "Precondition failed: points.ndim == 2")
@icontract.require(lambda points: points.shape[0] >= points.shape[1] + 1, "Precondition failed: points.shape[0] >= points.shape[1] + 1")
@icontract.ensure(lambda result, points: vertices.ndim == 2, "Postcondition failed: vertices.ndim == 2")
def compute_voronoi_raw(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute raw Voronoi vertices, cells, and ridge graphs from generator coordinates.

    Args:
        points: NDArray[np.float64]

    Returns:
        vertices: NDArray[np.float64]
    """
    import scipy.spatial
    return scipy.spatial.Voronoi(points=points) # type: ignore

@register_atom(witness_clip_voronoi_boundaries, name="clip_voronoi_boundaries")
@icontract.require(lambda vertices, regions, points, point_region, bbox: bbox.shape == (2, points.shape[1]), "Precondition failed: bbox.shape == (2, points.shape[1])")
@icontract.ensure(lambda result, vertices, regions, points, point_region, bbox: result is not None, "Postcondition failed: result is not None")
def clip_voronoi_boundaries(vertices: NDArray[np.float64], regions: int, points: NDArray[np.float64], point_region: NDArray[np.int32], bbox: NDArray[np.float64]) -> float:
    """Clip open infinite Voronoi regions using a spatial bounding box.

    Args:
        vertices: NDArray[np.float64]
        regions: list[list[int]]
        points: NDArray[np.float64]
        point_region: NDArray[np.int32]
        bbox: NDArray[np.float64]

    Returns:
        clipped_regions: list[list[float]]
    """
    import scipy.spatial
    return scipy.spatial.Voronoi(vertices=vertices, regions=regions, points=points, point_region=point_region, bbox=bbox) # type: ignore

