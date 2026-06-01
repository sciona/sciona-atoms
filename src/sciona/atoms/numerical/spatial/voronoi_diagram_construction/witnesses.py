from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_voronoi_raw(points: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_voronoi_raw."""
    _ = (points)
    return AbstractArray(shape=points.shape, dtype=points.dtype)

def witness_clip_voronoi_boundaries(vertices: AbstractArray, regions: AbstractScalar | int, points: AbstractArray, point_region: AbstractArray, bbox: AbstractArray) -> AbstractScalar:
    """Ghost witness for clip_voronoi_boundaries."""
    _ = (vertices, regions, points, point_region, bbox)
    return AbstractScalar(dtype="float64")

