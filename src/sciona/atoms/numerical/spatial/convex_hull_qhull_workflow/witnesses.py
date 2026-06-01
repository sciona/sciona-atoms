from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_convex_hull(points: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_convex_hull."""
    _ = (points)
    return AbstractArray(shape=points.shape, dtype=points.dtype)

def witness_verify_point_containment(points: AbstractArray, equations: AbstractArray) -> AbstractScalar:
    """Ghost witness for verify_point_containment."""
    _ = (points, equations)
    return AbstractScalar(dtype="float64")

