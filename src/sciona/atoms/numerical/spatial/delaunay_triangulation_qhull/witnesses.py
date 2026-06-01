from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_delaunay(points: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_delaunay."""
    _ = (points)
    return AbstractArray(shape=points.shape, dtype=points.dtype)

def witness_validate_empty_circumspheres(points: AbstractArray, simplices: AbstractArray) -> AbstractScalar:
    """Ghost witness for validate_empty_circumspheres."""
    _ = (points, simplices)
    return AbstractScalar(dtype="float64")

