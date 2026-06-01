from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_query_radius_neighbors(X: AbstractArray, radius: AbstractScalar | float, metric: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for query_radius_neighbors."""
    _ = (X, radius, metric)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_assemble_radius_graph(distances: AbstractArray, indices: AbstractArray, radius: AbstractScalar | float, mode: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for assemble_radius_graph."""
    _ = (distances, indices, radius, mode)
    return AbstractScalar(dtype="float64")

