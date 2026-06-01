from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_isolation_trees(X: AbstractArray, n_estimators: AbstractScalar | int, max_samples: AbstractScalar | int) -> AbstractScalar:
    """Ghost witness for build_isolation_trees."""
    _ = (X, n_estimators, max_samples)
    return AbstractScalar(dtype="float64")

def witness_compute_path_lengths(forest: AbstractScalar | Any, query_points: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_path_lengths."""
    _ = (forest, query_points)
    return AbstractArray(shape=query_points.shape, dtype=query_points.dtype)

