from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_local_reachability_density(distances: AbstractArray, indices: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_local_reachability_density."""
    _ = (distances, indices)
    return AbstractArray(shape=distances.shape, dtype=distances.dtype)

def witness_score_outlier_factors(indices: AbstractArray, lrd: AbstractArray) -> AbstractArray:
    """Ghost witness for score_outlier_factors."""
    _ = (indices, lrd)
    return AbstractArray(shape=indices.shape, dtype=indices.dtype)

