from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_solve_optics_ordering(X: AbstractArray, min_samples: AbstractScalar | int, max_eps: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for solve_optics_ordering."""
    _ = (X, min_samples, max_eps)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_extract_optics_dbscan(ordering: AbstractArray, reachability: AbstractArray, eps: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for extract_optics_dbscan."""
    _ = (ordering, reachability, eps)
    return AbstractArray(shape=ordering.shape, dtype=ordering.dtype)

