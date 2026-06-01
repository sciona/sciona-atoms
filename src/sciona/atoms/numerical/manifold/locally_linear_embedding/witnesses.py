from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_solve_lle_weights(X: AbstractArray, indices: AbstractArray, reg: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for solve_lle_weights."""
    _ = (X, indices, reg)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_solve_lle_embedding(indices: AbstractArray, weights: AbstractArray, n_components: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for solve_lle_embedding."""
    _ = (indices, weights, n_components)
    return AbstractArray(shape=indices.shape, dtype=indices.dtype)

