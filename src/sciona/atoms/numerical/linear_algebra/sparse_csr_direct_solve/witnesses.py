from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_sparse_factorize_matrix(A: AbstractScalar | Any) -> AbstractArray:
    """Ghost witness for sparse_factorize_matrix."""
    _ = (A)
    return AbstractArray(shape=(), dtype="float64")

def witness_sparse_solve_rhs(solve_fn: AbstractArray, b: AbstractArray) -> AbstractArray:
    """Ghost witness for sparse_solve_rhs."""
    _ = (solve_fn, b)
    return AbstractArray(shape=solve_fn.shape, dtype=solve_fn.dtype)

