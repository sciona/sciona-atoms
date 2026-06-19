from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_laplacian_matrix(adj_matrix: AbstractArray, normalized: AbstractScalar | bool) -> AbstractArray:
    """Ghost witness for compute_laplacian_matrix."""
    _ = (adj_matrix, normalized)
    return AbstractArray(shape=adj_matrix.shape, dtype=adj_matrix.dtype)

def witness_solve_smallest_eigen(laplacian: AbstractArray, k: AbstractScalar | int) -> Tuple[AbstractArray, AbstractArray]:
    """Ghost witness for solve_smallest_eigen."""
    _ = (laplacian, k)
    k_val = int(k) if isinstance(k, (int, float)) else 1
    return AbstractArray(shape=(k_val,), dtype="float64"), AbstractArray(shape=(laplacian.shape[0], k_val), dtype="float64")


