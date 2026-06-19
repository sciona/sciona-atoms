from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_solve_dominant_eigenvector(adj_matrix: AbstractArray, max_iter: AbstractScalar | int) -> Tuple[AbstractScalar, AbstractArray]:
    """Ghost witness for solve_dominant_eigenvector."""
    _ = (adj_matrix, max_iter)
    return AbstractScalar(dtype="complex128"), AbstractArray(shape=(adj_matrix.shape[0],), dtype="complex128")

def witness_perron_frobenius_correct(eigenvector: AbstractArray) -> AbstractArray:
    """Ghost witness for perron_frobenius_correct."""
    _ = (eigenvector)
    return AbstractArray(shape=eigenvector.shape, dtype="float64")


