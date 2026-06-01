from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_knn_relations(X: AbstractArray, n_neighbors: AbstractScalar | int, metric: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for compute_knn_relations."""
    _ = (X, n_neighbors, metric)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_format_sparse_adjacency(distances: AbstractArray, indices: AbstractArray, mode: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for format_sparse_adjacency."""
    _ = (distances, indices, mode)
    return AbstractScalar(dtype="float64")

