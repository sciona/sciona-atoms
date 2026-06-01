from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_linkage(X: AbstractArray, method: AbstractScalar | str, metric: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for compute_linkage."""
    _ = (X, method, metric)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_cut_tree(linkage_matrix: AbstractArray, n_clusters: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for cut_tree."""
    _ = (linkage_matrix, n_clusters)
    return AbstractArray(shape=linkage_matrix.shape, dtype=linkage_matrix.dtype)

