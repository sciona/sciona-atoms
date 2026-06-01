from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_initialize_kmeans_plus_plus(X: AbstractArray, n_clusters: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for initialize_kmeans_plus_plus."""
    _ = (X, n_clusters)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_assign_clusters(X: AbstractArray, centroids: AbstractArray) -> AbstractArray:
    """Ghost witness for assign_clusters."""
    _ = (X, centroids)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

def witness_update_centroids(X: AbstractArray, labels: AbstractArray, n_clusters: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for update_centroids."""
    _ = (X, labels, n_clusters)
    return AbstractArray(shape=X.shape, dtype=X.dtype)

