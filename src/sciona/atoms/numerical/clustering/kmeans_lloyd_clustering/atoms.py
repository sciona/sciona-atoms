from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_initialize_kmeans_plus_plus,
    witness_assign_clusters,
    witness_update_centroids,
)

@register_atom(witness_initialize_kmeans_plus_plus, name="initialize_kmeans_plus_plus")
@icontract.require(lambda X, n_clusters: n_clusters <= X.shape[0], "Precondition failed: n_clusters <= X.shape[0]")
@icontract.ensure(lambda result, X, n_clusters: initial_centroids.shape == (n_clusters, X.shape[1]), "Postcondition failed: initial_centroids.shape == (n_clusters, X.shape[1])")
def initialize_kmeans_plus_plus(X: NDArray[np.float64], n_clusters: int) -> NDArray[np.float64]:
    """Use distance-weighted probability distribution to spread out initial centroids.

    Args:
        X: NDArray[np.float64]
        n_clusters: int

    Returns:
        initial_centroids: NDArray[np.float64]
    """
    import sklearn.cluster
    return sklearn.cluster.kmeans_plusplus(X=X, n_clusters=n_clusters) # type: ignore

@register_atom(witness_assign_clusters, name="assign_clusters")
@icontract.require(lambda X, centroids: X.shape[1] == centroids.shape[1], "Precondition failed: X.shape[1] == centroids.shape[1]")
@icontract.ensure(lambda result, X, centroids: labels.shape[0] == X.shape[0], "Postcondition failed: labels.shape[0] == X.shape[0]")
def assign_clusters(X: NDArray[np.float64], centroids: NDArray[np.float64]) -> NDArray[np.int32]:
    """Assign each data point to its closest centroid.

    Args:
        X: NDArray[np.float64]
        centroids: NDArray[np.float64]

    Returns:
        labels: NDArray[np.int32]
    """
    import sklearn.cluster
    return sklearn.cluster.KMeans(X=X, centroids=centroids) # type: ignore

@register_atom(witness_update_centroids, name="update_centroids")
@icontract.require(lambda X, labels, n_clusters: n_clusters > 0, "Precondition failed: n_clusters > 0")
@icontract.ensure(lambda result, X, labels, n_clusters: centroids.shape == (n_clusters, X.shape[1]), "Postcondition failed: centroids.shape == (n_clusters, X.shape[1])")
def update_centroids(X: NDArray[np.float64], labels: NDArray[np.int32], n_clusters: int) -> NDArray[np.float64]:
    """Recalculate cluster centroids as the coordinate mean of assigned members.

    Args:
        X: NDArray[np.float64]
        labels: NDArray[np.int32]
        n_clusters: int

    Returns:
        centroids: NDArray[np.float64]
    """
    import sklearn.cluster
    return sklearn.cluster.KMeans(X=X, labels=labels, n_clusters=n_clusters) # type: ignore

