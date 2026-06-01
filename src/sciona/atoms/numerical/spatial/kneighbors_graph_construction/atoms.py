from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_knn_relations,
    witness_format_sparse_adjacency,
)

@register_atom(witness_compute_knn_relations, name="compute_knn_relations")
@icontract.require(lambda X, n_neighbors, metric: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, n_neighbors, metric: n_neighbors > 0, "Precondition failed: n_neighbors > 0")
@icontract.require(lambda X, n_neighbors, metric: n_neighbors < X.shape[0], "Precondition failed: n_neighbors < X.shape[0]")
@icontract.ensure(lambda result, X, n_neighbors, metric: distances.shape[0] == X.shape[0], "Postcondition failed: distances.shape[0] == X.shape[0]")
def compute_knn_relations(X: NDArray[np.float64], n_neighbors: int, metric: str = None) -> NDArray[np.float64]:
    """Query nearest neighbors for all points to build relational maps.

    Args:
        X: NDArray[np.float64]
        n_neighbors: int
        metric: str

    Returns:
        distances: NDArray[np.float64]
    """
    import sklearn.neighbors
    return sklearn.neighbors.NearestNeighbors(X=X, n_neighbors=n_neighbors, metric=metric) # type: ignore

@register_atom(witness_format_sparse_adjacency, name="format_sparse_adjacency")
@icontract.require(lambda distances, indices, mode: distances.shape == indices.shape, "Precondition failed: distances.shape == indices.shape")
@icontract.ensure(lambda result, distances, indices, mode: graph.shape[0] == distances.shape[0], "Postcondition failed: graph.shape[0] == distances.shape[0]")
@icontract.ensure(lambda result, distances, indices, mode: graph.shape[1] == distances.shape[0], "Postcondition failed: graph.shape[1] == distances.shape[0]")
def format_sparse_adjacency(distances: NDArray[np.float64], indices: NDArray[np.int64], mode: str) -> Any:
    """Construct a Compressed Sparse Row (CSR) adjacency matrix from KNN indices.

    Args:
        distances: NDArray[np.float64]
        indices: NDArray[np.int64]
        mode: str

    Returns:
        graph: scipy.sparse.csr_matrix
    """
    import sklearn.neighbors
    return sklearn.neighbors.kneighbors_graph(distances=distances, indices=indices, mode=mode) # type: ignore

