from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_query_radius_neighbors,
    witness_assemble_radius_graph,
)

@register_atom(witness_query_radius_neighbors, name="query_radius_neighbors")
@icontract.require(lambda X, radius, metric: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.require(lambda X, radius, metric: radius > 0.0, "Precondition failed: radius > 0.0")
@icontract.ensure(lambda result, X, radius, metric: len(indices) == X.shape[0], "Postcondition failed: len(indices) == X.shape[0]")
def query_radius_neighbors(X: NDArray[np.float64], radius: float, metric: str = None) -> NDArray[np.object_]:
    """Perform range-search queries to identify all neighbors within a given metric distance.

    Args:
        X: NDArray[np.float64]
        radius: float
        metric: str

    Returns:
        distances: NDArray[np.object_]
    """
    import sklearn.neighbors.NearestNeighbors
    return sklearn.neighbors.NearestNeighbors.radius_neighbors(X=X, radius=radius, metric=metric) # type: ignore

@register_atom(witness_assemble_radius_graph, name="assemble_radius_graph")
@icontract.require(lambda distances, indices, radius, mode: len(distances) == len(indices), "Precondition failed: len(distances) == len(indices)")
@icontract.ensure(lambda result, distances, indices, radius, mode: graph.shape[0] == len(distances), "Postcondition failed: graph.shape[0] == len(distances)")
def assemble_radius_graph(distances: NDArray[np.object_], indices: NDArray[np.object_], radius: float, mode: str) -> Any:
    """Format the jagged neighborhood arrays into a standardized sparse CSR matrix.

    Args:
        distances: NDArray[np.object_]
        indices: NDArray[np.object_]
        radius: float
        mode: str

    Returns:
        graph: scipy.sparse.csr_matrix
    """
    import sklearn.neighbors
    return sklearn.neighbors.radius_neighbors_graph(distances=distances, indices=indices, radius=radius, mode=mode) # type: ignore

