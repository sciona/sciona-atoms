from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_shortest_paths,
    witness_classical_mds,
)

@register_atom(witness_solve_shortest_paths, name="solve_shortest_paths")
@icontract.require(lambda knn_graph: knn_graph.shape[0] == knn_graph.shape[1], "Precondition failed: knn_graph.shape[0] == knn_graph.shape[1]")
@icontract.ensure(lambda result, knn_graph: geodesics.ndim == 2, "Postcondition failed: geodesics.ndim == 2")
@icontract.ensure(lambda result, knn_graph: np.all(np.isfinite(geodesics)), "Postcondition failed: np.all(np.isfinite(geodesics))")
def solve_shortest_paths(knn_graph: Any) -> NDArray[np.float64]:
    """Execute Dijkstra/Floyd-Warshall to resolve complete geodesic distances on a sparse graph.

    Args:
        knn_graph: scipy.sparse.csr_matrix

    Returns:
        geodesics: NDArray[np.float64]
    """
    import scipy.sparse.csgraph
    return scipy.sparse.csgraph.shortest_path(knn_graph=knn_graph) # type: ignore

@register_atom(witness_classical_mds, name="classical_mds")
@icontract.require(lambda geodesics, n_components: n_components > 0, "Precondition failed: n_components > 0")
@icontract.ensure(lambda result, geodesics, n_components: embedding.shape == (geodesics.shape[0], n_components), "Postcondition failed: embedding.shape == (geodesics.shape[0], n_components)")
def classical_mds(geodesics: NDArray[np.float64], n_components: int) -> NDArray[np.float64]:
    """Perform classical Multi-dimensional Scaling (MDS) on geodesic distances.

    Args:
        geodesics: NDArray[np.float64]
        n_components: int

    Returns:
        embedding: NDArray[np.float64]
    """
    import sklearn.manifold
    return sklearn.manifold.MDS(geodesics=geodesics, n_components=n_components) # type: ignore

