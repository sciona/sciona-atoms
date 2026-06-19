from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_hnsw_index,
    witness_query_hnsw_index,
)

@register_atom(witness_build_hnsw_index, name="build_hnsw_index")
@icontract.require(lambda data, M=None, ef_construction=None: data.ndim == 2)
@icontract.require(lambda data, M=None, ef_construction=None: M is None or M > 0)
@icontract.require(lambda data, M=None, ef_construction=None: ef_construction is None or ef_construction > 0)
@icontract.ensure(lambda result: result is not None)
def build_hnsw_index(
    data: NDArray[np.float64],
    M: int = 16,
    ef_construction: int = 200,
) -> Any:
    """Build a multi-layer hierarchical navigable small world graph index.

    Parameters
    ----------
    data : NDArray[np.float64]
        Matrix of points to index, of shape (n_samples, n_features).
    M : int, default 16
        Number of bi-directional links created for every new element.
    ef_construction : int, default 200
        Size of the dynamic candidate list for construction.

    Returns
    -------
    index : Any
        Built hnswlib.Index object.
    """
    import hnswlib
    if M is None:
        M = 16
    if ef_construction is None:
        ef_construction = 200

    dim = data.shape[1]
    num_elements = data.shape[0]

    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(
        max_elements=num_elements, M=M, ef_construction=ef_construction
    )
    index.add_items(data)
    return index

@register_atom(witness_query_hnsw_index, name="query_hnsw_index")
@icontract.require(lambda index, query_points, k=None, ef_search=None: query_points.ndim == 2)
@icontract.require(lambda index, query_points, k=None, ef_search=None: k is None or k >= 1)
@icontract.require(lambda index, query_points, k=None, ef_search=None: ef_search is None or ef_search > 0)
@icontract.ensure(lambda result, query_points, k=None: result[0].shape == (query_points.shape[0], k if k is not None else 10))
@icontract.ensure(lambda result, query_points, k=None: result[1].shape == (query_points.shape[0], k if k is not None else 10))
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])))
def query_hnsw_index(
    index: Any,
    query_points: NDArray[np.float64],
    k: int = 10,
    ef_search: int = 50,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Query the built HNSW graph for fast approximate nearest neighbors.

    Parameters
    ----------
    index : Any
        Built hnswlib.Index object.
    query_points : NDArray[np.float64]
        Matrix of query points of shape (n_queries, n_features).
    k : int, default 10
        Number of nearest neighbors to query.
    ef_search : int, default 50
        Dynamic candidate list size for search.

    Returns
    -------
    distances : NDArray[np.float64]
        Approximate distances to neighbors, of shape (n_queries, k).
    indices : NDArray[np.int64]
        Approximate indices of neighbors, of shape (n_queries, k).
    """
    if k is None:
        k = 10
    if ef_search is None:
        ef_search = 50

    index.set_ef(ef_search)
    labels, distances = index.knn_query(query_points, k=k)
    return distances.astype(np.float64), labels.astype(np.int64)


