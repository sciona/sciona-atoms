"""Graph traversal wrappers for the namespace pilot."""

from __future__ import annotations

import icontract
import numpy as np


@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0],),
    "BFS result must have one entry per node",
)
@icontract.ensure(lambda result: result.ndim == 1, "BFS result must be 1D")
def bfs(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Breadth-first search over a square adjacency matrix."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order

    graph = csr_matrix(adj)
    order, _ = breadth_first_order(graph, source, directed=True)
    result = np.full(adj.shape[0], -1, dtype=np.intp)
    result[: len(order)] = order
    return result


@icontract.require(
    lambda adj: adj.ndim == 2 and adj.shape[0] == adj.shape[1],
    "Adjacency matrix must be square",
)
@icontract.ensure(
    lambda result, adj: result.shape == (adj.shape[0],),
    "DFS result must have one entry per node",
)
@icontract.ensure(lambda result: result.ndim == 1, "DFS result must be 1D")
def dfs(adj: np.ndarray, source: int = 0) -> np.ndarray:
    """Depth-first search over a square adjacency matrix."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import depth_first_order

    graph = csr_matrix(adj)
    order, _ = depth_first_order(graph, source, directed=True)
    result = np.full(adj.shape[0], -1, dtype=np.intp)
    result[: len(order)] = order
    return result
